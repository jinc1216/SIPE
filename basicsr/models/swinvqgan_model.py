"""Stage-1 SwinVQGAN model used by SIPE-SR.

This is adapted from ITER's SwinVQGAN training model, but keeps compatibility
with the non-accelerate BasicSR trainer.  If accelerate is
used externally, ``set_acc`` still works as an optional path.
"""

from collections import OrderedDict
from os import path as osp
import copy

import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.dino_semantic import DINOSemanticLoss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

try:
    import pyiqa  # type: ignore
except Exception:
    pyiqa = None


@MODEL_REGISTRY.register()
class SwinVQGANModel(BaseModel):
    """Train ITER-style SwinVQGAN as SIPE-SR's high-quality codebook prior."""

    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.net_d = None
        self.use_dis = False

        self.metric_funcs = {}
        if self.opt.get('val', {}).get('metrics') is not None:
            for _, opt_m in self.opt['val']['metrics'].items():
                mopt = opt_m.copy()
                name = mopt.pop('type', None)
                mopt.pop('better', None)
                if pyiqa is not None:
                    self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)
                else:
                    get_root_logger().warning(f'Metric {name} requires pyiqa and is disabled.')

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            get_root_logger().info(f'Loading net_g from {load_path}')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load', False))

        self.cri_dino = None
        if self.is_train:
            self.init_training_settings()
            self.use_dis = bool(
                self.net_d is not None and self.cri_gan is not None and
                self.opt['train'].get('gan_opt') and self.opt['train']['gan_opt'].get('loss_weight', 0) != 0)
            self.net_d_best = copy.deepcopy(self.net_d) if self.net_d is not None else None
        self.net_g_best = copy.deepcopy(self.net_g)

    def _safe_build_loss(self, opt, name):
        if not opt:
            return None
        try:
            loss = build_loss(opt).to(self.device)
            self.model_to_device(loss)
            return loss
        except Exception as exc:
            get_root_logger().warning(f'Disable {name}: failed to build loss from {opt}: {exc}')
            return None

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.setup_amp(train_opt, get_root_logger())
        self.net_g.train()
        self.net_d = build_network(self.opt['network_d']) if self.opt.get('network_d') else None
        if self.net_d is not None:
            self.net_d = self.model_to_device(self.net_d)
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', False))
            self.net_d.train()

        self.cri_pix = self._safe_build_loss(train_opt.get('pixel_opt'), 'pixel loss')
        self.cri_perceptual = self._safe_build_loss(train_opt.get('perceptual_opt'), 'perceptual loss')
        self.cri_gan = self._safe_build_loss(train_opt.get('gan_opt'), 'GAN loss')
        self.cri_struct = None
        if train_opt.get('struct_opt') and pyiqa is not None:
            self.cri_struct = pyiqa.create_metric(train_opt['struct_opt']['type'], as_loss=True).to(self.device)
        elif train_opt.get('struct_opt'):
            get_root_logger().warning('struct_opt is configured but pyiqa is unavailable; structural loss is disabled.')

        if train_opt.get('dino_opt'):
            dino_opt = train_opt['dino_opt'].copy()
            self.cri_dino = DINOSemanticLoss(**dino_opt).to(self.device)
            self.cri_dino.eval()
            get_root_logger().info(f'DINO semantic loss is enabled: {dino_opt}')
        else:
            self.cri_dino = None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        params = [p for p in self.net_g.parameters() if p.requires_grad]
        if self.cri_dino is not None and hasattr(self.cri_dino, 'trainable_parameters'):
            dino_params = list(self.cri_dino.trainable_parameters())
            if dino_params:
                params.extend(dino_params)
                dino_param_count = sum(p.numel() for p in dino_params)
                get_root_logger().info(f'{dino_param_count:,} DINO latent-projector parameters will be optimized.')
        optim_g_opt = train_opt['optim_g'].copy()
        optim_type = optim_g_opt.pop('type')
        self.optimizer_g = getattr(torch.optim, optim_type)(params, **optim_g_opt)
        self.optimizers.append(self.optimizer_g)
        if self.net_d is not None:
            optim_d_opt = train_opt['optim_d'].copy()
            optim_type = optim_d_opt.pop('type')
            self.optimizer_d = getattr(torch.optim, optim_type)(self.net_d.parameters(), **optim_d_opt)
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['gt'].to(self.device)
        self.gt = self.lq

    def set_acc(self, acc):
        self.accelerator = acc
        prepare_items = [('net_g', self.net_g)]
        if self.cri_dino is not None and hasattr(self.cri_dino, 'latent_projector'):
            latent_projector = self.cri_dino.latent_projector
            dino_params = [p for p in latent_projector.parameters() if p.requires_grad]
            if dino_params:
                prepare_items.append(('cri_dino.latent_projector', latent_projector))
        if self.net_d is not None:
            prepare_items.append(('net_d', self.net_d))
        prepare_items.append(('optimizer_g', self.optimizer_g))
        if self.net_d is not None:
            prepare_items.append(('optimizer_d', self.optimizer_d))

        prepared = acc.prepare(*[obj for _, obj in prepare_items])
        for (name, _), obj in zip(prepare_items, prepared):
            if name == 'cri_dino.latent_projector':
                self.cri_dino.latent_projector = obj
            else:
                setattr(self, name, obj)

    def _backward(self, loss):
        if hasattr(self, 'accelerator'):
            self.accelerator.backward(loss)
        else:
            self.amp_backward(loss)

    @staticmethod
    def _set_requires_grad(net, requires_grad):
        if net is None:
            return
        for p in net.parameters():
            p.requires_grad = requires_grad

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        self._set_requires_grad(self.net_d, False)

        need_dino_latent = self.cri_dino is not None and getattr(self.cri_dino, 'latent_weight', 0) > 0
        model_input = self.gt
        mask_opt = train_opt.get('dino_mask_opt', None)
        use_masked_reconstruction = self.cri_dino is not None and mask_opt is not None and mask_opt.get('enabled', False)
        dino_target_cache = None
        if self.cri_dino is not None:
            need_dino_image = getattr(self.cri_dino, 'global_weight', 0) > 0 or getattr(self.cri_dino, 'local_weight', 0) > 0
            if use_masked_reconstruction or need_dino_image or need_dino_latent:
                dino_target_cache = self.cri_dino.prepare_features(self.gt)
        if use_masked_reconstruction:
            mask = self.cri_dino.semantic_mask(
                self.gt,
                mask_ratio=mask_opt.get('mask_ratio', 0.30),
                mode=mask_opt.get('mode', 'semantic'),
                random_ratio=mask_opt.get('random_ratio', 0.0),
                semantic_prob=mask_opt.get('semantic_prob', None),
                feature_cache=dino_target_cache,
            )
            model_input = self.cri_dino.apply_mask(self.gt, mask, fill=mask_opt.get('fill', 0.5))
            self.masked_gt = model_input.detach()
        else:
            if hasattr(self, 'masked_gt'):
                delattr(self, 'masked_gt')
            if hasattr(self, 'masked_output'):
                delattr(self, 'masked_output')

        with self.amp_autocast():
            if need_dino_latent:
                self.output, l_codebook, _, quant_feat = self.net_g(
                    model_input, return_pred_indices=True, return_quant=True)
            else:
                self.output, l_codebook = self.net_g(model_input)
                quant_feat = None
        if use_masked_reconstruction:
            self.masked_output = self.output
        l_g_total = 0
        loss_dict = OrderedDict()

        if train_opt.get('codebook_opt'):
            l_codebook = l_codebook.float().mean() * train_opt['codebook_opt'].get('loss_weight', 1.0)
            l_g_total = l_g_total + l_codebook
            loss_dict['l_codebook'] = l_codebook

        if self.cri_pix:
            with self.amp_autocast():
                l_pix = self.cri_pix(self.output, self.gt)
            l_g_total = l_g_total + l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_struct:
            with self.amp_autocast():
                l_struct = (1 - self.cri_struct(self.output, self.gt).mean()) * train_opt['struct_opt']['loss_weight']
            l_g_total = l_g_total + l_struct
            loss_dict['l_struct'] = l_struct

        if self.cri_perceptual:
            with self.amp_autocast():
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_g_total = l_g_total + l_percep.mean()
                loss_dict['l_percep'] = l_percep.mean()
            if l_style is not None:
                l_g_total = l_g_total + l_style
                loss_dict['l_style'] = l_style

        if self.cri_dino is not None:
            dino_losses = self.cri_dino(self.output, self.gt, latent=quant_feat, target_cache=dino_target_cache)
            for k, v in dino_losses.items():
                l_g_total = l_g_total + v
                loss_dict[k] = v

        if self.net_d is not None and self.use_dis and current_iter > self.net_d_start_iter:
            with self.amp_autocast():
                fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total = l_g_total + l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        self._backward(l_g_total.mean())
        self.amp_optimizer_step(self.optimizer_g)
        self.optimizer_g.zero_grad()

        if self.net_d is not None and not train_opt.get('fixed_disc', False) and self.use_dis and current_iter > self.net_d_start_iter:
            self._set_requires_grad(self.net_d, True)
            with self.amp_autocast():
                real_d_pred = self.net_d(self.gt.float())
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            self._backward(l_d_real)
            with self.amp_autocast():
                fake_d_pred = self.net_d(self.output.float().detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            self._backward(l_d_fake)
            self.amp_optimizer_step(self.optimizer_d)
            self.optimizer_d.zero_grad()

        self.amp_scaler_update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def test(self):
        self.net_g.eval()
        net_g = self.get_bare_model(self.net_g)
        self.output = net_g.test(self.lq)
        self.net_g.train()
        return self.output

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        get_root_logger().info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt.get('val', {}).get('metrics') is not None and len(self.metric_funcs) > 0
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_funcs.keys()}
            self._initialize_best_metric_results(dataset_name)
        pbar = tqdm(total=len(dataloader), unit='image')
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            sr_img = tensor2img(self.output)
            if save_img:
                save_img_path = osp.join(self.opt['path']['visualization'], 'image_results', f'{current_iter}', f'{img_name}.png') \
                    if self.opt['is_train'] else osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{self.opt["name"]}.png')
                if save_as_dir:
                    imwrite(sr_img, osp.join(save_as_dir, f'{img_name}.png'))
                imwrite(sr_img, save_img_path)
            if with_metrics:
                pred = img2tensor(sr_img).unsqueeze(0).to(self.device) / 255.0
                for name, func in self.metric_funcs.items():
                    self.metric_results[name] += func(pred, self.gt).item()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            key_metric = self.opt.get('val', {}).get('key_metric')
            if key_metric is not None:
                if key_metric in self.metric_results:
                    best_updated = self._update_best_metric_result(
                        dataset_name, key_metric, self.metric_results[key_metric], current_iter)
                    if best_updated:
                        for metric, value in self.metric_results.items():
                            self._update_metric_result(dataset_name, metric, value, current_iter)
                        self.copy_model(self.net_g, self.net_g_best)
                        self.save_network(self.net_g, 'net_g_best', '')
                else:
                    get_root_logger().warning(
                        f'key_metric={key_metric} is not in validation metrics: '
                        f'{list(self.metric_results.keys())}. Skip best checkpoint update.')
            else:
                best_updated = False
                for metric, value in self.metric_results.items():
                    best_updated |= self._update_best_metric_result(dataset_name, metric, value, current_iter)
                if best_updated:
                    self.copy_model(self.net_g, self.net_g_best)
                    self.save_network(self.net_g, 'net_g_best', '')

            log_str = f'Validation {dataset_name}\n'
            for metric, value in self.metric_results.items():
                log_str += f'\t # {metric}: {value:.4f}'
                if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
                    best_info = self.best_metric_results[dataset_name].get(metric)
                    if best_info is not None:
                        log_str += f'\tBest: {best_info["val"]:.4f} @ {best_info["iter"]} iter'
                log_str += '\n'
                if tb_logger:
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
            get_root_logger().info(log_str)

    @torch.no_grad()
    def get_current_visuals(self):
        vis_samples = 4
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        if hasattr(self, 'output'):
            out_dict['output'] = self.output.detach().cpu()[:vis_samples]
        if hasattr(self, 'masked_gt'):
            out_dict['masked_gt'] = self.masked_gt.detach().cpu()[:vis_samples]
        if hasattr(self, 'masked_output'):
            out_dict['masked_output'] = self.masked_output.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        return OrderedDict((k, v.float().clamp(0, 1)) for k, v in out_dict.items())

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
