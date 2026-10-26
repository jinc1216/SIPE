import datetime
import logging
import math
import time
import torch
import os
import shutil
from os import path as osp
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

try:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
except ImportError:  # pragma: no cover - optional training dependency
    Accelerator = None
    DistributedDataParallelKwargs = None


def _wait_for_all_processes(accelerator, opt):
    """Keep ranks aligned around main-process-only I/O/validation."""
    if accelerator is not None:
        accelerator.wait_for_everyone()
    elif opt.get('dist', False) and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _scale_positive_number(value, scale):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if value <= 0:
        return value
    return int(round(value * scale))


def _scale_scheduler_opt(scheduler_opt, scale):
    if not isinstance(scheduler_opt, dict):
        return
    for key in ('milestones', 'restarts', 'periods'):
        if key in scheduler_opt:
            value = scheduler_opt[key]
            if isinstance(value, (list, tuple)):
                scheduler_opt[key] = [_scale_positive_number(v, scale) for v in value]
            else:
                scheduler_opt[key] = _scale_positive_number(value, scale)


def _scale_iter_options(opt, scale):
    """Convert optimizer-update based config values to micro-iteration values."""
    if scale == 1:
        return

    train_opt = opt.get('train', {})
    opt['gradient_accumulation_original_train'] = {}

    def scale_dict_values(cfg, path='train'):
        for key, value in list(cfg.items()):
            cur_path = f'{path}.{key}'
            if isinstance(value, dict):
                scale_dict_values(value, cur_path)
                continue
            should_scale = (
                key in ('total_iter', 'warmup_iter', 'start_iter', 'net_d_init_iters', 'net_d_start_iter') or
                (key.endswith('_iter') and key != 'gradient_accumulation_steps')
            )
            if should_scale:
                opt['gradient_accumulation_original_train'][cur_path] = value
                cfg[key] = _scale_positive_number(value, scale)

    scale_dict_values(train_opt)
    _scale_scheduler_opt(train_opt.get('scheduler'), scale)

    for section_name, keys in (
        ('val', ('val_freq',)),
        ('logger', ('print_freq', 'save_checkpoint_freq', 'save_latest_freq', 'show_tf_imgs_freq')),
    ):
        section = opt.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key in section:
                opt['gradient_accumulation_original_train'][f'{section_name}.{key}'] = section[key]
                section[key] = _scale_positive_number(section[key], scale)


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)



def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    has_grad_accum_opt = 'gradient_accumulation_steps' in opt['train']
    grad_accum_steps = int(opt['train'].get('gradient_accumulation_steps', 1))
    if grad_accum_steps < 1:
        raise ValueError('train:gradient_accumulation_steps must be >= 1.')
    _scale_iter_options(opt, grad_accum_steps)
    use_accelerate = has_grad_accum_opt
    if use_accelerate:
        if Accelerator is None:
            raise ImportError(
                'train:gradient_accumulation_steps requires accelerate. '
                'Install it with `pip install accelerate` or remove gradient_accumulation_steps.')
        if opt['dist']:
            raise ValueError(
                'gradient_accumulation_steps uses Accelerate in this repo, including when it is set to 1. '
                'Launch with `accelerate launch basicsr/train.py -opt ...` instead of `--launcher pytorch`.')
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=opt.get('find_unused_parameters', False))
        mixed_precision = 'no'
        if opt['train'].get('use_amp', False):
            amp_dtype = str(opt['train'].get('amp_dtype', 'float16')).lower()
            if amp_dtype in ('float16', 'fp16'):
                mixed_precision = 'fp16'
            elif amp_dtype in ('bfloat16', 'bf16'):
                mixed_precision = 'bf16'
        accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision=mixed_precision)
        verbose = accelerator.is_main_process
        opt['accelerate_num_processes'] = accelerator.num_processes
        opt['configured_num_gpu'] = opt.get('num_gpu', 0)
        if opt['num_gpu'] != 0:
            opt['num_gpu'] = 1
    else:
        accelerator = None
        verbose = opt['rank'] == 0

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and verbose:
            os.makedirs(osp.join(opt['root_path'], 'tb_logger_archived'), exist_ok=True)
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    if use_accelerate:
        logger.info(
            f'Use Accelerate gradient accumulation: {grad_accum_steps} micro-batches per optimizer step; '
            f'mixed_precision={mixed_precision}.')
        logger.info('Iteration and frequency options in the YAML are treated as optimizer-update counts '
                    'and have been scaled internally to micro-iterations.')
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
    if accelerator is not None:
        train_loader = accelerator.prepare(train_loader)
        num_iter_per_epoch = len(train_loader)
        total_epochs = math.ceil(total_iters / num_iter_per_epoch)
        logger.info('Accelerate dataloader statistics:'
                    f'\n\tNumber of processes: {accelerator.num_processes}'
                    f'\n\tMicro-batches per optimizer step: {grad_accum_steps}'
                    f'\n\tLocal iter number per epoch: {num_iter_per_epoch}'
                    f'\n\tAdjusted total epochs: {total_epochs}; iters: {total_iters}.')

    # create model
    model = build_model(opt)
    if accelerator is not None:
        if not hasattr(model, 'set_acc'):
            raise AttributeError(f'{model.__class__.__name__} does not implement set_acc for gradient accumulation.')
        model.set_acc(accelerator)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.' "Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            if accelerator is not None:
                with accelerator.accumulate(*model.get_accumulation_models()):
                    model.optimize_parameters(current_iter)
            else:
                model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            # log
            if current_iter % opt['logger']['print_freq'] == 0 and verbose:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save images
            if current_iter % (opt['logger']['show_tf_imgs_freq']) == 0 and verbose:
                visual_imgs = model.get_current_visuals()
                if tb_logger:
                    for k, v in visual_imgs.items(): 
                        tb_logger.add_images(f'ckpt_imgs/{k}', v.clamp(0, 1), current_iter)

            # save models and training states
            save_checkpoint = current_iter % opt['logger']['save_checkpoint_freq'] == 0
            if save_checkpoint:
                _wait_for_all_processes(accelerator, opt)
                if verbose:
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter)
                _wait_for_all_processes(accelerator, opt)

            save_latest = current_iter % opt['logger']['save_latest_freq'] == 0
            if save_latest:
                _wait_for_all_processes(accelerator, opt)
                if verbose:
                    logger.info('Saving models and training states.')
                    model.save(epoch, -1)
                _wait_for_all_processes(accelerator, opt)

            # validation
            do_validation = opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0)
            if do_validation:
                _wait_for_all_processes(accelerator, opt)
                if len(val_loaders) > 1 and verbose:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(
                        val_loader,
                        current_iter,
                        tb_logger if verbose else None,
                        opt['val']['save_img'] and verbose)
                _wait_for_all_processes(accelerator, opt)

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    _wait_for_all_processes(accelerator, opt)
    if verbose:
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f'End of training. Time consumed: {consumed_time}')
        logger.info('Save the latest model.')
        model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    _wait_for_all_processes(accelerator, opt)
    if opt.get('val') is not None:
        _wait_for_all_processes(accelerator, opt)
        for val_loader in val_loaders:
            model.validation(
                val_loader,
                current_iter,
                tb_logger if verbose else None,
                opt['val']['save_img'] and verbose)
        _wait_for_all_processes(accelerator, opt)
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)

