"""Training models for SIPE-SR.

The models here implement the SIPE-SR three-stage training protocol:
  1) SwinVQGANModel trains the HQ prior/codebook.
  2) SIPEPredictorModel trains the LR-to-token predictor and HR refiner.
  3) SIPEEvaluatorModel trains the token evaluator used by iterative decoding.
"""

from collections import OrderedDict
from os import path as osp
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.archs.sipe_arch import SIPEEvaluator
from basicsr.losses import build_loss
from basicsr.losses.dino_semantic import DINOSemanticLoss
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.utils.mask_schedule import schedule
from basicsr.utils.parallel_decode import (
    cosine_remask_count,
    decode_evaluator,
    resolve_num_refine_steps,
    resolve_token_update_mode,
    select_topk_remask,
    select_predictor_tokens,
    correctness_transition_target,
    transition_error_probability,
    transition_probabilities,
    accept_token_gains,
    propose_evaluator_tokens,
)

try:
    import pyiqa  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    pyiqa = None


def _bare(model: BaseModel, net):
    return model.get_bare_model(net)


def _synchronize_hq_prior(model: BaseModel, net_hq, net_g) -> None:
    """Make GT tokenization and predictor decoding use one identical VQ prior.

    Predictor checkpoints normally contain a frozen VQ branch.  Loading that
    checkpoint after ``pretrain_network_hq`` could silently overwrite only the
    predictor copy and make GT token IDs incompatible with its decoder.  The
    caller establishes the desired source-of-truth order first, then this helper
    mirrors the predictor's final VQ state into the standalone HQ tokenizer.
    """
    source = _bare(model, net_g).vqgan
    target = _bare(model, net_hq)
    target.load_state_dict(source.state_dict(), strict=True)


def _vq_indices_and_feats(vqgan, gt: torch.Tensor):
    """Encode GT with SwinVQGAN and return rec, target features, indices."""
    rec, _, indices = vqgan(gt, return_pred_indices=True)
    if indices.dim() == 3:
        indices = indices.unsqueeze(1)
    b, _, h, w = indices.shape
    shape = [b, h, w, vqgan.embed_dim]
    feats = vqgan.quantize.get_codebook_entry(indices.long(), shape)
    return rec, feats, indices.long()


def _ramp_multiplier(opt: Optional[dict], current_iter: int) -> float:
    """Linear warmup multiplier for heavy semantic losses."""
    if not opt:
        return 1.0
    start_iter = int(opt.get('start_iter', 0))
    if current_iter < start_iter:
        return 0.0
    warmup_iter = int(opt.get('warmup_iter', 0))
    if warmup_iter <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (current_iter - start_iter + 1) / float(warmup_iter))))


def _soft_decode_settings(train_opt: dict, current_iter: int):
    """Training-only differentiable token decoding options for Stage-2 predictor."""
    opt = train_opt.get('soft_decode_opt', {}) or {}
    enabled = bool(opt.get('enabled', train_opt.get('soft_decode', True)))
    tau = float(opt.get('tau', train_opt.get('soft_decode_tau', 1.0)))
    tau_min = float(opt.get('tau_min', tau))
    anneal_iter = int(opt.get('anneal_iter', 0))
    if anneal_iter > 0:
        progress = min(1.0, max(0.0, current_iter / float(anneal_iter)))
        tau = tau + (tau_min - tau) * progress
    tau = max(tau, 1e-6)
    straight_through = bool(opt.get('straight_through', True))
    return enabled, tau, straight_through


def _random_token_mask(b: int, h: int, w: int, num_lq_positions: int, device) -> torch.Tensor:
    """Sample a random LQ-context token mask."""
    total = h * w
    num_lq_positions = int(max(0, min(total, num_lq_positions)))
    mask = torch.zeros(b, total, device=device, dtype=torch.bool)
    if num_lq_positions > 0:
        samples = torch.rand(b, total, dtype=torch.float, device=device).topk(num_lq_positions, dim=1).indices
        mask.scatter_(dim=1, index=samples, value=True)
    return mask.reshape(b, 1, h, w)


def _score_guided_token_mask(score: torch.Tensor, num_lq_positions: int, random_ratio: float = 0.0) -> torch.Tensor:
    """Select the requested number of token positions from a semantic score map."""
    b, _, h, w = score.shape
    total = h * w
    num_lq_positions = int(max(0, min(total, num_lq_positions)))
    mask = torch.zeros(b, total, device=score.device, dtype=torch.bool)
    if num_lq_positions <= 0:
        return mask.reshape(b, 1, h, w)
    flat_score = score.flatten(1).float()
    random_ratio = float(max(0.0, min(1.0, random_ratio)))
    if random_ratio > 0:
        flat_score = (1.0 - random_ratio) * flat_score + random_ratio * torch.rand_like(flat_score)
    samples = torch.topk(flat_score, k=num_lq_positions, dim=1).indices
    mask.scatter_(dim=1, index=samples, value=True)
    return mask.reshape(b, 1, h, w)




def _flip_pad_to_multiple(x: torch.Tensor, multiple: int):
    """Flip-pad an image tensor so the LR shape is divisible by ``multiple``."""
    _, _, h_old, w_old = x.shape
    h_pad = (multiple - h_old % multiple) % multiple
    w_pad = (multiple - w_old % multiple) % multiple
    if h_pad > 0:
        target_h = h_old + h_pad
        while x.shape[2] < target_h:
            x = torch.cat([x, torch.flip(x, [2])], dim=2)
        x = x[:, :, :target_h, :]
    if w_pad > 0:
        target_w = w_old + w_pad
        while x.shape[3] < target_w:
            x = torch.cat([x, torch.flip(x, [3])], dim=3)
        x = x[:, :, :, :target_w]
    return x, h_old, w_old


def _weighted_mean(loss_map: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight is None:
        return loss_map.mean()
    while weight.dim() < loss_map.dim():
        weight = weight.unsqueeze(1)
    weight = weight.to(loss_map).clamp_min(0)
    return (loss_map * weight).sum() / weight.sum().clamp_min(1e-6)


def _evaluator_refinement_options(train_opt):
    """Read the one-step evaluator training action settings."""
    return train_opt.get('evaluator_refinement_opt', {}) or {}


def _candidate_action_schedule(total):
    """Draw one continuous-cosine training budget, independently of validation."""
    sample_r = float(np.random.uniform())
    return {
        'action_sample_r': sample_r,
        'action_mask_count': min(total - 1, max(
            0, math.floor(total * math.cos(math.pi * sample_r / 2)))),
    }


@torch.no_grad()
def _initial_evaluator_state(net_g, lq_feats, val_opt=None, *, train_opt=None):
    """Use detached LR-only c0 with configured token selection, without completion."""
    _, _, h, w = lq_feats.shape
    train_opt = train_opt or {}
    mode = (train_opt.get('evaluator_sampling_opt', {}) or {}).get('mode', 'multinomial')
    probs, ids = select_predictor_tokens(
        net_g.predict_logits(lq_feats), mode,
        float(train_opt.get('sample_temperature', 1.0)), step=0)
    return {'token_ids': ids, 'probs': probs, 'prefix_predictor_calls': 1,
            **_candidate_action_schedule(h * w)}


@torch.no_grad()
def _evaluator_context_mask(lq_feats, train_opt, *, lq=None, gt=None,
                            semantic_teacher=None, target_cache=None, current_iter=0):
    """Stage-2 mask policy, shared by GT and self context: LR/HR gap + noise."""
    b, _, h, w = lq_feats.shape
    count = min(h * w - 1, max(0, math.floor(schedule(
        np.random.uniform(), h * w,
        method=train_opt.get('token_schedule', 'cosine')) * h * w)))
    input_opt = train_opt.get('evaluator_input_opt', {}) or {}
    guidance_opt = train_opt.get('dino_guidance_opt', {}) or {}
    score = None
    if input_opt.get('mask_guided', False) and _ramp_multiplier(guidance_opt, current_iter) > 0:
        if semantic_teacher is None or lq is None or gt is None:
            raise ValueError('Guided context masks require DINO, LR and HR images.')
        reference = reference_cache = None
        if float(guidance_opt.get('difficulty_weight', 1.0)) > 0:
            reference = F.interpolate(
                lq.float(), size=gt.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
            reference_cache = semantic_teacher.prepare_features(reference)
        score = semantic_teacher.token_score_map(
            gt, (h, w), reference=reference, target_cache=target_cache,
            reference_cache=reference_cache,
            saliency_weight=guidance_opt.get('saliency_weight', 0.0),
            difficulty_weight=guidance_opt.get('difficulty_weight', 1.0))
    if score is None:
        return _random_token_mask(b, h, w, count, lq_feats.device)
    return _score_guided_token_mask(
        score, count, random_ratio=guidance_opt.get('mask_random_ratio', 0.15))


@torch.no_grad()
def _gt_context_evaluator_state(net_g, lq_feats, gt_indices, train_opt, val_opt, *,
                                lq=None, gt=None, semantic_teacher=None,
                                target_cache=None, current_iter=0):
    """Predict every candidate ID from Stage-2 LR/GT-vector mixed inputs.

    GT context is used only here; the next reconstruction action uses the
    candidate's own vectors and an independent r~Uniform(0,1) cosine budget.
    """
    b, _, h, w = lq_feats.shape
    if tuple(gt_indices.shape) != (b, 1, h, w):
        raise ValueError('GT-context token IDs must align with LR features.')
    gt_feats = net_g.tokens_to_feats(gt_indices).to(lq_feats)
    token_mask = _evaluator_context_mask(
        lq_feats, train_opt, lq=lq, gt=gt, semantic_teacher=semantic_teacher,
        target_cache=target_cache, current_iter=current_iter)
    mixed = torch.where(token_mask, lq_feats, gt_feats)
    mode = (train_opt.get('evaluator_sampling_opt', {}) or {}).get('mode', 'multinomial')
    probs, ids = select_predictor_tokens(
        net_g.predict_logits(mixed), mode,
        float(train_opt.get('sample_temperature', 1.0)), step=1)

    return {
        'token_ids': ids, 'probs': probs, 'source_mask': token_mask.flatten(1),
        'prefix_predictor_calls': 1,
        **_candidate_action_schedule(h * w),
    }


@torch.no_grad()
def _rollout_evaluator_state(net_g, evaluator, lq_feats, val_opt):
    """Uniformly sample a deployed c_t, t in [0,T-1], using an eval-mode critic.

    Reuse inference decoding for the detached prefix, retaining the full T-step
    schedule. Only the subsequent critic call/action receives training gradients.
    No HR image, GT token or DINO feature participates in rollout selection.
    """
    horizon = resolve_num_refine_steps(val_opt)
    if horizon < 1:
        raise ValueError('Rollout training requires val.num_refine_steps >= 1.')
    requested_step = int(np.random.randint(horizon))
    b, _, h, w = lq_feats.shape

    def tokens_to_logits(x, h_=0, w_=0, evaluator_call=False, cond_feats=None, proposal_ids=None):
        if evaluator_call:
            return evaluator(x, h_, w_, cond_feats=cond_feats if evaluator.use_cond_feats else None,
                             proposal_ids=proposal_ids)
        return net_g.predict_logits(x)

    _, _, rollout = decode_evaluator(
        mask_tokens=torch.full((b, h * w), -1, dtype=torch.long, device=lq_feats.device),
        lq_feats=lq_feats, tokens_to_logits=tokens_to_logits, tokens_to_feats=net_g.tokens_to_feats,
        num_refine_steps=horizon, max_refine_steps=requested_step, return_state=True,
        token_selection=val_opt.get('decode_sampling', val_opt.get('token_selection', 'multinomial')),
        sample_temperature=float(val_opt.get('sample_temperature', 1.0)),
        remask_budget_schedule=val_opt.get('remask_budget_schedule'),
        remask_budget_scale=val_opt.get('remask_budget_scale', 1.0),
        token_update_mode=resolve_token_update_mode(val_opt))
    # Tiny grids/zero budgets can stop before the sampled state. Use the actual
    # terminal state and its zero next-action budget, as inference would.
    state_step = rollout['state_step']
    return {
        'token_ids': rollout['token_ids'].detach(),
        'state_step': state_step, 'sampled_state_step': requested_step,
        'sampling_num_iter': horizon + 1,
        'action_mask_count': cosine_remask_count(
            h * w, state_step, horizon, val_opt.get('remask_budget_scale', 1.0)),
        'prefix_predictor_calls': int(rollout['diagnostics']['predictor_calls'][0].item()),
        'prefix_evaluator_calls': int(rollout['diagnostics']['evaluator_calls'][0].item()),
    }


@torch.no_grad()
def _self_context_evaluator_state(net_g, lq_feats, train_opt, *, lq=None, gt=None,
                                  semantic_teacher=None, cond_feats=None, target_cache=None,
                                  current_iter=0, val_opt=None):
    """Stage-2 self context with the same LR/HR mask policy as GT context.

    Obtain c0 using configured selection on LR-only logits, then use the Stage-2 cosine
    mask count in [0, N-1]. Guided masks rank LR-upsample/HR DINO gaps plus random
    noise; otherwise positions are uniform random. HR guides positions only:
    context vectors come from c0, and every candidate ID is newly predicted.
    The next action independently draws r~Uniform(0,1) for its cosine budget;
    validation step counts and budget scales do not affect this training draw.
    """
    b, _, h, w = lq_feats.shape
    logits_lr = net_g.predict_logits(lq_feats)
    mode = (train_opt.get('evaluator_sampling_opt', {}) or {}).get('mode', 'multinomial')
    probs_lr, initial_ids = select_predictor_tokens(
        logits_lr, mode, float(train_opt.get('sample_temperature', 1.0)), step=0)
    initial_feats = net_g.tokens_to_feats(initial_ids.reshape(b, 1, h, w)).to(lq_feats)
    token_mask = _evaluator_context_mask(
        lq_feats, train_opt, lq=lq, gt=gt, semantic_teacher=semantic_teacher,
        target_cache=target_cache, current_iter=current_iter)
    mixed = torch.where(token_mask, lq_feats, initial_feats)
    probs, ids = select_predictor_tokens(
        net_g.predict_logits(mixed), mode,
        float(train_opt.get('sample_temperature', 1.0)), step=1)
    return {
        'token_ids': ids, 'probs': probs, 'probs_lr': probs_lr,
        'source_mask': token_mask.flatten(1), 'initial_token_ids': initial_ids,
        'prefix_predictor_calls': 2,
        **_candidate_action_schedule(h * w),
    }


def _predictor_mixture_probabilities(train_opt):
    """Stage-2 mature probabilities: self_context_prob is conditional on mixed input."""
    opt = train_opt.get('evaluator_input_opt', {}) or {}
    full_lq = float(opt.get('full_lq_prob', 0.2))
    self_context = float(opt.get('self_context_prob', 0.5))
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in (full_lq, self_context)):
        raise ValueError('evaluator_input_opt full_lq_prob/self_context_prob must be in [0, 1].')
    return full_lq, self_context


@torch.no_grad()
def _predictor_mixture_evaluator_state(net_g, lq_feats, gt_indices, train_opt, val_opt, **kwargs):
    """Batch-level Stage-2 input mixture, using the final Stage-2 proportions.

    Pure LR and GT/self completion all use the configured full-grid selection.
    Every branch is detached and shares an independent continuous-cosine
    next-action budget. No Stage-2 training warmup is replayed. val_opt is
    accepted for caller compatibility, but does not control training budgets.
    """
    full_lq_prob, self_context_prob = _predictor_mixture_probabilities(train_opt)
    if np.random.uniform() < full_lq_prob:
        state = _initial_evaluator_state(net_g, lq_feats, val_opt, train_opt=train_opt)
        state['source_mask'] = torch.ones_like(state['token_ids'], dtype=torch.bool)
        source = 'full_lq'
    elif torch.rand((), device=lq_feats.device).item() < self_context_prob:
        state = _self_context_evaluator_state(net_g, lq_feats, train_opt, val_opt=val_opt, **kwargs)
        source = 'self_context'
    else:
        state = _gt_context_evaluator_state(net_g, lq_feats, gt_indices, train_opt, val_opt, **kwargs)
        source = 'gt_context'
    state['input_source'] = source
    return state


def _evaluator_transition_loss(self_logits, pair_logits, old_ids, proposal_ids, gt_ids,
                               token_weight=None):
    """One optionally spatially weighted CE over self-pairs and changed pairs.

    Self-pairs teach current correctness for proposal selection. Identical-ID
    proposal positions add no new action example and are not counted twice.
    Detached spatial weights are shared by both pair types; labels are unchanged.
    GT-dependent weighting can affect calibration, although no class weights are
    used. The global denominator handles unequal weight sums across DDP ranks.
    """
    target_self = correctness_transition_target(old_ids, old_ids, gt_ids)
    target_pair = correctness_transition_target(old_ids, proposal_ids, gt_ids)
    if self_logits.shape != (*old_ids.shape, 4) or pair_logits.shape != self_logits.shape:
        raise ValueError('Both transition outputs must have shape (B,N,4).')
    if not torch.isfinite(self_logits).all() or not torch.isfinite(pair_logits).all():
        raise FloatingPointError('Non-finite transition logits.')
    self_ce = F.cross_entropy(self_logits.float().transpose(1, 2), target_self, reduction='none')
    pair_ce = F.cross_entropy(pair_logits.float().transpose(1, 2), target_pair, reduction='none')
    changed = proposal_ids.ne(old_ids)
    if token_weight is None:
        numerator = self_ce.sum() + (pair_ce * changed).sum()
        denominator = numerator.new_tensor(old_ids.numel()) + changed.sum().to(numerator)
    else:
        if token_weight.shape != old_ids.shape:
            raise ValueError('Transition token weights must have shape (B,N).')
        token_weight = token_weight.detach().to(self_ce)
        if not torch.isfinite(token_weight).all() or (token_weight < 1).any():
            raise ValueError('Transition token weights must be finite and at least 1.')
        changed_weight = token_weight * changed
        numerator = (self_ce * token_weight).sum() + (pair_ce * changed_weight).sum()
        denominator = token_weight.sum() + changed_weight.sum()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        torch.distributed.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM)
    # DDP averages rank gradients; undo that average for a global sample mean.
    loss = numerator * world_size / denominator.clamp_min(1)
    logs = {
        # Report raw, unweighted CE diagnostics.
        'l_current_correctness': self_ce.detach().mean(),
        'l_changed_transition': (pair_ce.detach() * changed).sum() / changed.sum().clamp_min(1),
        'transition_changed_ratio': changed.float().mean(),
    }
    if token_weight is not None:
        logs.update(semantic_weight_mean=token_weight.mean(),
                    semantic_weight_min=token_weight.min(),
                    semantic_weight_max=token_weight.max())
    return loss, logs


@torch.no_grad()
def _token_gain_diagnostics(old_ids, proposal_ids, final_ids, gt_ids, valid_mask=None):
    """Token-count diagnostics ONLY. Never used by the inference controller."""
    if not (old_ids.shape == proposal_ids.shape == final_ids.shape == gt_ids.shape):
        raise ValueError('Token-count diagnostic sequences must align.')
    valid = torch.ones_like(old_ids, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if valid.shape != old_ids.shape:
        raise ValueError('Diagnostic valid_mask must align with the token grid.')
    c0, cp, cf = old_ids.eq(gt_ids), proposal_ids.eq(gt_ids), final_ids.eq(gt_ids)
    requested_change = proposal_ids.ne(old_ids) & valid
    submitted_change = final_ids.ne(old_ids) & valid
    repairs = ~c0 & cf & valid
    damage = c0 & ~cf & valid
    actual_delta = ((cf.long() - c0.long()) * valid).sum(-1)
    repair_count, damage_count = repairs.sum(-1), damage.sum(-1)
    if not torch.equal(actual_delta, repair_count - damage_count):
        raise AssertionError('Correct-count conservation failed.')
    oracle_gain = (~c0 & cp & valid).sum(-1)
    stats = {
        'correct_count_before': (c0 & valid).sum(-1).float().mean(),
        'correct_count_proposal': (cp & valid).sum(-1).float().mean(),
        'correct_count_after': (cf & valid).sum(-1).float().mean(),
        'net_correct_gain': actual_delta.float().mean(),
        'repaired_count': repair_count.float().mean(),
        'damaged_count': damage_count.float().mean(),
        'proposal_oracle_gain': oracle_gain.float().mean(),
        'proposal_net_correct_gain': (((cp.long() - c0.long()) * valid).sum(-1)).float().mean(),
        'accepted_change_count': submitted_change.sum(-1).float().mean(),
        'accepted_wrong_to_wrong_count': (submitted_change & ~c0 & ~cf).sum(-1).float().mean(),
        'acceptance_rate': submitted_change.sum().float() / requested_change.sum().clamp_min(1),
        'accepted_repair_precision': repairs.sum().float() / submitted_change.sum().clamp_min(1),
        'repair_recall': repairs.sum().float() / oracle_gain.sum().clamp_min(1),
        'correct_rate_before': (c0 & valid).sum().float() / valid.sum().clamp_min(1),
        'correct_rate_after': (cf & valid).sum().float() / valid.sum().clamp_min(1),
    }
    return stats


def _rgb_to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor in [0, 1] to Matlab-style Y channel."""
    if img.dim() < 4 or img.shape[1] != 3:
        return img
    coeffs = img.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
    return (img * coeffs).sum(dim=1, keepdim=True).add_(16.0).div_(255.0)


def _psnr(pred: torch.Tensor,
          target: torch.Tensor,
          crop_border: int = 0,
          test_y_channel: bool = False) -> torch.Tensor:
    if crop_border > 0:
        pred = pred[..., crop_border:-crop_border, crop_border:-crop_border]
        target = target[..., crop_border:-crop_border, crop_border:-crop_border]
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    if test_y_channel:
        pred = _rgb_to_y_channel(pred)
        target = _rgb_to_y_channel(target)
    mse = F.mse_loss(pred, target)
    if mse <= 0:
        return torch.tensor(float('inf'), device=pred.device)
    return 10 * torch.log10(1.0 / mse)


def _build_optional_metric(name: str, device, **kwargs):
    name = name.lower()
    if pyiqa is not None:
        return pyiqa.create_metric(name, device=device, **kwargs)
    if name == 'psnr':
        crop_border = int(kwargs.get('crop_border', 0))
        test_y_channel = bool(kwargs.get('test_y_channel', False))
        return lambda pred, gt: _psnr(
            pred, gt, crop_border=crop_border, test_y_channel=test_y_channel)
    return None


class _LossBuilderMixin:
    def _image_supervision_losses(self, image, gt, current_iter, target_cache=None):
        """Shared Stage-2/3 decoded-image objectives, including their schedules.

        Token CE, encoder feature loss and latent alignment are deliberately
        outside this helper: they do not supervise the reconstructed image.
        Loss networks must preserve autograd with respect to ``image``.
        """
        train_opt = self.opt['train']
        losses = OrderedDict()
        with self.amp_autocast():
            if self.cri_pix is not None:
                losses['l_pix'] = self.cri_pix(image, gt)
            if self.cri_struct is not None:
                losses['l_struct'] = (1 - self.cri_struct(image, gt).mean()) * train_opt['struct_opt']['loss_weight']
            if self.cri_perceptual is not None:
                perceptual, style = self.cri_perceptual(image, gt)
                if perceptual is not None:
                    losses['l_percep'] = perceptual.mean()
                if style is not None:
                    losses['l_style'] = style
        dino_ramp = _ramp_multiplier(train_opt.get('dino_guidance_opt', {}) or {}, current_iter)
        if self.cri_dino is not None and dino_ramp > 0:
            for name, value in self.cri_dino.image_pair_loss(image, gt, target_cache=target_cache).items():
                losses[name] = dino_ramp * value
        if self.use_dis and current_iter > self.net_d_start_iter:
            with self.amp_autocast():
                prediction = self.net_d(image)
            losses['l_g_gan'] = self.cri_gan(prediction, True, is_disc=False)
        return losses

    def _clear_validation_tensors(self):
        """Drop per-image tensors before returning unused CUDA cache blocks."""
        for name in ('lq', 'gt', 'output', 'predictor_output', 'hq_rec',
                     'refinement_diagnostics', '_training_token_ids'):
            if hasattr(self, name):
                delattr(self, name)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _safe_build_loss(self, opt: Optional[dict], loss_name: str):
        if not opt:
            return None
        try:
            loss = build_loss(opt).to(self.device)
            self.model_to_device(loss)
            return loss
        except Exception as exc:
            get_root_logger().warning(f'Disable {loss_name}: failed to build loss from {opt}: {exc}')
            return None

    def _build_metric_funcs(self):
        self.metric_funcs = {}
        val_opt = self.opt.get('val', {})
        if val_opt.get('metrics') is None:
            return
        for _, opt in val_opt['metrics'].items():
            mopt = opt.copy()
            name = mopt.pop('type', None)
            mopt.pop('better', None)
            metric = _build_optional_metric(name, self.device, **mopt)
            if metric is not None:
                self.metric_funcs[name] = metric
            else:
                get_root_logger().warning(f'Metric {name} requires pyiqa and is disabled in this environment.')

    def _is_main_process(self) -> bool:
        if hasattr(self, 'accelerator'):
            return bool(self.accelerator.is_main_process)
        return int(self.opt.get('rank', 0)) == 0

    def _update_best_metric_and_save(self, dataset_name, current_iter, net, net_label):
        """Update validation best records and save the corresponding best net."""
        key_metric = self.opt.get('val', {}).get('key_metric')
        best_updated = False
        if key_metric is not None:
            if key_metric in self.metric_results:
                best_updated = self._update_best_metric_result(
                    dataset_name, key_metric, self.metric_results[key_metric], current_iter)
                if best_updated:
                    for metric, value in self.metric_results.items():
                        self._update_metric_result(dataset_name, metric, value, current_iter)
            else:
                get_root_logger().warning(
                    f'key_metric={key_metric} is not in validation metrics: '
                    f'{list(self.metric_results.keys())}. Skip best checkpoint update.')
        else:
            for metric, value in self.metric_results.items():
                best_updated |= self._update_best_metric_result(dataset_name, metric, value, current_iter)

        if best_updated and self._is_main_process():
            self.save_network(net, f'{net_label}_best', '')
        return best_updated

    def _log_validation_metrics_with_best(self, current_iter, dataset_name, tb_logger):
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


@MODEL_REGISTRY.register()
class SIPEPredictorModel(_LossBuilderMixin, BaseModel):
    """Stage-2 SIPE-SR predictor training model."""

    def __init__(self, opt):
        super().__init__(opt)

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if opt.get('network_hq') is not None:
            self.net_hq = build_network(opt['network_hq'])
        else:
            hq_opt = opt['network_g']['vqgan_opt'].copy()
            hq_opt['type'] = 'VQModel'
            self.net_hq = build_network(hq_opt)
        self.net_hq = self.model_to_device(self.net_hq)
        self.print_network(self.net_hq)

        self._build_metric_funcs()

        # Load the predictor first, then re-apply the designated Stage-1 prior.
        # This order prevents a predictor checkpoint from silently replacing
        # only its internal VQ branch after the standalone GT tokenizer was
        # loaded, which would make GT token IDs and decoded tokens incompatible.
        load_g = opt['path'].get('pretrain_network_g', None)
        if load_g is not None:
            get_root_logger().info(f'Loading net_g from {load_g}')
            self.load_network(self.net_g, load_g, opt['path'].get('strict_load', False))

        load_hq = opt['path'].get('pretrain_network_hq', None)
        if load_hq is not None:
            self.load_network(self.net_hq, load_hq, opt['path'].get('strict_load_hq', False))
            self.load_network(
                _bare(self, self.net_g).vqgan,
                load_hq,
                opt['path'].get('strict_load_hq', False),
            )
        _synchronize_hq_prior(self, self.net_hq, self.net_g)

        for p in _bare(self, self.net_hq).parameters():
            p.requires_grad = False
        for p in _bare(self, self.net_g).vqgan.parameters():
            p.requires_grad = False
        self.net_hq.eval()
        _bare(self, self.net_g).vqgan.eval()

        if self.is_train:
            self.init_training_settings()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def init_training_settings(self):
        self.net_g.train()
        self.net_hq.eval()
        _bare(self, self.net_g).vqgan.eval()

        train_opt = self.opt['train']
        self.setup_amp(train_opt, get_root_logger())
        self.net_d = build_network(self.opt['network_d']) if self.opt.get('network_d') else None
        self.use_dis = False
        if self.net_d is not None:
            self.net_d = self.model_to_device(self.net_d)
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', False))
            self.net_d.train()
            self.use_dis = bool(train_opt.get('gan_opt') and train_opt['gan_opt'].get('loss_weight', 0) != 0)

        self.cri_pix = self._safe_build_loss(train_opt.get('pixel_opt'), 'pixel loss')
        self.cri_perceptual = self._safe_build_loss(train_opt.get('perceptual_opt'), 'perceptual loss')
        self.cri_gan = self._safe_build_loss(train_opt.get('gan_opt'), 'GAN loss')
        self.cri_struct = None
        if train_opt.get('struct_opt') and pyiqa is not None:
            self.cri_struct = pyiqa.create_metric(train_opt['struct_opt']['type'], as_loss=True).to(self.device)
        elif train_opt.get('struct_opt'):
            get_root_logger().warning('struct_opt is configured but pyiqa is unavailable; structural loss is disabled.')

        self.cri_dino = None
        if train_opt.get('dino_opt'):
            dino_opt = train_opt['dino_opt'].copy()
            try:
                self.cri_dino = DINOSemanticLoss(**dino_opt).to(self.device)
                self.cri_dino.eval()
                get_root_logger().info(f'DINO semantic guidance is enabled for SIPEPredictor: {dino_opt}')
            except Exception as exc:
                if train_opt.get('dino_allow_fail', False):
                    get_root_logger().warning(f'DINO guidance disabled because the teacher could not be built: {exc}')
                    self.cri_dino = None
                else:
                    raise

        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        frozen_params = 0
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                frozen_params += v.numel()
        if frozen_params:
            get_root_logger().info(f'{frozen_params:,} frozen net_g parameters will not be optimized (SwinVQGAN prior).')
        if self.cri_dino is not None:
            dino_params = list(self.cri_dino.trainable_parameters())
            if dino_params:
                optim_params.extend(dino_params)
                dino_param_count = sum(p.numel() for p in dino_params)
                get_root_logger().info(f'{dino_param_count:,} DINO latent-projector parameters will be optimized.')
        optim_opt = dict(train_opt['optim_g'])
        optim_type = optim_opt.pop('type')
        self.optimizer_g = getattr(torch.optim, optim_type)(optim_params, **optim_opt)
        self.optimizers.append(self.optimizer_g)

        if self.net_d is not None:
            optim_d_opt = dict(train_opt['optim_d'])
            optim_type = optim_d_opt.pop('type')
            self.optimizer_d = getattr(torch.optim, optim_type)(self.net_d.parameters(), **optim_d_opt)
            self.optimizers.append(self.optimizer_d)

    def set_acc(self, acc):
        """Prepare trainable networks and optimizers for the accelerate trainer."""
        self.accelerator = acc
        if self.net_d is not None:
            prepared = acc.prepare(self.net_g, self.net_d, self.net_hq, self.optimizer_g, self.optimizer_d)
            self.net_g, self.net_d, self.net_hq, self.optimizer_g, self.optimizer_d = prepared
        else:
            self.net_g, self.net_hq, self.optimizer_g = acc.prepare(self.net_g, self.net_hq, self.optimizer_g)

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        net_hq = _bare(self, self.net_hq)

        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        with torch.no_grad(), self.amp_autocast():
            self.hq_rec, target_feats, gt_indices = _vq_indices_and_feats(net_hq, self.gt)
            b, _, h, w = gt_indices.shape

        # Optional DINO cache shared by semantic mask sampling, weighted token
        # supervision, latent alignment and image-space semantic distillation.
        dino_guidance_opt = dict(train_opt.get('dino_guidance_opt', {}) or {})
        dino_ramp = _ramp_multiplier(dino_guidance_opt, current_iter)
        dino_score = None
        dino_target_cache = None
        if self.cri_dino is not None and dino_ramp > 0:
            with torch.no_grad():
                dino_target_cache = self.cri_dino.prepare_features(self.gt)
                need_reference = float(dino_guidance_opt.get('difficulty_weight', 1.0)) > 0
                reference = None
                reference_cache = None
                if need_reference:
                    reference = F.interpolate(self.lq.float(), size=self.gt.shape[-2:], mode='bicubic', align_corners=False).clamp(0, 1)
                    reference_cache = self.cri_dino.prepare_features(reference)
                dino_score = self.cri_dino.token_score_map(
                    self.gt,
                    (h, w),
                    reference=reference,
                    target_cache=dino_target_cache,
                    reference_cache=reference_cache,
                    saliency_weight=dino_guidance_opt.get('saliency_weight', 0.0),
                    difficulty_weight=dino_guidance_opt.get('difficulty_weight', 1.0),
                )

        soft_decode_enabled, soft_decode_tau, soft_decode_st = _soft_decode_settings(train_opt, current_iter)
        # Choose the context source independently of mask selection. Both GT
        # and self-context use the same semantic-gap + random-noise policy.
        self_context_opt = train_opt.get('predictor_self_context_opt', {}) or {}
        self_context_enabled = bool(self_context_opt.get('enabled', False))
        self_context_target_prob = max(0.0, min(1.0, float(self_context_opt.get('prob', 0.0))))
        if 'start_iter' in self_context_opt:
            self_context_start = int(self_context_opt['start_iter'])
        else:
            start_ratio = float(self_context_opt.get('start_ratio', 0.5))
            self_context_start = int(float(train_opt.get('total_iter', 0)) * start_ratio)
        self_context_prob = 0.0
        use_self_context = False
        if self_context_enabled and current_iter >= self_context_start and self_context_target_prob > 0:
            context_warmup = int(self_context_opt.get('warmup_iter', 0))
            if context_warmup > 0:
                context_progress = min(
                    1.0,
                    max(0.0, (current_iter - self_context_start + 1) / float(context_warmup)),
                )
            else:
                context_progress = 1.0
            self_context_prob = self_context_target_prob * context_progress
            use_self_context = torch.rand((), device=self.device).item() < self_context_prob

        # Only the explicit full-LQ branch may remove all token context.
        # Mixed inputs round the mask count down, allowing zero LR positions.
        full_lq_prob = float(train_opt.get('predictor_full_lq_prob', 0.0))
        full_lq_prob = max(0.0, min(1.0, full_lq_prob))
        if np.random.uniform() < full_lq_prob:
            token_mask = torch.ones(b, 1, h, w, dtype=torch.bool, device=self.device)
        else:
            r = np.random.uniform()
            mask_method = train_opt.get('token_schedule', 'cosine')
            # Preserve at least one GT/self-context token, even at r=0 or if
            # floating-point rounding yields a mask ratio of 1. For N=1 this
            # branch uses token context only. CE still supervises all outputs.
            num_lq_positions = min(h * w - 1, max(
                0,
                math.floor(schedule(r, h * w, method=mask_method) * h * w),
            ))
            if dino_score is not None and dino_guidance_opt.get('mask_guided', True):
                token_mask = _score_guided_token_mask(
                    dino_score,
                    num_lq_positions,
                    random_ratio=dino_guidance_opt.get('mask_random_ratio', 0.15),
                )
            else:
                token_mask = _random_token_mask(b, h, w, num_lq_positions, self.device)

        with self.amp_autocast():
            logits, lq_feats, self.output = self.net_g(
                self.lq,
                hq_feats=target_feats,
                token_mask=token_mask,
                code_only=False,
                soft_decode=soft_decode_enabled,
                soft_decode_tau=soft_decode_tau,
                soft_decode_straight_through=soft_decode_st,
                use_self_context=use_self_context,
            )

        image_output = self.output

        l_g_total = 0
        loss_dict = OrderedDict()
        if soft_decode_enabled:
            loss_dict['soft_decode_tau'] = logits.new_tensor(soft_decode_tau)
        if self_context_enabled:
            loss_dict['predictor_self_context'] = logits.new_tensor(float(use_self_context))
            loss_dict['predictor_self_context_prob'] = logits.new_tensor(self_context_prob)

        semantic_weight = None
        if dino_score is not None:
            semantic_weight = 1.0 + dino_ramp * float(dino_guidance_opt.get('token_weight_boost', 1.0)) * dino_score

        feat_weight = train_opt.get('feature_opt', {}).get('loss_weight', 1.0)
        if feat_weight:
            feat_map = (lq_feats.float() - target_feats.detach().float()).pow(2).mean(dim=1, keepdim=True)
            feat_boost = float(dino_guidance_opt.get('feature_weight_boost', 0.0)) if dino_score is not None else 0.0
            feat_loss_weight = 1.0 + dino_ramp * feat_boost * dino_score if feat_boost > 0 else None
            l_feat = _weighted_mean(feat_map, feat_loss_weight) * feat_weight
            l_g_total = l_g_total + l_feat
            loss_dict['l_feat_encoder'] = l_feat

        token_opt = train_opt.get('cross_entropy_opt', {})
        token_weight = token_opt.get('token_weight', 1.0)
        if token_weight:
            ce_map = F.cross_entropy(
                logits.permute(0, 2, 1).float(),
                gt_indices.flatten(1),
                reduction='none',
                label_smoothing=0.0,
            )
            # Supervise every output ID, including GT/self-context positions.
            # The input mask only selects LR versus token features.
            # Without optional semantic reweighting this is mean CE over B*H*W.
            token_loss_weight = torch.ones_like(ce_map)
            if semantic_weight is not None and dino_guidance_opt.get('token_reweight', True):
                token_loss_weight = token_loss_weight * semantic_weight.flatten(1)
            l_token = (
                (ce_map * token_loss_weight).sum()
                / token_loss_weight.sum().clamp_min(1.0)
            ) * token_weight
            l_g_total = l_g_total + l_token
            loss_dict['l_token'] = l_token

        image_losses = self._image_supervision_losses(
            image_output, self.gt, current_iter, target_cache=dino_target_cache)
        for name, value in image_losses.items():
            l_g_total = l_g_total + value
            loss_dict[name] = value

        if self.cri_dino is not None and dino_ramp > 0 and dino_guidance_opt.get('lq_latent_weight', 0) > 0:
            dino_losses = self.cri_dino.latent_alignment_loss(
                lq_feats, self.gt, target_cache=dino_target_cache,
                weight=dino_guidance_opt['lq_latent_weight'], name='l_dino_lq_latent')
            for k, v in dino_losses.items():
                v = v * dino_ramp
                l_g_total = l_g_total + v
                loss_dict[k] = v

        self.amp_backward(l_g_total.mean())
        self.amp_optimizer_step(self.optimizer_g)
        self.optimizer_g.zero_grad()

        if self.net_d is not None:
            self.fixed_disc = self.opt['train'].get('fixed_disc', False)
            if not self.fixed_disc and self.use_dis and current_iter > self.net_d_start_iter:
                for p in self.net_d.parameters():
                    p.requires_grad = True
                with self.amp_autocast():
                    real_d_pred = self.net_d(self.gt.float())
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                loss_dict['l_d_real'] = l_d_real
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                self.amp_backward(l_d_real)
                with self.amp_autocast():
                    fake_d_pred = self.net_d(image_output.float().detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                loss_dict['l_d_fake'] = l_d_fake
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                self.amp_backward(l_d_fake)
                self.amp_optimizer_step(self.optimizer_d)
                self.optimizer_d.zero_grad()

        self.amp_scaler_update()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def test(self):
        was_training = self.net_g.training
        self.net_g.eval()
        net_g = _bare(self, self.net_g)
        min_size = self.opt.get('val', {}).get('tile_min_size', 8000 * 8000)
        if self.lq.shape[-1] * self.lq.shape[-2] < min_size:
            self.output = net_g.test(self.lq)
        else:
            self.output = net_g.test_tile(self.lq)
        if was_training:
            self.net_g.train()
            _bare(self, self.net_g).vqgan.eval()
        return self.output

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        get_root_logger().info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir):
        self._clear_validation_tensors()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt.get('val', {}).get('metrics') is not None and len(getattr(self, 'metric_funcs', {})) > 0
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_funcs.keys()}
            self._initialize_best_metric_results(dataset_name)

        pbar = tqdm(total=len(dataloader), unit='image')
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            sr_img = tensor2img(self.output)
            # Metrics need the converted image and GT, not the GPU SR output.
            del self.lq, self.output
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], 'image_results', f'{current_iter}', f'{img_name}.png')
                else:
                    suffix = self.opt.get('val', {}).get('suffix', self.opt['name'])
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{suffix}.png')
                if save_as_dir:
                    imwrite(sr_img, osp.join(save_as_dir, f'{img_name}.png'))
                imwrite(sr_img, save_img_path)

            if with_metrics and hasattr(self, 'gt'):
                pred = img2tensor(sr_img).unsqueeze(0).to(self.device) / 255.0
                for name, func in self.metric_funcs.items():
                    self.metric_results[name] += func(pred, self.gt).item()
                del pred

            self._clear_validation_tensors()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')

        pbar.close()
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        self._update_best_metric_and_save(dataset_name, current_iter, self.net_g, 'net_g')
        self._log_validation_metrics_with_best(current_iter, dataset_name, tb_logger)

    @torch.no_grad()
    def get_current_visuals(self):
        vis_samples = 4
        out_dict = OrderedDict()
        if hasattr(self, 'output'):
            out_dict['lq_up'] = F.interpolate(self.lq, self.output.shape[-2:], mode='bicubic', align_corners=False).detach().cpu()[:vis_samples]
            out_dict['output'] = self.output.detach().cpu()[:vis_samples]
        else:
            out_dict['lq'] = self.lq.detach().cpu()[:vis_samples]
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:vis_samples]
        if hasattr(self, 'hq_rec'):
            out_dict['hq_rec'] = self.hq_rec.detach().cpu()[:vis_samples]
        return OrderedDict((k, v.float().clamp(0, 1)) for k, v in out_dict.items())

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class SIPEEvaluatorModel(_LossBuilderMixin, BaseModel):
    """Learn per-token correctness transitions, keeping all of Stage 2 frozen."""

    def __init__(self, opt):
        super().__init__(opt)
        train_opt = opt.get('train', {}) or {}
        input_opt = train_opt.get('evaluator_input_opt', {}) or {}
        self.evaluator_input_mode = str(input_opt.get('mode', 'rollout'))
        if self.evaluator_input_mode not in (
                'rollout', 'initial_prediction', 'gt_context', 'self_context', 'predictor_mixture'):
            raise ValueError('Unknown evaluator input mode.')
        self.use_semantic_input_mask = bool(input_opt.get('mask_guided', False)) and self.is_train
        if self.use_semantic_input_mask and self.evaluator_input_mode not in (
                'gt_context', 'self_context', 'predictor_mixture'):
            raise ValueError('Semantic input masks only apply to context state builders.')
        self.semantic_weight_rho = 0.0
        self.use_semantic_weight = False
        if self.is_train:
            weight_opt = train_opt.get('evaluator_semantic_weight_opt', {}) or {}
            unknown = set(weight_opt) - {'enabled', 'rho'}
            if unknown:
                raise ValueError(f'Unknown evaluator_semantic_weight_opt keys: {sorted(unknown)}; '
                                 'use enabled and rho.')
            self.semantic_weight_rho = float(weight_opt.get('rho', 0.2))
            if not math.isfinite(self.semantic_weight_rho) or self.semantic_weight_rho < 0:
                raise ValueError('evaluator_semantic_weight_opt.rho must be finite and nonnegative.')
            self.use_semantic_weight = (bool(weight_opt.get('enabled', False))
                                        and self.semantic_weight_rho > 0)
            objective = train_opt.get('evaluator_supervision_opt', {}) or {}
            if objective != {'target_mode': 'correctness_transition'}:
                raise ValueError('Use evaluator_supervision_opt: {target_mode: correctness_transition}; '
                                 'legacy risk targets are not supported; spatial CE weights belong '
                                 'in evaluator_semantic_weight_opt.')
            for key in ('cross_entropy_opt', 'evaluator_semantic_opt', 'pixel_opt',
                        'perceptual_opt', 'struct_opt', 'gan_opt'):
                if train_opt.get(key):
                    raise ValueError(f'Remove train.{key}: Stage 3 uses correctness-transition CE only.')
            if (train_opt.get('evaluator_reconstruction_opt') or {}).get('enabled', False):
                raise ValueError('Disable old ST reconstruction: train on actual discrete transitions.')
            val_opt = opt.get('val', {}) or {}
            if resolve_num_refine_steps(val_opt) < 1:
                raise ValueError('Stage-3 training needs at least one candidate proposal step.')
            sampling = (train_opt.get('evaluator_sampling_opt', {}) or {}).get('mode', 'multinomial')
            if sampling != val_opt.get('decode_sampling', val_opt.get('token_selection', 'multinomial')):
                raise ValueError('Training/validation token sampling must match.')
            if float(train_opt.get('sample_temperature', 1.0)) != float(val_opt.get('sample_temperature', 1.0)):
                raise ValueError('Training/validation sampling temperatures must match.')
            if resolve_token_update_mode(_evaluator_refinement_options(train_opt)) != resolve_token_update_mode(val_opt):
                raise ValueError('Training/validation proposal update modes must match.')
        self.net_d = self.cri_pix = self.cri_perceptual = self.cri_struct = self.cri_gan = None
        self.cri_dino = None
        self.net_g = build_network(opt['network_g'])
        # Each rank owns a fixed predictor; do not prepare it for training.
        self.net_g = self.net_g.to(self.device)
        self.print_network(self.net_g)

        self.net_evaluator = build_network(opt['network_evaluator'])
        self.net_evaluator = self.net_evaluator.to(self.device)
        self.print_network(self.net_evaluator)

        if opt.get('network_hq') is not None:
            self.net_hq = build_network(opt['network_hq'])
        else:
            hq_opt = opt['network_g']['vqgan_opt'].copy()
            hq_opt['type'] = 'VQModel'
            self.net_hq = build_network(hq_opt)
        self.net_hq = self.net_hq.to(self.device)

        self._build_metric_funcs()

        load_g = opt['path'].get('pretrain_network_g', None)
        if load_g is not None:
            self.load_network(self.net_g, load_g, opt['path'].get('strict_load', False))

        # The complete frozen Stage-2 checkpoint is the source of truth.
        # An explicit HQ prior must match it, never silently overwrite it.
        load_hq = opt['path'].get('pretrain_network_hq', None)
        if load_hq is not None:
            self.load_network(self.net_hq, load_hq, opt['path'].get('strict_load_hq', False))
            if load_g is not None:
                predictor_prior = _bare(self, self.net_g).vqgan.state_dict()
                requested_prior = _bare(self, self.net_hq).state_dict()
                if (predictor_prior.keys() != requested_prior.keys() or any(
                        not torch.equal(value, requested_prior[key])
                        for key, value in predictor_prior.items())):
                    raise ValueError(
                        'Stage-3 pretrain_network_hq differs from the frozen Stage-2 VQ prior. '
                        'Use pretrain_network_hq: null or the identical Stage-1 checkpoint.')
            else:
                _bare(self, self.net_g).vqgan.load_state_dict(
                    _bare(self, self.net_hq).state_dict(), strict=True)
        _synchronize_hq_prior(self, self.net_hq, self.net_g)

        load_evaluator = opt['path'].get('pretrain_network_evaluator', None)
        if load_evaluator is not None:
            self.load_network(
                self.net_evaluator,
                load_evaluator,
                opt['path'].get('strict_load_evaluator', False),
            )

        # Independent critic embeddings retain their random/loaded weights.
        # The frozen_vq mode copies and checks the Stage-2 codebook.
        evaluator = _bare(self, self.net_evaluator)
        evaluator.prepare_token_embedding(
            _bare(self, self.net_g).vqgan.quantize.embedding.weight,
            checkpoint_loaded=load_evaluator is not None,
        )

        # All predictor components stay frozen: LR encoder, Transformer, VQ
        # prior, decoder, SFT/conditioning paths, and any refinement modules.
        for net in (self.net_g, self.net_hq):
            net.eval()
            for parameter in net.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

        # Only evaluator parameters, including independent token embeddings,
        # need a training wrapper.
        self.net_evaluator = self.model_to_device(self.net_evaluator)


        get_root_logger().info(
            'Evaluator: four correctness transitions; one CE objective; per-token '
            'acceptance p(repair)>p(damage); no semantic-risk/image loss or learned predictor updates.')
        if self.is_train:
            self.init_training_settings()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        for name in ('_training_token_ids', 'output'):
            if hasattr(self, name):
                delattr(self, name)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        elif hasattr(self, 'gt'):
            del self.gt

    def _set_training_modes(self):
        self.net_g.eval()
        self.net_hq.eval()
        if self.cri_dino is not None:
            self.cri_dino.eval()
        self.net_evaluator.train()

    def init_training_settings(self):
        self._set_training_modes()
        train_opt = self.opt['train']
        self.setup_amp(train_opt, get_root_logger())
        evaluator = _bare(self, self.net_evaluator)
        condition = (train_opt.get('evaluator_condition_opt') or {}).get('use_lq_feats', evaluator.use_cond_feats)
        if bool(condition) != evaluator.use_cond_feats:
            raise ValueError('Training LR condition must match network_evaluator.use_cond_feats.')
        if self.use_semantic_input_mask or self.use_semantic_weight:
            if not train_opt.get('dino_opt'):
                raise ValueError('Semantic CE weights or guided input masks require train.dino_opt.')
            # Training-only frozen teacher; never replaces correctness labels.
            self.cri_dino = DINOSemanticLoss(**train_opt['dino_opt']).to(self.device).eval().requires_grad_(False)
        if self.use_semantic_weight:
            get_root_logger().info(
                f'Evaluator spatial CE weighting: w=1+rho*s, rho={self.semantic_weight_rho}; '
                's is the frozen DINO GT/LR-up feature gap; inference is unchanged.')
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        frozen = list(self.net_g.parameters()) + list(self.net_hq.parameters())
        if self.cri_dino is not None:
            frozen += list(self.cri_dino.parameters())
        if any(p.requires_grad for p in frozen):
            raise RuntimeError('Only evaluator weights may be trainable in Stage 3.')
        config = dict(self.opt['train']['optim_evaluator'])
        name = config.pop('type')
        params = [p for p in self.net_evaluator.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError('No trainable evaluator parameters.')
        self.optimizer_evaluator = getattr(torch.optim, name)(
            [{'params': params, 'name': 'evaluator_correctness_transition'}], **config)
        self.optimizers.append(self.optimizer_evaluator)
        get_root_logger().info(f'Trainable evaluator parameters: {sum(p.numel() for p in params):,}; '
                               'trainable Stage-2 parameters: 0.')

    def set_acc(self, acc):
        self.accelerator = acc
        self.net_evaluator, self.optimizer_evaluator = acc.prepare(
            self.net_evaluator, self.optimizer_evaluator)
        self._set_training_modes()

    def resume_training(self, resume_state):
        optimizers = resume_state.get('optimizers', [])
        if (len(optimizers) != 1 or len(optimizers[0].get('param_groups', [])) != 1
                or optimizers[0]['param_groups'][0].get('name') != 'evaluator_correctness_transition'):
            raise ValueError('Cannot resume a legacy evaluator optimizer; start a new Stage-3 run.')
        super().resume_training(resume_state)

    def optimize_parameters(self, current_iter):
        if not hasattr(self, 'gt'):
            raise ValueError('Stage-3 supervised training requires HR targets.')
        self._set_training_modes()
        train_opt, val_opt = self.opt['train'], self.opt.get('val', {}) or {}
        net_g, net_hq, evaluator = _bare(self, self.net_g), _bare(self, self.net_hq), _bare(self, self.net_evaluator)
        horizon = resolve_num_refine_steps(val_opt)
        evaluator.eval()
        with torch.no_grad(), self.amp_autocast():
            quant, gt_indices = net_hq.encode_to_indices(self.gt)
            b, _, h, w = quant.shape
            gt_indices = gt_indices.reshape(b, 1, h, w).long()
            del quant
            lq_feats = net_g.encode_lq_feats(self.lq).detach()
            if lq_feats.shape[-2:] != (h, w):
                raise ValueError('Frozen LR and GT token grids must agree.')
            kwargs = dict(lq=self.lq, gt=self.gt, semantic_teacher=self.cri_dino,
                          target_cache=None, current_iter=current_iter)
            if self.cri_dino is not None:
                kwargs['target_cache'] = self.cri_dino.prepare_features(self.gt)
            token_weight = None
            if self.use_semantic_weight:
                reference = F.interpolate(
                    self.lq.float(), size=self.gt.shape[-2:], mode='bicubic',
                    align_corners=False).clamp(0, 1)
                score = self.cri_dino.token_score_map(
                    self.gt, (h, w), reference=reference,
                    target_cache=kwargs['target_cache'], difficulty_weight=1.0)
                token_weight = (1.0 + self.semantic_weight_rho * score.flatten(1)).detach()
            if self.evaluator_input_mode == 'rollout':
                state = _rollout_evaluator_state(net_g, evaluator, lq_feats, val_opt)
            elif self.evaluator_input_mode == 'initial_prediction':
                state = _initial_evaluator_state(net_g, lq_feats, train_opt=train_opt)
            elif self.evaluator_input_mode == 'gt_context':
                state = _gt_context_evaluator_state(net_g, lq_feats, gt_indices, train_opt, val_opt, **kwargs)
            elif self.evaluator_input_mode == 'predictor_mixture':
                state = _predictor_mixture_evaluator_state(net_g, lq_feats, gt_indices, train_opt, val_opt, **kwargs)
            else:
                state = _self_context_evaluator_state(net_g, lq_feats, train_opt, val_opt=val_opt, **kwargs)
            old_ids, gt_ids = state['token_ids'].detach(), gt_indices.flatten(1)
            # Non-rollout state builders are retained, but all NEXT actions use
            # the same discrete cosine schedule and proposal routine as testing.
            step = state.get('state_step')
            if step is None:
                step = int(np.random.randint(horizon))
            count = cosine_remask_count(h * w, step, horizon, val_opt.get('remask_budget_scale', 1.0))
            cond = lq_feats if evaluator.use_cond_feats else None
            self_logits_eval = evaluator(old_ids, h, w, cond_feats=cond, proposal_ids=old_ids)
            proposal_ids, decision, calls = propose_evaluator_tokens(
                lq_feats, old_ids, transition_error_probability(self_logits_eval), count,
                net_g.predict_logits, net_g.tokens_to_feats,
                token_selection=val_opt.get('decode_sampling', 'multinomial'),
                sample_temperature=float(val_opt.get('sample_temperature', 1.0)),
                token_update_mode=resolve_token_update_mode(val_opt),
                state_step=step, num_refine_steps=horizon)
        # A single wrapped forward/backward: both pair types train the SAME head.
        self.net_evaluator.train()
        with self.amp_autocast():
            logits = self.net_evaluator(
                torch.cat((old_ids, old_ids), 0), h, w,
                cond_feats=torch.cat((lq_feats, lq_feats), 0) if evaluator.use_cond_feats else None,
                proposal_ids=torch.cat((old_ids, proposal_ids), 0))
        self_logits, pair_logits = logits.split(b, 0)
        loss, loss_logs = _evaluator_transition_loss(
            self_logits, pair_logits, old_ids, proposal_ids, gt_ids, token_weight=token_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite correctness-transition loss.')
        self.amp_backward(loss)
        self.amp_optimizer_step(self.optimizer_evaluator)
        self.optimizer_evaluator.zero_grad(set_to_none=True)
        self.amp_scaler_update()
        with torch.no_grad():
            updated, accepted, gain = accept_token_gains(old_ids, proposal_ids, pair_logits)
            self._training_token_ids = updated.reshape(b, 1, h, w).detach()
            probs = transition_probabilities(pair_logits)
            target = correctness_transition_target(old_ids, proposal_ids, gt_ids)
            loss_dict = OrderedDict(l_evaluator=loss.detach(), **loss_logs)
            loss_dict.update(_token_gain_diagnostics(old_ids, proposal_ids, updated, gt_ids))
            for name, label in (('repair', 1), ('damage', 2)):
                matches = target.eq(label)
                loss_dict[f'{name}_prob_on_true_{name}'] = (
                    probs[..., label] * matches).sum() / matches.sum().clamp_min(1)
            loss_dict.update(
                evaluator_error_mean=transition_error_probability(self_logits).mean(),
                predicted_accepted_correct_gain=(gain * accepted).sum(-1).mean(),
                remask_ratio=decision['mask'].float().mean(),
                sampled_state_step=loss.new_tensor(float(step)),
                proposal_predictor_calls=loss.new_tensor(float(calls)),
                prefix_predictor_calls=loss.new_tensor(float(state.get('prefix_predictor_calls', 1))),
                prefix_evaluator_calls=loss.new_tensor(float(state.get('prefix_evaluator_calls', 0))))
        self.log_dict = self.reduce_loss_dict(loss_dict)

    @torch.no_grad()
    def _validation_token_counts(self, history, proposals, actual_steps, h_old, w_old, h, w):
        # Called only AFTER decode_evaluator returns. Labels cannot alter actions.
        if not hasattr(self, 'gt'):
            return {}
        net_g, net_hq = _bare(self, self.net_g), _bare(self, self.net_hq)
        scale, downsample = net_g.scale_factor, net_g.vq_downsample
        if self.gt.shape[-2:] != (h_old * scale, w_old * scale):
            get_root_logger().warning('Skip token-count diagnostics: LR/HR dimensions do not align.')
            return {}
        multiple = max(1, downsample * net_g.predictor_window_size)
        gt_pad, _, _ = _flip_pad_to_multiple(self.gt, multiple)
        quant, gt_ids = net_hq.encode_to_indices(gt_pad)
        if quant.shape[-2:] != (h, w):
            raise ValueError('Validation GT tokenizer and predicted token grids differ.')
        del quant
        b = history.shape[0]
        gt_ids = gt_ids.reshape(b, h * w).long()
        valid = torch.zeros((b, h, w), dtype=torch.bool, device=gt_ids.device)
        # Exclude synthetic padding and boundary cells that are not fully inside HR.
        valid[:, :self.gt.shape[-2] // downsample, :self.gt.shape[-1] // downsample] = True
        valid = valid.flatten(1)
        stats = {}
        if not valid.any():
            return stats
        denominator = valid.sum(-1).clamp_min(1)
        for t in range(history.shape[1]):
            correct = (history[:, t].eq(gt_ids) & valid).sum(-1)
            stats[f'correct_count_step_{t}'] = correct.float().mean()
            stats[f'correct_rate_step_{t}'] = (correct.float() / denominator).mean()
            if t and t <= actual_steps:
                detail = _token_gain_diagnostics(history[:, t - 1], proposals[:, t], history[:, t], gt_ids, valid)
                for key in ('net_correct_gain', 'repaired_count', 'damaged_count',
                            'proposal_oracle_gain', 'proposal_net_correct_gain'):
                    stats[f'{key}_step_{t}'] = detail[key]
        stats['net_correct_gain'] = stats[f'correct_count_step_{history.shape[1] - 1}'] - stats['correct_count_step_0']
        return stats

    @torch.no_grad()
    def test(self):
        self.net_g.eval()
        self.net_evaluator.eval()
        net_g, evaluator = _bare(self, self.net_g), _bare(self, self.net_evaluator)
        val_opt = self.opt.get('val', {}) or {}
        multiple = max(1, (net_g.vq_downsample // net_g.scale_factor) * net_g.predictor_window_size)
        lq_pad, h_old, w_old = _flip_pad_to_multiple(self.lq, multiple)
        lq_feats, sft_cond = net_g.encode_lq_context(lq_pad)
        b, _, h, w = lq_feats.shape
        def callback(x, h_=0, w_=0, evaluator_call=False, cond_feats=None, proposal_ids=None):
            if evaluator_call:
                return evaluator(x, h_, w_, cond_feats=cond_feats if evaluator.use_cond_feats else None,
                                 proposal_ids=proposal_ids)
            return net_g.predict_logits(x)
        history, _, state = decode_evaluator(
            torch.full((b, h * w), -1, dtype=torch.long, device=self.device),
            lq_feats, callback, net_g.tokens_to_feats,
            num_refine_steps=resolve_num_refine_steps(val_opt),
            token_selection=val_opt.get('decode_sampling', val_opt.get('token_selection', 'multinomial')),
            sample_temperature=float(val_opt.get('sample_temperature', 1.0)),
            remask_budget_schedule=val_opt.get('remask_budget_schedule', 'full_cosine'),
            remask_budget_scale=val_opt.get('remask_budget_scale', 1.0),
            token_update_mode=resolve_token_update_mode(val_opt), return_state=True)
        self.refinement_diagnostics = {key: value.detach().float().mean() for key, value in state['diagnostics'].items()}
        self.refinement_diagnostics.update(self._validation_token_counts(
            history, state['proposal_seqs'], state['state_step'], h_old, w_old, h, w))
        self.output = net_g.decode_tokens(history[:, -1].reshape(b, 1, h, w), lq_pad, cond_feats=sft_cond)
        self.predictor_output = net_g.decode_tokens(history[:, 0].reshape(b, 1, h, w), lq_pad, cond_feats=sft_cond)
        self.output = self.output[..., :h_old * net_g.scale_factor, :w_old * net_g.scale_factor]
        self.predictor_output = self.predictor_output[..., :h_old * net_g.scale_factor, :w_old * net_g.scale_factor]
        self.refinement_diagnostics['image_delta_l1'] = (self.output.float() - self.predictor_output.float()).abs().mean()
        if self.is_train:
            self._set_training_modes()
        return self.output

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir=None):
        get_root_logger().info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img, save_as_dir)

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, save_as_dir):
        self._clear_validation_tensors()
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = (self.opt.get('val', {}).get('metrics') is not None
                        and len(getattr(self, 'metric_funcs', {})) > 0)
        if with_metrics:
            self.metric_results = {name: 0.0 for name in self.metric_funcs}
            self.predictor_metric_results = {name: 0.0 for name in self.metric_funcs}
            self._initialize_best_metric_results(dataset_name)
        diagnostics_sum = {}
        num_samples, metric_samples = 0, 0
        pbar = tqdm(total=len(dataloader), unit='batch')
        for val_data in dataloader:
            self.feed_data(val_data)
            self.test()
            batch_size = self.output.shape[0]
            num_samples += batch_size
            for key, value in self.refinement_diagnostics.items():
                diagnostics_sum[key] = diagnostics_sum.get(key, 0.0) + float(value) * batch_size
            # Keep CPU images for saving/8-bit conversion; release both GPU
            # outputs before LPIPS and other full-resolution metrics run.
            self.output = self.output.cpu()
            self.predictor_output = self.predictor_output.cpu()
            del self.lq
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            for j in range(batch_size):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][j]))[0]
                sr_img = tensor2img(self.output[j:j+1])
                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], 'image_results',
                            f'{current_iter}', f'{img_name}.png')
                    else:
                        suffix = self.opt.get('val', {}).get('suffix', self.opt['name'])
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{suffix}.png')
                    if save_as_dir:
                        imwrite(sr_img, osp.join(save_as_dir, f'{img_name}.png'))
                    imwrite(sr_img, save_img_path)
                if with_metrics and hasattr(self, 'gt'):
                    # Apply the same 8-bit/color/range conversion to both outputs.
                    pred = img2tensor(sr_img).unsqueeze(0).to(self.device) / 255.0
                    base_img = tensor2img(self.predictor_output[j:j+1])
                    base = img2tensor(base_img).unsqueeze(0).to(self.device) / 255.0
                    for name, func in self.metric_funcs.items():
                        self.metric_results[name] += func(pred, self.gt[j:j+1]).item()
                        self.predictor_metric_results[name] += func(base, self.gt[j:j+1]).item()
                    metric_samples += 1
                    del pred, base
            self._clear_validation_tensors()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()
        if num_samples == 0:
            get_root_logger().warning('Empty validation dataset; no metrics were computed.')
            return
        diag_text = ', '.join(f'{k}={v / num_samples:.6f}' for k, v in diagnostics_sum.items())
        get_root_logger().info('Validation refinement diagnostics: ' + diag_text)
        if with_metrics and metric_samples:
            for metric in self.metric_results:
                self.metric_results[metric] /= metric_samples
                self.predictor_metric_results[metric] /= metric_samples
                get_root_logger().info(
                    f'{metric}: predictor={self.predictor_metric_results[metric]:.6f}, '
                    f'iterative={self.metric_results[metric]:.6f}, '
                    f'delta={self.metric_results[metric] - self.predictor_metric_results[metric]:+.6f}')
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        elif with_metrics:
            get_root_logger().warning('No GT images in validation; no reference metrics computed.')

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        self._update_best_metric_and_save(
            dataset_name,
            current_iter,
            self.net_evaluator,
            'net_evaluator',
        )
        self._log_validation_metrics_with_best(current_iter, dataset_name, tb_logger)

    @torch.no_grad()
    def get_current_visuals(self):
        out_dict = OrderedDict(lq=self.lq.detach().cpu()[:4])
        if hasattr(self, 'output'):
            out_dict['output'] = self.output.detach().cpu()[:4]
        elif hasattr(self, '_training_token_ids'):
            # Decode only when image logging requests a preview, never for labels.
            with self.amp_autocast():
                preview = _bare(self, self.net_g).decode_tokens(
                    self._training_token_ids[:4], self.lq[:4])
            out_dict['output'] = preview.float().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()[:4]
        return OrderedDict((k, v.float().clamp(0, 1)) for k, v in out_dict.items())

    def save(self, epoch, current_iter):
        self.save_network(self.net_evaluator, 'net_evaluator', current_iter)
        self.save_training_state(epoch, current_iter)
