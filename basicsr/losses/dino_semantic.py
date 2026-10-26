"""DINO/DINOv2/DINOv3 semantic guidance used by SIPE-SR.

The module is deliberately training-only: the frozen teacher supervises the
SwinVQGAN tokenizer and/or the SIPE predictor, but it is not serialized into
``net_g`` and is never required at inference time.  It supports official
DINO-family PyTorch-Hub ViT backbones as well as a tiny deterministic teacher
for CPU unit tests and offline smoke tests.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

TensorCache = Dict[str, Union[torch.Tensor, Tuple[int, int]]]


class _TinyDINOBackbone(nn.Module):
    """A lightweight DINO-like patch-token teacher for validation only.

    It mimics the DINOv2 ``forward_features`` API by returning a normalized CLS
    token and normalized patch tokens.  Parameters are frozen by the outer loss
    class, but gradients still flow from the loss to the input image.
    """

    def __init__(self, patch_size: int = 8, embed_dim: int = 64) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(1216)
        with torch.no_grad():
            weight = torch.randn(self.proj.weight.shape, generator=gen, dtype=self.proj.weight.dtype)
            weight = weight / math.sqrt(3 * self.patch_size * self.patch_size)
            self.proj.weight.copy_(weight)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        patches = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        patches = self.norm(patches)
        cls = patches.mean(dim=1)
        return {
            'x_norm_clstoken': F.normalize(cls, dim=-1, eps=1e-6),
            'x_norm_patchtokens': F.normalize(patches, dim=-1, eps=1e-6),
        }


class DINOSemanticLoss(nn.Module):
    """Frozen DINO-family feature supervision.

    The teacher produces global CLS features and dense patch tokens.  SIPE uses
    them in three ways:

    1. image-level and patch-level semantic distillation between SR and HR;
    2. latent/codebook alignment through a lightweight learnable projector;
    3. semantic/difficulty token scores for SIPE mask sampling and token-loss
       reweighting.

    Args:
        backend: ``'torchhub'`` for official DINO-family models, or ``'tiny'``
            for the CPU-only smoke-test teacher.
        model_name: e.g. ``dinov2_vitb14_reg`` or ``dinov3_vitb16``.
            ``tiny`` and ``tiny_dino`` also select the tiny teacher.
        repo_or_dir/source: passed to ``torch.hub.load`` when ``backend`` is
            ``'torchhub'``.  Set ``source='local'`` and ``repo_or_dir`` to a
            local DINOv2/DINOv3 checkout for offline training.
        weights: optional checkpoint path or URL forwarded to ``torch.hub.load``.
            DINOv3 backbones commonly require this argument.
        pretrained: forwarded to ``torch.hub.load`` for official backbones.
        hub_kwargs: additional keyword arguments forwarded to ``torch.hub.load``.
        input_size: teacher resolution.  Use a multiple of ``patch_size``.
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vitb14_reg',
        repo_or_dir: str = 'facebookresearch/dinov2',
        source: str = 'github',
        backend: str = 'torchhub',
        weights: Optional[str] = None,
        pretrained: bool = True,
        hub_kwargs: Optional[Dict[str, Any]] = None,
        input_size: int = 224,
        patch_size: int = 14,
        global_weight: float = 0.05,
        local_weight: float = 0.10,
        latent_weight: float = 0.10,
        latent_alignment_mode: str = 'projector',
        latent_dim: int = 256,
        latent_projector_hidden_dim: int = 512,
        latent_projector_depth: int = 2,
        teacher_dim: int = 0,
        latent_proj_seed: int = 1216,
        normalize_latent: bool = True,
        tiny_embed_dim: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.repo_or_dir = repo_or_dir
        self.source = source
        self.backend = backend.lower()
        self.weights = weights
        self.pretrained = bool(pretrained)
        self.hub_kwargs = dict(hub_kwargs or {})
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.global_weight = float(global_weight)
        self.local_weight = float(local_weight)
        self.latent_weight = float(latent_weight)
        self.latent_alignment_mode = str(latent_alignment_mode).lower()
        self.latent_dim = int(latent_dim)
        self.latent_projector_hidden_dim = int(latent_projector_hidden_dim)
        self.latent_projector_depth = int(latent_projector_depth)
        self.latent_proj_seed = int(latent_proj_seed)
        self.normalize_latent = bool(normalize_latent)
        self.eps = eps

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('rand_proj', torch.empty(0), persistent=False)
        self._rand_proj_shape: Tuple[int, int] = (0, 0)

        self.teacher = self._load_teacher(int(tiny_embed_dim))
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.teacher_dim = int(teacher_dim) if int(teacher_dim) > 0 else self._infer_teacher_dim()
        self.latent_projector = self._build_latent_projector()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self):
        """Return training-only adapter parameters owned by this loss."""
        return [p for p in self.latent_projector.parameters() if p.requires_grad]

    def _load_teacher(self, tiny_embed_dim: int) -> nn.Module:
        name = self.model_name.lower()
        if self.backend == 'tiny' or name in {'tiny', 'tiny_dino', 'tiny_dinov2', 'tiny_dinov3'}:
            return _TinyDINOBackbone(patch_size=self.patch_size, embed_dim=tiny_embed_dim)

        hub_kwargs = dict(self.hub_kwargs)
        hub_kwargs.setdefault('pretrained', self.pretrained)
        if self.weights not in (None, ''):
            hub_kwargs['weights'] = self.weights
        return torch.hub.load(self.repo_or_dir, self.model_name, source=self.source, **hub_kwargs)

    def _infer_teacher_dim(self) -> int:
        dim = getattr(self.teacher, 'embed_dim', None)
        if dim is not None:
            return int(dim)

        name = self.model_name.lower()
        if 'vit7b' in name:
            return 4096
        if 'vith' in name:
            return 1280
        if 'vitg' in name:
            return 1536
        if 'vitl' in name:
            return 1024
        if 'vitb' in name:
            return 768
        if 'vits' in name:
            return 384
        if self.backend == 'tiny' or name in {'tiny', 'tiny_dino', 'tiny_dinov2'}:
            return int(getattr(self.teacher, 'embed_dim'))
        raise ValueError(
            f'Cannot infer DINO teacher feature dimension from model_name={self.model_name}. '
            'Set dino_opt.teacher_dim explicitly.'
        )

    def _build_latent_projector(self) -> nn.Module:
        if self.latent_alignment_mode in {'random', 'fixed_random', 'random_projection'}:
            return nn.Identity()
        if self.latent_alignment_mode not in {'projector', 'learned_projector', 'adapter'}:
            raise ValueError(
                f'Unsupported latent_alignment_mode: {self.latent_alignment_mode}. '
                'Expected projector or random.'
            )
        if self.latent_dim <= 0:
            raise ValueError('latent_dim must be positive when latent_alignment_mode=projector.')

        depth = max(1, self.latent_projector_depth)
        hidden_dim = self.latent_projector_hidden_dim
        if hidden_dim <= 0:
            hidden_dim = max(self.latent_dim, min(self.teacher_dim, 512))
        if depth == 1:
            return nn.Conv2d(self.latent_dim, self.teacher_dim, kernel_size=1)

        layers = [nn.Conv2d(self.latent_dim, hidden_dim, kernel_size=1), nn.GELU()]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1), nn.GELU()])
        layers.append(nn.Conv2d(hidden_dim, self.teacher_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def _autocast_off(self, x: torch.Tensor):
        if x.device.type == 'cuda':
            return torch.amp.autocast(device_type='cuda', enabled=False)
        return contextlib.nullcontext()

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0, 1)
        if self.input_size > 0:
            target_size = (self.input_size // self.patch_size) * self.patch_size
            target_size = max(self.patch_size, target_size)
            if x.shape[-2] != target_size or x.shape[-1] != target_size:
                x = F.interpolate(x, size=(target_size, target_size), mode='bicubic', align_corners=False)
                x = x.clamp(0, 1)
        return (x - self.mean.to(x)) / self.std.to(x)

    @staticmethod
    def _infer_hw(num_tokens: int, fallback_hw: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            return side, side
        if fallback_hw is not None and fallback_hw[0] * fallback_hw[1] == num_tokens:
            return fallback_hw
        raise RuntimeError(f'Cannot infer a 2D grid from {num_tokens} patch tokens.')

    def extract_features(self, x: torch.Tensor, detach: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Return DINO CLS token, patch tokens and patch grid size."""
        x_in = self._preprocess(x)
        patch_hw = None
        if self.input_size > 0:
            target_size = (self.input_size // self.patch_size) * self.patch_size
            patch_hw = (target_size // self.patch_size, target_size // self.patch_size)

        def _forward():
            if hasattr(self.teacher, 'forward_features'):
                out = self.teacher.forward_features(x_in)
                if isinstance(out, dict):
                    cls = out.get('x_norm_clstoken', None)
                    patches = out.get('x_norm_patchtokens', None)
                    if cls is not None and patches is not None:
                        return cls, patches
            if hasattr(self.teacher, 'get_intermediate_layers'):
                out = self.teacher.get_intermediate_layers(x_in, n=1, return_class_token=True)
                out0 = out[0]
                if isinstance(out0, tuple):
                    patches, cls = out0
                else:
                    cls, patches = out0[:, 0], out0[:, 1:]
                return cls, patches
            out = self.teacher(x_in)
            if out.dim() == 2:
                return out, out[:, None, :]
            return out[:, 0], out[:, 1:]

        if detach:
            with torch.no_grad(), self._autocast_off(x_in):
                cls, patches = _forward()
            cls, patches = cls.detach(), patches.detach()
        else:
            with self._autocast_off(x_in):
                cls, patches = _forward()

        hw = self._infer_hw(patches.shape[1], patch_hw)
        return cls.float(), patches.float(), hw

    @torch.no_grad()
    def prepare_features(self, x: torch.Tensor) -> TensorCache:
        """Cache target/reference features to avoid repeated teacher forwards."""
        cls, patches, hw = self.extract_features(x, detach=True)
        return {'cls': cls, 'patch': patches, 'hw': hw}

    @staticmethod
    def _cache_to_tuple(cache: TensorCache) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        return cache['cls'], cache['patch'], cache['hw']  # type: ignore[return-value]

    def image_pair_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        target_cache: Optional[TensorCache] = None,
        global_weight: Optional[float] = None,
        local_weight: Optional[float] = None,
        prefix: str = '',
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        gw = self.global_weight if global_weight is None else float(global_weight)
        lw = self.local_weight if local_weight is None else float(local_weight)
        if gw <= 0 and lw <= 0:
            return losses

        pred_cls, pred_patch, _ = self.extract_features(pred, detach=False)
        if target_cache is None:
            with torch.no_grad():
                tgt_cls, tgt_patch, _ = self.extract_features(target, detach=True)
        else:
            tgt_cls, tgt_patch, _ = self._cache_to_tuple(target_cache)

        if gw > 0:
            pred_cls = F.normalize(pred_cls, dim=-1, eps=self.eps)
            tgt_cls = F.normalize(tgt_cls, dim=-1, eps=self.eps)
            losses[f'{prefix}l_dino_global'] = gw * (1.0 - (pred_cls * tgt_cls).sum(dim=-1)).mean()

        if lw > 0:
            pred_patch = F.normalize(pred_patch, dim=-1, eps=self.eps)
            tgt_patch = F.normalize(tgt_patch, dim=-1, eps=self.eps)
            losses[f'{prefix}l_dino_local'] = lw * (1.0 - (pred_patch * tgt_patch).sum(dim=-1)).mean()

        return losses

    def _project_patch_features(self, patch_map: torch.Tensor, out_dim: int) -> torch.Tensor:
        in_dim = patch_map.shape[1]
        shape = (in_dim, int(out_dim))
        if self.rand_proj.numel() == 0 or self._rand_proj_shape != shape:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(self.latent_proj_seed)
            proj = torch.randn(shape, generator=gen) / math.sqrt(float(out_dim))
            self.rand_proj = proj.to(device=patch_map.device, dtype=patch_map.dtype)
            self._rand_proj_shape = shape
        proj = self.rand_proj.to(device=patch_map.device, dtype=patch_map.dtype)
        return torch.einsum('bchw,co->bohw', patch_map, proj)

    def latent_alignment_loss(
        self,
        latent: Optional[torch.Tensor],
        target: torch.Tensor,
        *,
        target_cache: Optional[TensorCache] = None,
        weight: Optional[float] = None,
        name: str = 'l_dino_latent',
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        lw = self.latent_weight if weight is None else float(weight)
        if lw <= 0 or latent is None:
            return losses

        with torch.no_grad():
            if target_cache is None:
                _, tgt_patch, hw = self.extract_features(target, detach=True)
            else:
                _, tgt_patch, hw = self._cache_to_tuple(target_cache)
            b, _, c = tgt_patch.shape
            tgt_patch = tgt_patch.transpose(1, 2).reshape(b, c, hw[0], hw[1]).float()
            if tgt_patch.shape[-2:] != latent.shape[-2:]:
                tgt_patch = F.interpolate(tgt_patch, size=latent.shape[-2:], mode='bicubic', align_corners=False)

        if self.latent_alignment_mode in {'random', 'fixed_random', 'random_projection'}:
            pred = latent.float()
            with torch.no_grad():
                tgt = self._project_patch_features(tgt_patch, latent.shape[1]).float()
        else:
            if latent.shape[1] != self.latent_dim:
                raise RuntimeError(
                    f'latent_alignment_mode=projector expects latent_dim={self.latent_dim}, '
                    f'but got latent with {latent.shape[1]} channels. '
                    'Set dino_opt.latent_dim to the VQ latent/channel dimension.'
                )
            pred = self.latent_projector(latent.float())
            tgt = tgt_patch.float()
        if self.normalize_latent:
            pred = F.normalize(pred, dim=1, eps=self.eps)
            tgt = F.normalize(tgt, dim=1, eps=self.eps)
        losses[name] = lw * (1.0 - (pred * tgt).sum(dim=1)).mean()
        return losses

    @torch.no_grad()
    def token_score_map(
        self,
        target: torch.Tensor,
        out_hw: Tuple[int, int],
        *,
        reference: Optional[torch.Tensor] = None,
        target_cache: Optional[TensorCache] = None,
        reference_cache: Optional[TensorCache] = None,
        saliency_weight: float = 0.0,
        difficulty_weight: float = 1.0,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Return an LR-anchor DINO difficulty score map on a token grid.

        The score is the absolute cosine distance ``(1 - cos(HR, LR_up)) / 2``.
        It measures where the LR observation has lost recoverable semantic
        structure without mean-feature outlier saliency or per-image min-max
        amplification.  ``saliency_weight`` and ``normalize`` are accepted only
        for compatibility with older configs.
        """
        if target_cache is None:
            _, tgt_patch, hw = self.extract_features(target, detach=True)
        else:
            _, tgt_patch, hw = self._cache_to_tuple(target_cache)
        tgt_n = F.normalize(tgt_patch.float(), dim=-1, eps=self.eps)

        score = torch.zeros(tgt_patch.shape[:2], dtype=torch.float32, device=target.device)
        if reference is not None and difficulty_weight > 0:
            if reference_cache is None:
                _, ref_patch, ref_hw = self.extract_features(reference, detach=True)
            else:
                _, ref_patch, ref_hw = self._cache_to_tuple(reference_cache)
            ref_n = F.normalize(ref_patch.float(), dim=-1, eps=self.eps)
            if ref_hw != hw or ref_n.shape[1] != tgt_n.shape[1]:
                b, _, c = ref_n.shape
                ref_map = ref_n.transpose(1, 2).reshape(b, c, ref_hw[0], ref_hw[1])
                ref_map = F.interpolate(ref_map, size=hw, mode='bicubic', align_corners=False)
                ref_n = ref_map.flatten(2).transpose(1, 2)
                ref_n = F.normalize(ref_n, dim=-1, eps=self.eps)
            difficulty = 0.5 * (1.0 - (tgt_n * ref_n).sum(dim=-1))
            score = (float(difficulty_weight) * difficulty).clamp(0, 1)

        score = score.reshape(score.shape[0], 1, hw[0], hw[1])
        score = F.interpolate(score, size=out_hw, mode='bicubic', align_corners=False)
        return score.clamp(0, 1)

    @torch.no_grad()
    def token_error_map(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        out_hw: Tuple[int, int],
        *,
        pred_cache: Optional[TensorCache] = None,
        target_cache: Optional[TensorCache] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Return local DINO feature disagreement on an arbitrary token grid.

        Stage 3 combines this training-only error with binary ID mismatch to
        build continuous risk targets (or legacy BCE weights). A sampled token
        can differ from the GT code but still be
        visually tolerable, while a small number of semantically important token
        errors can dominate perceptual quality.  The returned map measures
        patch-token absolute cosine distance between decoded prediction and HR
        target, then resizes it to the SwinVQGAN latent grid.  ``normalize`` is
        accepted only for compatibility with older configs; per-image min-max
        normalization is intentionally not applied.
        """
        if pred_cache is None:
            _, pred_patch, pred_hw = self.extract_features(pred, detach=True)
        else:
            _, pred_patch, pred_hw = self._cache_to_tuple(pred_cache)
        if target_cache is None:
            _, tgt_patch, tgt_hw = self.extract_features(target, detach=True)
        else:
            _, tgt_patch, tgt_hw = self._cache_to_tuple(target_cache)

        pred_patch = F.normalize(pred_patch.float(), dim=-1, eps=self.eps)
        tgt_patch = F.normalize(tgt_patch.float(), dim=-1, eps=self.eps)
        if pred_hw != tgt_hw or pred_patch.shape[1] != tgt_patch.shape[1]:
            b, _, c = pred_patch.shape
            pred_map = pred_patch.transpose(1, 2).reshape(b, c, pred_hw[0], pred_hw[1])
            pred_map = F.interpolate(pred_map, size=tgt_hw, mode='bicubic', align_corners=False)
            pred_patch = pred_map.flatten(2).transpose(1, 2)
            pred_patch = F.normalize(pred_patch, dim=-1, eps=self.eps)

        score = 0.5 * (1.0 - (pred_patch * tgt_patch).sum(dim=-1))
        score = score.reshape(score.shape[0], 1, tgt_hw[0], tgt_hw[1])
        score = F.interpolate(score, size=out_hw, mode='bicubic', align_corners=False)
        return score.clamp(0, 1)

    @torch.no_grad()
    def semantic_mask(
        self,
        x: torch.Tensor,
        mask_ratio: Union[float, Tuple[float, float]] = 0.30,
        mode: str = 'semantic',
        random_ratio: float = 0.0,
        semantic_prob: Optional[float] = None,
        feature_cache: Optional[TensorCache] = None,
    ) -> torch.Tensor:
        """Build a patch-aligned image-space binary mask.

        ``mode='mixed'`` follows MacTok-style masking: random masking and DINO
        semantic masking are sampled as two alternative strategies instead of
        blending random noise into semantic scores.  ``mask_ratio`` may be a
        scalar or a two-value range, in which case a ratio is uniformly sampled
        per image in the batch.  ``feature_cache`` may be supplied when the
        same target image features are also used by DINO losses.
        """
        b, _, h, w = x.shape
        if isinstance(mask_ratio, (tuple, list)):
            if len(mask_ratio) != 2:
                raise ValueError(f'mask_ratio range must have two values, got {mask_ratio}.')
            ratio_min = float(mask_ratio[0])
            ratio_max = float(mask_ratio[1])
            if ratio_min > ratio_max:
                ratio_min, ratio_max = ratio_max, ratio_min
            ratios = torch.empty(b, device=x.device).uniform_(ratio_min, ratio_max)
        elif torch.is_tensor(mask_ratio):
            ratios = mask_ratio.to(device=x.device, dtype=torch.float32).flatten()
            if ratios.numel() == 1:
                ratios = ratios.expand(b)
            elif ratios.numel() != b:
                raise ValueError(f'mask_ratio tensor must have 1 or {b} values, got {ratios.numel()}.')
        else:
            ratios = x.new_full((b,), float(mask_ratio))
        ratios = ratios.float().clamp(0.0, 1.0)
        if torch.all(ratios <= 0):
            return x.new_zeros(b, 1, h, w)

        if feature_cache is None:
            _, patch, hw = self.extract_features(x, detach=True)
        else:
            _, patch, hw = self._cache_to_tuple(feature_cache)
        n = patch.shape[1]
        k = torch.round(ratios * n).long().clamp(0, n)

        mode = mode.lower()
        if mode == 'random':
            score = torch.rand(b, n, device=x.device)
        elif mode == 'semantic':
            patch_n = F.normalize(patch.float(), dim=-1, eps=self.eps)
            center = F.normalize(patch_n.mean(dim=1, keepdim=True), dim=-1, eps=self.eps)
            score = 1.0 - (patch_n * center).sum(dim=-1)
        elif mode == 'mixed':
            patch_n = F.normalize(patch.float(), dim=-1, eps=self.eps)
            center = F.normalize(patch_n.mean(dim=1, keepdim=True), dim=-1, eps=self.eps)
            semantic_score = 1.0 - (patch_n * center).sum(dim=-1)
            random_score = torch.rand(b, n, device=x.device)
            if semantic_prob is None:
                # Backward compatibility: under MacTok-style mixed masking,
                # random_ratio is interpreted as the random-strategy probability.
                semantic_prob = 1.0 - float(max(0.0, min(1.0, random_ratio))) if random_ratio else 0.5
            semantic_prob = float(max(0.0, min(1.0, semantic_prob)))
            use_semantic = torch.rand(b, 1, device=x.device) < semantic_prob
            score = torch.where(use_semantic, semantic_score, random_score)
        else:
            raise ValueError(f'Unsupported DINO mask mode: {mode}. Expected semantic, random, or mixed.')

        patch_mask = torch.zeros(b, n, device=x.device, dtype=x.dtype)
        for i in range(b):
            if k[i] > 0:
                idx = torch.topk(score[i], k=int(k[i].item()), dim=0).indices
                patch_mask[i].scatter_(0, idx, 1.0)
        patch_mask = patch_mask.reshape(b, 1, hw[0], hw[1])
        return F.interpolate(patch_mask, size=(h, w), mode='nearest')

    @staticmethod
    def apply_mask(x: torch.Tensor, mask: torch.Tensor, fill: float = 0.5) -> torch.Tensor:
        return x * (1.0 - mask) + float(fill) * mask

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        *,
        target_cache: Optional[TensorCache] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = self.image_pair_loss(pred, target, target_cache=target_cache)
        losses.update(self.latent_alignment_loss(latent, target, target_cache=target_cache))
        return losses


# Backward-compatible alias for older reports.
DINOv2SemanticLoss = DINOSemanticLoss
