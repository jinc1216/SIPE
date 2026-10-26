"""SIPE-SR architectures.

This file implements semantic-guided iterative token prediction and evaluation
for blind image super-resolution. The high-quality codebook/decoder is
SwinVQGAN (VQModel).
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops.layers.torch import Rearrange

from basicsr.utils.registry import ARCH_REGISTRY
from .network_swinir import RSTB
from .fema_utils import ResBlock
from .vqgan_swin_arch import VQModel


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert token masks to bool with shape (B, 1, H, W)."""
    if mask.dtype != torch.bool:
        mask = mask > 0
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask


def _zero_last_conv(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv2d):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


class SIPEPredictor(nn.Module):
    """Swin token predictor for SIPE-SR.

    Input features are latent maps in the SwinVQGAN embedding space.  The module
    predicts one codebook distribution for each latent location.
    """

    def __init__(self,
                 input_resolution: Tuple[int, int] = (32, 32),
                 embed_dim: int = 256,
                 blk_depth: int = 6,
                 blk_num: int = 4,
                 num_heads: int = 8,
                 window_size: int = 8,
                 codebook_size: int = 1024,
                 **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList([
            RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            for _ in range(blk_num)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.idx_pred_layer = nn.Linear(embed_dim, codebook_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        for block in self.swin_blks:
            x = block(x, (h, w))
        x = self.norm_out(x)
        return self.idx_pred_layer(x)


class SRFeatureEncoder(nn.Module):
    """Map LR images to the HR SwinVQGAN latent grid.

    SwinVQGAN downsamples HR images by 8.  Therefore the LR encoder downsamples
    by 8 / scale_factor so that output spatial size is HR/8:
        x2: LR -> /4, x4: LR -> /2.
    """

    def __init__(self,
                 in_ch: int,
                 feat_dim: int,
                 scale_factor: int,
                 vq_downsample: int = 8,
                 base_ch: int = 64,
                 blk_depth: int = 6,
                 swin_blk_num: int = 2,
                 window_size: int = 8,
                 num_heads: int = 8):
        super().__init__()
        if scale_factor not in (2, 4):
            raise ValueError(f'SIPE-SR currently supports scale_factor 2 or 4, got {scale_factor}.')
        if vq_downsample % scale_factor != 0:
            raise ValueError(f'vq_downsample={vq_downsample} must be divisible by scale_factor={scale_factor}.')
        down_factor = vq_downsample // scale_factor
        depth = int(np.log2(down_factor))
        if 2 ** depth != down_factor:
            raise ValueError(f'Invalid down_factor={down_factor} for scale_factor={scale_factor}.')

        self.scale_factor = int(scale_factor)
        blocks = [nn.Conv2d(in_ch, base_ch, 3, 1, 1)]
        self.pyramid_channels = {self.scale_factor: int(base_ch)}
        cur_ch = base_ch
        stride = self.scale_factor
        for _ in range(depth):
            out_ch = cur_ch * 2
            blocks.append(nn.Sequential(
                nn.Conv2d(cur_ch, out_ch, 3, stride=2, padding=1),
                ResBlock(out_ch, out_ch),
                ResBlock(out_ch, out_ch),
            ))
            cur_ch = out_ch
            stride *= 2
            self.pyramid_channels[stride] = int(cur_ch)
        blocks.append(nn.Conv2d(cur_ch, feat_dim, 1, 1, 0))
        self.body = nn.Sequential(*blocks)
        self.swin = _LatentSwinLayers(
            embed_dim=feat_dim,
            swin_dim=feat_dim,
            blk_num=swin_blk_num,
            blk_depth=blk_depth,
            window_size=window_size,
            num_heads=num_heads,
        )

    def _forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the pre-projection feature and its HR-stride pyramid."""
        x = self.body[0](x)
        stride = self.scale_factor
        pyramid = {str(stride): x}
        for index in range(1, len(self.body) - 1):
            x = self.body[index](x)
            stride *= 2
            pyramid[str(stride)] = x
        return x, pyramid

    def forward_with_pyramid(self,
                             x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Encode once and expose intermediate features for SFT."""
        x, pyramid = self._forward_backbone(x)
        latent = self.swin(self.body[-1](x))
        return latent, pyramid

    def forward_pyramid(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build only the shared pyramid for standalone token decoding."""
        _, pyramid = self._forward_backbone(x)
        return pyramid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, _ = self.forward_with_pyramid(x)
        return latent


class _LatentSwinLayers(nn.Module):
    """Swin layers on latent maps."""

    def __init__(self,
                 embed_dim: int,
                 swin_dim: int = 128,
                 blk_num: int = 2,
                 blk_depth: int = 6,
                 window_size: int = 8,
                 num_heads: int = 8,
                 input_resolution: Tuple[int, int] = (32, 32),
                 **kwargs):
        super().__init__()
        self.conv_in = nn.Conv2d(embed_dim, swin_dim, 3, 1, 1)
        self.conv_out = nn.Conv2d(swin_dim, embed_dim, 3, 1, 1)
        self.swin_blks = nn.ModuleList([
            RSTB(swin_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            for _ in range(blk_num)
        ])

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x_in)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        for block in self.swin_blks:
            x = block(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.conv_out(x)
        return x + x_in


class GuidedSRRefiner(nn.Module):
    """Lightweight HR-space refiner conditioned on bicubic LR.

    This SR-aware image-space fusion decodes texture from SwinVQGAN tokens,
    upsamples LR for structure, and predicts a small residual.  The residual is
    gated by a learnable scalar to keep the VQ prior stable at initialization.
    """

    def __init__(self, channels: int = 64, blocks: int = 4, residual_scale: float = 0.1):
        super().__init__()
        layers = [nn.Conv2d(6, channels, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(blocks):
            layers.append(ResBlock(channels, channels))
        layers.append(nn.Conv2d(channels, 3, 3, 1, 1))
        self.body = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, decoded: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        up_lq = F.interpolate(lq.float(), size=decoded.shape[-2:], mode='bicubic', align_corners=False)
        residual = self.body(torch.cat([decoded, up_lq], dim=1))
        return (decoded + self.residual_scale * residual).clamp(0, 1)


class ResidualSFTBlock(nn.Module):
    """Residual SFT with fixed unit-strength condition injection."""

    def __init__(self, channels: int, zero_init: bool = True):
        super().__init__()
        self.encode = ResBlock(2 * channels, channels)
        self.scale = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )
        if zero_init:
            _zero_last_conv(self.scale)
            _zero_last_conv(self.shift)

    def forward(self,
                cond_feat: torch.Tensor,
                dec_feat: torch.Tensor) -> torch.Tensor:
        if cond_feat.shape[-2:] != dec_feat.shape[-2:]:
            cond_feat = F.interpolate(cond_feat, size=dec_feat.shape[-2:], mode='bilinear', align_corners=False)
        cond_feat = cond_feat.to(dtype=dec_feat.dtype, device=dec_feat.device)
        h = self.encode(torch.cat([cond_feat, dec_feat], dim=1))
        scale = self.scale(h)
        shift = self.shift(h)
        # Inject the complete affine residual with fixed unit strength.
        residual = dec_feat * scale + shift
        return dec_feat + residual


class SharedLQSFTConditionAdapter(nn.Module):
    """Adapt the token encoder pyramid for SFT without a second image encoder.

    Shared encoder activations are detached before the trainable adapters.
    Reconstruction losses therefore train the adapters and
    SFT blocks, but cannot update the token encoder through the shortcut path.
    """

    def __init__(self,
                 source_channels: Dict[int, int],
                 level_channels: Dict[int, int]):
        super().__init__()
        source_channels = {int(k): int(v) for k, v in source_channels.items()}
        self.level_channels = {int(k): int(v) for k, v in level_channels.items()}
        missing = sorted(set(self.level_channels) - set(source_channels))
        if missing:
            raise ValueError(
                f'Shared LQ encoder has no features for SFT levels {missing}; '
                f'available levels are {sorted(source_channels)}.')

        self.output_adapters = nn.ModuleDict({
            str(level): nn.Sequential(
                nn.Conv2d(source_channels[level], channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels, channels),
            )
            for level, channels in self.level_channels.items()
        })

    def forward(self,
                encoder_feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        missing = sorted(
            str(level) for level in self.level_channels if str(level) not in encoder_feats)
        if missing:
            raise ValueError(f'Missing shared LQ encoder features for SFT levels {missing}.')

        # Share feature values and forward computation while stopping gradients
        # between the SFT shortcut and token encoder.
        return {
            str(level): self.output_adapters[str(level)](encoder_feats[str(level)].detach())
            for level in self.level_channels
        }


@ARCH_REGISTRY.register()
class SIPEEvaluator(nn.Module):
    """Paired token-correctness transition evaluator (Stage 3 only).

    One four-class head predicts (old correct, proposed correct):
    0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
    Expected correctness gain is p[1]-p[2]. On a self-pair (S,S),
    p[0]+p[1] estimates the current error probability for remask ranking.
    GT is used only to construct training labels, never as a network input.
    """

    def __init__(self, input_resolution=(32, 32), embed_dim=256, blk_depth=6,
                 blk_num=2, num_heads=8, window_size=8, codebook_size=1024,
                 use_cond_feats=True, token_embedding_mode='independent',
                 use_score_feats=False, num_score_channels=4,
                 cond_scale=1.0, score_scale=1.0, **kwargs):
        super().__init__()
        if token_embedding_mode not in ('independent', 'frozen_vq'):
            raise ValueError('token_embedding_mode must be independent or frozen_vq.')
        if use_score_feats:
            raise ValueError('Correctness-transition evaluation does not use score features.')
        if float(cond_scale) != 1.0 or float(score_scale) != 1.0:
            raise ValueError('Correctness-transition fusion has no adjustable feature scales.')
        self.embed_dim = int(embed_dim)
        self.codebook_size = int(codebook_size)
        self.token_embedding_mode = token_embedding_mode
        self.use_cond_feats = bool(use_cond_feats)
        self.use_score_feats = False
        self.tok_emb = nn.Embedding(self.codebook_size, self.embed_dim)
        self.tok_emb.weight.requires_grad_(token_embedding_mode == 'independent')
        self.tok_norm = nn.LayerNorm(self.embed_dim)
        # Ordered concatenation: the head sees both ACTUAL candidate IDs.
        self.pair_proj = nn.Linear(2 * self.embed_dim, self.embed_dim)
        if self.use_cond_feats:
            self.cond_norm = nn.LayerNorm(self.embed_dim)
            self.cond_proj = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.cond_norm = self.cond_proj = None
        self.swin_blks = nn.ModuleList([
            RSTB(self.embed_dim, input_resolution, blk_depth, num_heads,
                 window_size, patch_size=1, **kwargs) for _ in range(blk_num)])
        self.norm_out = nn.LayerNorm(self.embed_dim)
        self.transition_head = nn.Linear(self.embed_dim, 4)
        self.register_buffer('critic_format_version', torch.tensor(
            31 if token_embedding_mode == 'independent' else 32, dtype=torch.long))
        self.register_buffer('token_target_mode_id', torch.tensor(31, dtype=torch.long))
        self.register_buffer('conditional_format', torch.tensor(int(self.use_cond_feats), dtype=torch.long))

    @property
    def token_target_mode(self):
        return 'correctness_transition'

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Validate checkpoint structure and format even under strict=False.
        expected = self.state_dict()
        invalid = [name for name, value in expected.items()
                   if prefix + name not in state_dict
                   or state_dict[prefix + name].shape != value.shape]
        for name in ('critic_format_version', 'token_target_mode_id', 'conditional_format'):
            value = state_dict.get(prefix + name)
            if (value is None or value.numel() != 1 or value.dtype != torch.long
                    or int(value.item()) != int(expected[name].item())):
                invalid.append(name)
        if invalid:
            raise RuntimeError(
                'Incompatible correctness-transition evaluator checkpoint: '
                + ', '.join(sorted(set(invalid))) + '. Keep Stage 2 and start a new '
                'Stage-3 run with pretrain_network_evaluator/resume_state=null; '
                'legacy risk and whole-image comparison checkpoints cannot be resumed.')
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def prepare_token_embedding(self, codebook, checkpoint_loaded=False):
        if tuple(codebook.shape) != tuple(self.tok_emb.weight.shape):
            raise ValueError('Predictor codebook and evaluator embedding sizes differ.')
        if self.token_embedding_mode == 'independent':
            self.tok_emb.weight.requires_grad_(True)
            return
        if checkpoint_loaded and not torch.equal(
                self.tok_emb.weight, codebook.detach().to(self.tok_emb.weight)):
            raise ValueError('Evaluator checkpoint and frozen predictor use different codebooks.')
        self.tok_emb.weight.copy_(codebook.detach().to(self.tok_emb.weight))
        self.tok_emb.weight.requires_grad_(False)

    def _check_ids(self, ids, h, w):
        if ids.dim() in (3, 4):
            ids = ids.flatten(1)
        if ids.dim() != 2 or ids.shape[1] != h * w or h < 1 or w < 1:
            raise ValueError('Evaluator requires complete (B,H*W) token sequences.')
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError('Evaluator IDs must have an integer dtype.')
        if (ids < 0).any() or (ids >= self.codebook_size).any():
            raise ValueError('Evaluator sees complete candidate IDs, never masks.')
        return ids.long()

    def forward(self, token_ids, h, w, cond_feats=None, *, proposal_ids=None):
        old = self._check_ids(token_ids, h, w)
        new = old if proposal_ids is None else self._check_ids(proposal_ids, h, w)
        if new.shape != old.shape or new.device != old.device:
            raise ValueError('Old and proposed IDs must share shape/device.')
        b = old.shape[0]
        old_feats = self.tok_norm(self.tok_emb(old))
        new_feats = self.tok_norm(self.tok_emb(new))
        x = self.pair_proj(torch.cat((old_feats, new_feats), dim=-1))
        if self.use_cond_feats:
            if cond_feats is None or tuple(cond_feats.shape) != (b, self.embed_dim, h, w):
                raise ValueError('Conditional evaluator needs aligned frozen LR features.')
            cond = cond_feats.detach().flatten(2).transpose(1, 2).to(x.dtype)
            x = x + self.cond_proj(self.cond_norm(cond))
        for block in self.swin_blks:
            x = block(x, (h, w))
        return self.transition_head(self.norm_out(x)).float()


@ARCH_REGISTRY.register()
class SIPETokenNet(nn.Module):
    """SIPE predictor adapted to x2/x4 image super-resolution.

    Stage-1 prior: SwinVQGAN (VQModel).
    Stage-2 predictor: LR image -> SwinVQGAN latent feature -> SIPE token predictor.
    Stage-3: only SIPEEvaluator is trained; the complete Stage-2 predictor,
    including its VQ prior and decoder, remains frozen.
    """

    def __init__(self,
                 *,
                 in_channel: int = 3,
                 scale_factor: int = 4,
                 gt_resolution: int = 256,
                 LQ_stage: bool = True,
                 code_only: bool = False,
                 vqgan_opt: Optional[dict] = None,
                 predictor_blk_depth: int = 6,
                 predictor_blk_num: int = 4,
                 predictor_num_heads: int = 8,
                 predictor_window_size: int = 8,
                 lq_base_ch: int = 64,
                 lq_swin_blk_depth: int = 6,
                 lq_swin_blk_num: int = 2,
                 use_guided_refine: bool = False,
                 use_sft_fusion: bool = True,
                 sft_levels=None,
                 sft_cond_source: str = 'shared_lq',
                 sft_zero_init: bool = True,
                 sft_level_channels: Optional[Dict[int, int]] = None,
                 refine_blocks: int = 4,
                 refine_channels: int = 64,
                 **ignore_kwargs):
        super().__init__()
        if vqgan_opt is None:
            raise ValueError('SIPETokenNet requires vqgan_opt copied from the SwinVQGAN stage-1 config.')
        if scale_factor not in (2, 4):
            raise ValueError(f'Only x2/x4 SR is supported; got scale_factor={scale_factor}.')

        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.scale_factor = int(scale_factor)
        self.code_only = code_only
        self.predictor_window_size = int(predictor_window_size)
        self.use_sft_fusion = bool(use_sft_fusion)
        self.sft_cond_source = str(sft_cond_source).lower()
        self.sft_uses_shared_lq = False

        self.vqgan = VQModel(**vqgan_opt)
        self.embed_dim = int(vqgan_opt['embed_dim'])
        self.codebook_size = int(vqgan_opt['n_embed'])
        self.refiner = None
        self.sft_condition = None
        self.sft_blocks = nn.ModuleDict()

        # VQModel with ch_mult length m downsamples HR by 2**(m-1).
        self.vq_downsample = 2 ** (len(vqgan_opt['ddconfig']['ch_mult']) - 1)
        if self.vq_downsample != 8:
            # Other downsample factors must be divisible by the SR scale.
            if self.vq_downsample % self.scale_factor != 0:
                raise ValueError(
                    f'VQ downsample factor {self.vq_downsample} must be divisible by scale_factor {self.scale_factor}.')
        default_sft_levels = [level for level in (self.vq_downsample // 2, 2) if level >= 2]
        self.sft_levels = [int(v) for v in (sft_levels or default_sft_levels)]
        self.sft_levels = sorted(set(v for v in self.sft_levels if v >= 2), reverse=True)

        if LQ_stage:
            for p in self.vqgan.parameters():
                p.requires_grad = False
            self.vqgan.eval()

            self.lq_encoder = SRFeatureEncoder(
                in_ch=in_channel,
                feat_dim=self.embed_dim,
                scale_factor=self.scale_factor,
                vq_downsample=self.vq_downsample,
                base_ch=lq_base_ch,
                blk_depth=lq_swin_blk_depth,
                swin_blk_num=lq_swin_blk_num,
                window_size=predictor_window_size,
                num_heads=predictor_num_heads,
            )
            self.transformer = SIPEPredictor(
                embed_dim=self.embed_dim,
                blk_depth=predictor_blk_depth,
                blk_num=predictor_blk_num,
                num_heads=predictor_num_heads,
                window_size=predictor_window_size,
                codebook_size=self.codebook_size,
            )
            if self.use_sft_fusion:
                level_channels = self._infer_decoder_sft_channels(vqgan_opt.get('ddconfig') or {})
                if sft_level_channels is not None:
                    level_channels.update({int(k): int(v) for k, v in sft_level_channels.items()})
                level_channels = {level: level_channels[level] for level in self.sft_levels if level in level_channels}
                if not level_channels:
                    raise ValueError(f'Could not infer decoder channels for requested SFT levels: {self.sft_levels}.')

                if self.sft_cond_source in (
                    'shared_lq',
                    'shared_lq_encoder',
                    'ipc_shared_lq',
                ):
                    self.sft_uses_shared_lq = True
                    self.sft_condition = SharedLQSFTConditionAdapter(
                        source_channels=self.lq_encoder.pyramid_channels,
                        level_channels=level_channels,
                    )
                else:
                    raise ValueError(
                        f'Unsupported sft_cond_source={sft_cond_source}. '
                        'Use "shared_lq" (aliases: "shared_lq_encoder", "ipc_shared_lq").')
                self.sft_blocks = nn.ModuleDict({
                    str(level): ResidualSFTBlock(channels, zero_init=bool(sft_zero_init))
                    for level, channels in level_channels.items()
                })
            if use_guided_refine:
                self.refiner = GuidedSRRefiner(channels=refine_channels, blocks=refine_blocks)

    @staticmethod
    def _infer_decoder_sft_channels(ddconfig: dict) -> Dict[int, int]:
        nf = int(ddconfig.get('nf', 128))
        ch_mult = list(ddconfig.get('ch_mult', [1, 2, 2, 4]))
        level_channels = {}
        for idx in range(1, len(ch_mult)):
            stride = 2 ** idx
            level_channels[stride] = int(nf * ch_mult[idx])
        return level_channels

    def _decode_quant_with_sft(self,
                               quant: torch.Tensor,
                               lq: torch.Tensor,
                               cond_feats: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if cond_feats is None:
            if self.sft_uses_shared_lq:
                encoder_feats = self.lq_encoder.forward_pyramid(lq)
                cond_feats = self.sft_condition(encoder_feats)
            else:
                cond_feats = self.sft_condition(lq)
        hr_h = int(lq.shape[-2] * self.scale_factor)
        x = self.vqgan.to_decoder_space(quant)

        for block in self.vqgan.decoder.blocks:
            if block.__class__.__name__ == 'Upsample':
                stride = max(1, int(round(hr_h / float(x.shape[-2]))))
                key = str(stride)
                if key in self.sft_blocks and key in cond_feats:
                    x = self.sft_blocks[key](cond_feats[key], x)
            x = block(x)

        decoded = (x + 1) / 2
        return decoded.clamp(0, 1)

    def encode_lq_feats(self, lq: torch.Tensor) -> torch.Tensor:
        return self.lq_encoder(lq)

    def encode_lq_context(self,
                          lq: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Encode token and SFT conditions together so shared features run once."""
        if not self.use_sft_fusion:
            return self.encode_lq_feats(lq), None
        if self.sft_uses_shared_lq:
            lq_feats, encoder_feats = self.lq_encoder.forward_with_pyramid(lq)
            return lq_feats, self.sft_condition(encoder_feats)
        return self.encode_lq_feats(lq), self.sft_condition(lq)

    def predict_logits(self, feats: torch.Tensor) -> torch.Tensor:
        return self.transformer(feats)

    def tokens_to_feats(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 2:
            # caller must reshape if h/w is ambiguous
            raise ValueError('indices with shape (B, HW) are ambiguous; pass (B,1,H,W).')
        b, _, h, w = indices.shape
        shape = [b, h, w, self.embed_dim]
        indices = indices.long().to(self.vqgan.quantize.embedding.weight.device)
        return self.vqgan.quantize.get_codebook_entry(indices, shape)

    def decode_tokens(self,
                      indices: torch.Tensor,
                      lq: Optional[torch.Tensor] = None,
                      cond_feats: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        decoded = self.decode_quant(
            self.tokens_to_feats(indices.long()),
            lq,
            cond_feats=cond_feats,
        )
        return decoded

    def decode_quant(self,
                     quant: torch.Tensor,
                     lq: Optional[torch.Tensor] = None,
                     cond_feats: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if self.use_sft_fusion and lq is not None:
            decoded = self._decode_quant_with_sft(quant, lq, cond_feats=cond_feats)
        else:
            decoded = self.vqgan.decode(quant)
        if self.refiner is not None and lq is not None:
            decoded = self.refiner(decoded, lq)
        return decoded

    def logits_to_quant(self,
                        logits: torch.Tensor,
                        h: int,
                        w: int,
                        tau: float = 1.0,
                        straight_through: bool = True,
                        hard_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Soft/ST codebook projection; optional IDs preserve deployed sampling.

        Stage 2 keeps its original argmax forward when ``hard_ids`` is omitted.
        Stage-3 reconstruction can supply the actual annealed token draw while
        using ``tau`` only for the backward softmax surrogate.
        """
        b, n, k = logits.shape
        if n != h * w:
            raise ValueError(f'logits length {n} does not match token grid {(h, w)}.')
        codebook = self.vqgan.quantize.embedding.weight
        if k != codebook.shape[0]:
            raise ValueError(f'logits codebook size {k} does not match VQ codebook size {codebook.shape[0]}.')

        tau = max(float(tau), 1e-6)
        probs = F.softmax(logits.float() / tau, dim=-1)
        codebook = codebook.to(device=probs.device, dtype=probs.dtype)
        soft_quant = torch.matmul(probs, codebook).transpose(1, 2).reshape(b, self.embed_dim, h, w)

        if not straight_through:
            return soft_quant

        sampled_ids = hard_ids is not None
        if hard_ids is None:
            hard_ids = logits.argmax(dim=-1).reshape(b, 1, h, w)
        else:
            if hard_ids.shape not in ((b, n), (b, 1, h, w)) or hard_ids.dtype != torch.long:
                raise ValueError('hard_ids must be Long token IDs of shape (B, N) or (B, 1, H, W).')
            hard_ids = hard_ids.detach().reshape(b, 1, h, w)
        hard_quant = self.tokens_to_feats(hard_ids)
        if sampled_ids:
            # Keep the original codebook precision in the hard forward even
            # when autocast computes the soft matrix product in BF16/FP16.
            return hard_quant.detach() + (soft_quant - soft_quant.detach()).to(hard_quant)
        hard_quant = hard_quant.to(device=soft_quant.device, dtype=soft_quant.dtype)
        return soft_quant + (hard_quant - soft_quant).detach()

    def decode_logits_soft(self,
                           logits: torch.Tensor,
                           lq: Optional[torch.Tensor],
                           h: int,
                           w: int,
                           tau: float = 1.0,
                           straight_through: bool = True,
                           cond_feats: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        quant = self.logits_to_quant(logits, h, w, tau=tau, straight_through=straight_through)
        return self.decode_quant(quant, lq, cond_feats=cond_feats)

    def forward(self,
                input: torch.Tensor,
                hq_feats: Optional[torch.Tensor] = None,
                token_mask: Optional[torch.Tensor] = None,
                code_only: Optional[bool] = None,
                soft_decode: bool = False,
                soft_decode_tau: float = 1.0,
                soft_decode_straight_through: bool = True,
                use_self_context: bool = False):
        if not self.LQ_stage:
            dec, emb_loss = self.vqgan(input)
            return dec, emb_loss, None, None

        if code_only is None:
            code_only = self.code_only

        cond_feats = None
        if not code_only and self.use_sft_fusion:
            lq_feats, cond_feats = self.encode_lq_context(input)
        else:
            lq_feats = self.encode_lq_feats(input)
        b, c, h, w = lq_feats.shape

        if hq_feats is None:
            if self.training and token_mask is not None:
                raise ValueError('hq_feats is required when token_mask is provided during predictor training.')
            mixed_feats = lq_feats
        else:
            if hq_feats.shape[0] != b or hq_feats.shape[1] != c or hq_feats.shape[-2:] != (h, w):
                raise ValueError(
                    f'LQ latent shape {tuple(lq_feats.shape)} does not match GT latent shape {tuple(hq_feats.shape)}. '
                    'Check scale, gt_size, and SwinVQGAN downsample factor.')
            if token_mask is None:
                token_mask = torch.ones(b, 1, h, w, dtype=torch.bool, device=input.device)
            token_mask = _as_bool_mask(token_mask).to(input.device)
            if token_mask.all().item():
                mixed_feats = lq_feats
            else:
                if use_self_context:
                    with torch.no_grad():
                        logits_first = self.predict_logits(lq_feats)
                        self_ids = logits_first.argmax(dim=-1).reshape(b, 1, h, w)
                        known_feats = self.tokens_to_feats(self_ids)
                else:
                    known_feats = hq_feats.detach()
                known_feats = known_feats.to(device=lq_feats.device, dtype=lq_feats.dtype)
                mixed_feats = torch.where(token_mask, lq_feats, known_feats)

        logits = self.predict_logits(mixed_feats)

        if code_only:
            return logits, lq_feats

        if soft_decode:
            out_img = self.decode_logits_soft(
                logits,
                input,
                h,
                w,
                tau=soft_decode_tau,
                straight_through=soft_decode_straight_through,
                cond_feats=cond_feats,
            )
        else:
            out_tokens = logits.argmax(dim=2).reshape(b, 1, h, w)
            out_img = self.decode_tokens(out_tokens, input, cond_feats=cond_feats)
        return logits, lq_feats, out_img

    @torch.no_grad()
    def test(self, input: torch.Tensor):
        self.eval()
        # Padding: LR dimensions must make the HR latent grid both integral and
        # divisible by the Swin window.
        wsz = max(1, (self.vq_downsample // self.scale_factor) * self.predictor_window_size)
        _, _, h_old, w_old = input.shape
        h_pad = (wsz - h_old % wsz) % wsz
        w_pad = (wsz - w_old % wsz) % wsz
        if h_pad > 0:
            target_h = h_old + h_pad
            while input.shape[2] < target_h:
                input = torch.cat([input, torch.flip(input, [2])], 2)
            input = input[:, :, :target_h, :]
        if w_pad > 0:
            target_w = w_old + w_pad
            while input.shape[3] < target_w:
                input = torch.cat([input, torch.flip(input, [3])], 3)
            input = input[:, :, :, :target_w]

        lq_feats, cond_feats = self.encode_lq_context(input)
        logits = self.predict_logits(lq_feats)
        b, _, h, w = lq_feats.shape
        tokens = logits.argmax(dim=2).reshape(b, 1, h, w)
        output = self.decode_tokens(tokens, input, cond_feats=cond_feats)
        output = output[..., :h_old * self.scale_factor, :w_old * self.scale_factor]
        return output

    @torch.no_grad()
    def test_tile(self, input: torch.Tensor, tile_size: int = 256, tile_pad: int = 16):
        batch, channel, height, width = input.shape
        output = input.new_zeros(batch, channel, height * self.scale_factor, width * self.scale_factor)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                output_tile = self.test(input_tile)
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + (input_end_x - input_start_x) * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + (input_end_y - input_start_y) * self.scale_factor
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        return output
