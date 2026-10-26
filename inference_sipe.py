import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import torch
import yaml
from tqdm import tqdm

from basicsr.archs.sipe_arch import SIPETokenNet, SIPEEvaluator
from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.utils.options import ordered_yaml
from basicsr.utils.parallel_decode import decode_evaluator, resolve_num_refine_steps, resolve_token_update_mode


def load_yaml(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    if not isinstance(opt, dict):
        raise ValueError('--opt must contain a YAML configuration mapping.')
    return opt


def network_options(opt: Dict, key: str, expected_type: str) -> Dict:
    config = opt.get(key)
    if not isinstance(config, dict) or not config:
        raise ValueError(f'--opt must define {key}.')
    config = deepcopy(config)
    network_type = config.pop('type', expected_type)
    if network_type != expected_type:
        raise ValueError(f'{key}.type must be {expected_type}, got {network_type}.')
    return config


def get_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ('params_ema', 'params', 'state_dict'):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise TypeError(f'Checkpoint {ckpt_path} does not contain a state dict.')
    return {k.replace('module.', '', 1): v for k, v in ckpt.items()}


def load_state(module: torch.nn.Module, ckpt_path: str, strict: bool = True, name: str = 'network'):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'{name} checkpoint not found: {ckpt_path}')
    state = get_state_dict(ckpt_path)
    if not strict:
        current = module.state_dict()
        mismatched = [
            key for key, value in state.items()
            if key in current and tuple(value.shape) != tuple(current[key].shape)
        ]
        for key in mismatched:
            state.pop(key)
        if mismatched:
            print(
                f'[{name}] ignored {len(mismatched)} shape-mismatched tensors: '
                + ', '.join(mismatched)
            )
    missing, unexpected = module.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f'[{name}] loaded with strict={strict}; missing={len(missing)}, unexpected={len(unexpected)}')


def build_networks(args, opt: Dict):
    """Build only requested networks, without overriding YAML architecture options."""
    net_g_opt = network_options(opt, 'network_g', 'SIPETokenNet')
    if 'scale_factor' not in net_g_opt:
        raise ValueError('--opt must define network_g.scale_factor.')
    if opt.get('scale') is not None and opt['scale'] != net_g_opt['scale_factor']:
        raise ValueError('YAML scale and network_g.scale_factor must agree.')
    if not net_g_opt.get('LQ_stage', True):
        raise ValueError('SIPE-SR inference requires network_g.LQ_stage=true.')
    evaluator_opt = (network_options(opt, 'network_evaluator', 'SIPEEvaluator')
                     if args.evaluator_path else None)
    net_g = SIPETokenNet(**net_g_opt)
    if evaluator_opt is not None and evaluator_opt.get('use_score_feats', False):
        raise ValueError('The X2 evaluator requires use_score_feats=false.')
    net_evaluator = SIPEEvaluator(**evaluator_opt) if evaluator_opt is not None else None
    return net_g, net_evaluator


def list_images(input_path: str):
    root = Path(input_path)
    if root.is_file():
        return [str(root)]
    if not root.is_dir():
        raise FileNotFoundError(f'Image path does not exist: {root}')
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    return sorted(str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def read_image_tensor(path: str, device: torch.device):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Failed to read image: {path}')
    return img2tensor(image.astype('float32') / 255.0, bgr2rgb=True).unsqueeze(0).to(device)


def resize_if_needed(img: torch.Tensor, max_size: int):
    if max_size <= 0:
        return img, None
    _, _, h, w = img.shape
    if max(h, w) <= max_size:
        return img, None
    scale = max_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return torch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False), (h, w)


def flip_pad_to_multiple(x: torch.Tensor, multiple: int):
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


@torch.no_grad()
def infer_with_evaluator(net_g: SIPETokenNet,
                         net_evaluator: SIPEEvaluator,
                         img: torch.Tensor,
                         num_iter: Optional[int] = None,
                         val_opt: Optional[Dict] = None):
    """Use the same paired-gain controller and sampling as Stage-3 validation.

    No GT, semantic teacher, decoder feedback or image metric enters the loop.
    num_iter overrides only the number of candidate proposal actions.
    """
    val_opt = val_opt or {}
    count_opt = val_opt if num_iter is None else {'num_refine_steps': num_iter}
    num_iter = resolve_num_refine_steps(count_opt)
    # LR size must make the HR latent grid divisible by Swin window size.
    window_multiple = max(1, (net_g.vq_downsample // net_g.scale_factor) * net_g.predictor_window_size)
    img_pad, h_old, w_old = flip_pad_to_multiple(img, window_multiple)

    lq_feats, sft_cond_feats = net_g.encode_lq_context(img_pad)
    b, _, h, w = lq_feats.shape
    mask_tokens = torch.full((b, h * w), -1, dtype=torch.long, device=img.device)

    def tokens_to_logits(x: torch.Tensor,
                         h_: int = 0,
                         w_: int = 0,
                         evaluator: bool = False,
                         cond_feats: torch.Tensor = None,
                         proposal_ids: torch.Tensor = None) -> torch.Tensor:
        if evaluator:
            return net_evaluator(x, h_, w_, cond_feats=cond_feats if net_evaluator.use_cond_feats else None,
                                 proposal_ids=proposal_ids)
        return net_g.predict_logits(x)

    def tokens_to_feats(seq: torch.Tensor) -> torch.Tensor:
        return net_g.tokens_to_feats(seq.long())

    output_tokens, _ = decode_evaluator(
        mask_tokens=mask_tokens,
        lq_feats=lq_feats,
        tokens_to_logits=tokens_to_logits,
        tokens_to_feats=tokens_to_feats,
        num_refine_steps=num_iter,
        token_selection=val_opt.get('decode_sampling', val_opt.get('token_selection', 'multinomial')),
        remask_budget_schedule=val_opt.get('remask_budget_schedule', 'full_cosine'),
        remask_budget_scale=val_opt.get('remask_budget_scale', 1.0),
        token_update_mode=resolve_token_update_mode(val_opt),
        sample_temperature=float(val_opt.get('sample_temperature', 1.0)),
    )
    final_tokens = output_tokens[:, -1, :].reshape(b, 1, h, w)
    output = net_g.decode_tokens(final_tokens, img_pad, cond_feats=sft_cond_feats)
    return output[..., :h_old * net_g.scale_factor, :w_old * net_g.scale_factor]


@torch.no_grad()
def infer_tiled(infer_fn: Callable[[torch.Tensor], torch.Tensor],
                img: torch.Tensor,
                scale_factor: int,
                tile_size: int,
                tile_pad: int) -> torch.Tensor:
    if tile_size <= 0:
        return infer_fn(img)
    if tile_pad < 0:
        raise ValueError(f'tile_pad must be non-negative, got {tile_pad}.')

    b, c, h, w = img.shape
    if b != 1:
        raise ValueError('Tiled inference only supports batch size 1.')

    output = img.new_zeros(b, c, h * scale_factor, w * scale_factor)
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            py0 = max(y0 - tile_pad, 0)
            px0 = max(x0 - tile_pad, 0)
            py1 = min(y1 + tile_pad, h)
            px1 = min(x1 + tile_pad, w)

            tile = img[:, :, py0:py1, px0:px1]
            tile_out = infer_fn(tile)

            crop_y0 = (y0 - py0) * scale_factor
            crop_x0 = (x0 - px0) * scale_factor
            crop_y1 = crop_y0 + (y1 - y0) * scale_factor
            crop_x1 = crop_x0 + (x1 - x0) * scale_factor
            out_y0 = y0 * scale_factor
            out_x0 = x0 * scale_factor
            out_y1 = y1 * scale_factor
            out_x1 = x1 * scale_factor
            output[:, :, out_y0:out_y1, out_x0:out_x1] = \
                tile_out[:, :, crop_y0:crop_y1, crop_x0:crop_x1]
    return output


@torch.no_grad()
def infer_direct(net_g: SIPETokenNet, img: torch.Tensor):
    return net_g.test(img)


def main():
    parser = argparse.ArgumentParser(
        description='SIPE-SR CUDA/BF16 inference for x2/x4 image super-resolution; saves images only.')
    parser.add_argument('--opt', type=str, required=True, help='Stage-2/3 YAML; the sole source of model architecture parameters.')
    parser.add_argument('--predictor_path', type=str, required=True, help='Path to the trained SIPETokenNet checkpoint, including its VQ prior.')
    parser.add_argument('--evaluator_path', type=str, default=None, help='Path to trained SIPEEvaluator checkpoint. If omitted, direct argmax decoding is used.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input image or folder of LR images.')
    parser.add_argument('-o', '--output', type=str, default='results_sipe', help='Output folder.')
    parser.add_argument('-n', '--num_iter', type=int, default=None,
                        help='Argmax evaluator->predictor refinements AFTER argmax P0 (0 means P0 only); '
                             'defaults to YAML val.num_refine_steps.')
    parser.add_argument('--tile', type=int, default=0, help='LR tile size for tiled inference; <=0 disables tiling.')
    parser.add_argument('--tile_pad', type=int, default=32, help='LR overlap padding for tiled inference.')
    parser.add_argument(
        '--max_size',
        type=int,
        default=0,
        help='Resize very large LR images before inference; <=0 disables this. Use tiling for benchmarks.')
    parser.add_argument('--seed', type=int, default=None, help='Sampling seed; defaults to YAML manual_seed or 0.')
    parser.add_argument('--strict_load', action='store_true',
                        help='Force strict predictor loading; otherwise follow YAML path.strict_load_g/strict_load (default true). Evaluator mode is always strict.')
    args = parser.parse_args()

    opt = load_yaml(args.opt)
    val_opt = opt.get('val', {}) or {}
    if args.num_iter is not None and args.num_iter < 0:
        parser.error('--num_iter must be non-negative.')
    if args.tile_pad < 0:
        parser.error('--tile_pad must be non-negative.')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for inference; no CUDA device is available.')
    if not torch.cuda.is_bf16_supported():
        parser.error('The selected CUDA device does not support BF16 inference.')
    device = torch.device('cuda')
    print(f'Inference device: {device} ({torch.cuda.get_device_name(device)}); precision: BF16 autocast.')

    paths = list_images(args.input)
    if not paths:
        raise FileNotFoundError(f'No images found in {args.input}')
    if len({Path(path).stem for path in paths}) != len(paths):
        raise ValueError('LR filenames must have unique stems to avoid overwriting output images.')
    protected_paths = {Path(path).resolve() for path in paths}
    for path in paths:
        destination = Path(args.output) / f'{Path(path).stem}.png'
        if destination.resolve() in protected_paths:
            raise ValueError(f'Output would overwrite an LR input: {destination}')

    if args.evaluator_path:
        count_opt = val_opt if args.num_iter is None else {'num_refine_steps': args.num_iter}
        args.num_iter = resolve_num_refine_steps(count_opt)
        print(f'Evaluator decoding: P0 + up to {args.num_iter} proposal/accept steps; '
              f'sampling={val_opt.get("decode_sampling", "multinomial")}; '
              f'at most {args.num_iter + 1} predictor and {2 * args.num_iter} evaluator calls; '
              'accept positive predicted token-correctness gains only.')
    else:
        print('Predictor-only decoding: no evaluator is built or loaded.')

    seed = int(args.seed if args.seed is not None else opt.get('manual_seed', 0))
    torch.manual_seed(seed)
    net_g, net_evaluator = build_networks(args, opt)
    path_opt = opt.get('path') or {}
    strict = args.strict_load or bool(args.evaluator_path) or bool(
        path_opt.get('strict_load_g', path_opt.get('strict_load', True)))
    load_state(net_g, args.predictor_path, strict=strict, name='predictor')
    if net_evaluator is not None:
        load_state(net_evaluator, args.evaluator_path, strict=True, name='evaluator')
        # Preserve learned independent embeddings; only legacy frozen_vq
        # checkpoints require equality with the predictor codebook.
        net_evaluator.prepare_token_embedding(
            net_g.vqgan.quantize.embedding.weight, checkpoint_loaded=True)
        net_evaluator = net_evaluator.to(device).eval()

    net_g = net_g.to(device).eval()
    scale = net_g.scale_factor
    # Keep multinomial sampling independent of network initialization.
    torch.manual_seed(seed)
    os.makedirs(args.output, exist_ok=True)
    if net_evaluator is not None:
        infer_fn = lambda tile: infer_with_evaluator(
            net_g, net_evaluator, tile, val_opt=val_opt,
            num_iter=args.num_iter)
    else:
        infer_fn = lambda tile: infer_direct(net_g, tile)

    for path in tqdm(paths, unit='image'):
        img_name = Path(path).stem
        tensor = read_image_tensor(path, device)
        tensor, original_size = resize_if_needed(tensor, args.max_size)
        # Cover direct, evaluator-guided and tiled inference with the same AMP policy.
        # Keep model weights and postprocessing in FP32.
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = infer_tiled(infer_fn, tensor, scale, args.tile, args.tile_pad)
        output = output.float()

        if original_size is not None:
            h0, w0 = original_size
            output = torch.nn.functional.interpolate(
                output, size=(h0 * scale, w0 * scale), mode='bilinear', align_corners=False)
        out_img = tensor2img(output, rgb2bgr=True)
        imwrite(out_img, os.path.join(args.output, f'{img_name}.png'))

    print(f'Saved {len(paths)} inferred images to {args.output}')


if __name__ == '__main__':
    main()
