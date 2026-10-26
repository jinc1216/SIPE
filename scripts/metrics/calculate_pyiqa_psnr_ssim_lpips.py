import argparse
from pathlib import Path

import cv2
import torch

from basicsr.utils import img2tensor

try:
    import pyiqa
except ImportError as exc:
    raise ImportError('This script requires pyiqa. Install it with `pip install pyiqa`.') from exc


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def list_images(folder: str, recursive: bool = False):
    root = Path(folder)
    pattern = '**/*' if recursive else '*'
    return sorted(p for p in root.glob(pattern) if p.suffix.lower() in IMG_EXTS)


def read_img_tensor(path: Path, device: torch.device):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    tensor = img2tensor(img.astype('float32') / 255.0, bgr2rgb=True, float32=True)
    return tensor.unsqueeze(0).to(device)


def build_pairs(gt_dir: str, sr_dir: str, suffix: str = '', recursive: bool = False):
    gt_paths = list_images(gt_dir, recursive=recursive)
    sr_root = Path(sr_dir)
    pairs = []
    for gt_path in gt_paths:
        rel = gt_path.relative_to(gt_dir)
        sr_name = f'{gt_path.stem}{suffix}{gt_path.suffix}'
        sr_path = sr_root / rel.parent / sr_name
        if not sr_path.is_file():
            raise FileNotFoundError(f'Missing SR image for {gt_path}: expected {sr_path}')
        pairs.append((gt_path, sr_path))
    if not pairs:
        raise FileNotFoundError(f'No images found in {gt_dir}')
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Calculate PSNR/SSIM/LPIPS with pyiqa for paired GT and SR folders.')
    parser.add_argument('--gt', '--hq', dest='gt', required=True, help='Folder of HQ/GT images.')
    parser.add_argument('--sr', '--restored', dest='sr', required=True, help='Folder of super-resolved images.')
    parser.add_argument('--suffix', default='', help='Optional suffix in SR filenames, e.g. _x2.')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for PSNR/SSIM.')
    parser.add_argument('--test_y_channel', action='store_true', help='Calculate PSNR/SSIM on Y channel.')
    parser.add_argument('--recursive', action='store_true', help='Recursively match images by relative path.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    metrics = {
        'psnr': pyiqa.create_metric(
            'psnr', device=device, crop_border=args.crop_border, test_y_channel=args.test_y_channel),
        'ssim': pyiqa.create_metric(
            'ssim', device=device, crop_border=args.crop_border, test_y_channel=args.test_y_channel),
        'lpips': pyiqa.create_metric('lpips', device=device),
    }
    totals = {name: 0.0 for name in metrics}
    pairs = build_pairs(args.gt, args.sr, suffix=args.suffix, recursive=args.recursive)

    for idx, (gt_path, sr_path) in enumerate(pairs, 1):
        gt = read_img_tensor(gt_path, device)
        sr = read_img_tensor(sr_path, device)
        if gt.shape != sr.shape:
            raise ValueError(f'Shape mismatch: {gt_path} {tuple(gt.shape)} vs {sr_path} {tuple(sr.shape)}')

        values = {}
        with torch.no_grad():
            for name, metric in metrics.items():
                values[name] = float(metric(sr, gt).item())
                totals[name] += values[name]
        print(
            f'{idx:4d}: {gt_path.name:30s} '
            f'PSNR: {values["psnr"]:.6f}  SSIM: {values["ssim"]:.6f}  LPIPS: {values["lpips"]:.6f}')

    count = len(pairs)
    print(f'\nGT: {args.gt}')
    print(f'SR: {args.sr}')
    print(
        f'Average over {count} images: '
        f'PSNR: {totals["psnr"] / count:.6f}  '
        f'SSIM: {totals["ssim"] / count:.6f}  '
        f'LPIPS: {totals["lpips"] / count:.6f}')


if __name__ == '__main__':
    main()
