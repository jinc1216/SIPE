"""Evaluate paired SR/HR images and save per-image metrics and means to CSV.

Examples:
    python test_paired_metrics.py -p results -g datasets/HR --all_metrics False
    python test_paired_metrics.py -p results -g datasets/HR --all_metrics True --csv_path results/metrics.csv
    python test_paired_metrics.py -p results -g datasets/HR --metrics lpips dists niqe clipiqa
    python test_paired_metrics.py -p results -g datasets/HR --metrics lpips dists
    python test_paired_metrics.py -p results -g datasets/HR --metrics topiq_fr-pipal --csv_path results/topiq.csv

Image pairs are decoded on CPU and evaluated one at a time. Only scalar scores
are retained between pairs; TOPIQ attention maps and unused CUDA cache are
released before the next pair is loaded.
"""

import argparse
import csv
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BASIC_METRICS = ("psnr", "ssim", "lpips")
FULL_METRICS = BASIC_METRICS + ("topiq_fr-pipal", "dists", "niqe", "clipiqa")
REFERENCE_METRICS = {"psnr", "ssim", "lpips", "dists", "topiq_fr-pipal"}
DISPLAY_NAMES = {"clipiqa": "CLIP-IQA"}


def parse_bool(value):
    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected True or False.")


def list_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    images = sorted(
        item for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No image files found in: {path}")
    return images


def make_pairs(pred_path, gt_path):
    pred_path, gt_path = Path(pred_path), Path(gt_path)
    if pred_path.is_file() and gt_path.is_file():
        return [(pred_path, gt_path)]
    pred_images = list_images(pred_path)
    gt_by_name = {item.name: item for item in list_images(gt_path)}
    missing = [pred.name for pred in pred_images if pred.name not in gt_by_name]
    if missing:
        raise ValueError(f"Missing matching GT files: {', '.join(missing[:10])}")
    return [(pred, gt_by_name[pred.name]) for pred in pred_images]


def align_pair(pred, gt, max_size_diff, image_name):
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"Batch/channel mismatch for {image_name}: {pred.shape} vs {gt.shape}")
    pred_height, pred_width = pred.shape[-2:]
    gt_height, gt_width = gt.shape[-2:]
    if (pred_height, pred_width) == (gt_height, gt_width):
        return pred, gt
    if (abs(pred_height - gt_height) > max_size_diff
            or abs(pred_width - gt_width) > max_size_diff):
        raise ValueError(
            f"Image size mismatch is too large for {image_name}: "
            f"prediction={pred_width}x{pred_height}, GT={gt_width}x{gt_height}, "
            f"maximum allowed difference={max_size_diff}"
        )
    height, width = min(pred_height, gt_height), min(pred_width, gt_width)
    print(
        f"{image_name}: aligning prediction {pred_width}x{pred_height} and "
        f"GT {gt_width}x{gt_height} to {width}x{height} (cropping right/bottom)"
    )
    return pred[..., :height, :width], gt[..., :height, :width]


def build_metrics(create_metric, all_metrics, device, crop_border, test_y_channel, metric_names=None):
    names = metric_names if metric_names is not None else (FULL_METRICS if all_metrics else BASIC_METRICS)
    metrics = {}
    for name in dict.fromkeys(names):
        options = {"crop_border": crop_border, "test_y_channel": test_y_channel} \
            if name in ("psnr", "ssim") else {}
        metrics[name] = create_metric(name, device=device, **options).eval()
    return metrics


def evaluate_pair(metrics, prediction, aligned_prediction, gt):
    """Use aligned pairs for FR metrics and the original prediction for NR metrics."""
    scores = {}
    for name, metric in metrics.items():
        value = metric(aligned_prediction, gt) if name in REFERENCE_METRICS else metric(prediction)
        scores[name] = float(value.item())
    return scores


def evaluate_image_pair(metrics, pred_path, gt_path, read_tensor, device,
                        max_size_diff=4, crop_border=0):
    """Keep image tensors local so they are released before loading the next pair."""
    prediction = read_tensor(str(pred_path), rgb=True).unsqueeze(0)
    reference = read_tensor(str(gt_path), rgb=True).unsqueeze(0)
    aligned_prediction, reference = align_pair(
        prediction, reference, max_size_diff, pred_path.name,
    )
    height, width = aligned_prediction.shape[-2:]
    if (any(name in metrics for name in ("psnr", "ssim"))
            and (height <= 2 * crop_border or width <= 2 * crop_border)):
        raise ValueError(
            f"--crop_border {crop_border} leaves no valid pixels for "
            f"{pred_path.name} ({width}x{height})"
        )

    # Align on CPU first. FR-only runs upload just one SR tensor and one HR
    # tensor, rather than retaining both the original and cropped SR on CUDA.
    if all(name in REFERENCE_METRICS for name in metrics):
        aligned_prediction = aligned_prediction.to(device)
        prediction = aligned_prediction
    elif aligned_prediction.shape == prediction.shape:
        prediction = prediction.to(device)
        aligned_prediction = prediction
    else:
        prediction = prediction.to(device)
        aligned_prediction = aligned_prediction.to(device)
    reference = reference.to(device)
    return evaluate_pair(metrics, prediction, aligned_prediction, reference)


def release_metric_buffers(metrics):
    """Drop TOPIQ's diagnostic attention maps, which otherwise retain CUDA tensors."""
    topiq = metrics.get("topiq_fr-pipal")
    if topiq is not None:
        for module in topiq.modules():
            if hasattr(module, "attn_map"):
                module.attn_map = None


def format_scores(scores):
    return ", ".join(
        f"{DISPLAY_NAMES.get(name, name.upper())}={value:.6f}"
        for name, value in scores.items()
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--pred", required=True, help="Predicted SR image path or folder.")
    parser.add_argument("-g", "--gt", required=True, help="Paired HR image path or folder (required in both modes).")
    parser.add_argument(
        "--all_metrics", type=parse_bool, default=False, metavar="True|False",
        help="True: PSNR, SSIM, LPIPS, TOPIQ-FR-PIPAL, DISTS, NIQE, CLIP-IQA; "
             "False: PSNR, SSIM, LPIPS (default).",
    )
    parser.add_argument(
        "--metrics", nargs="+", type=str.lower, choices=FULL_METRICS,
        help="Compute only the specified metrics (space-separated, case-insensitive). "
             "Overrides --all_metrics; duplicate names are computed once.",
    )
    parser.add_argument("--crop_border", type=int, default=0, help="Border crop for PSNR/SSIM only.")
    parser.add_argument("--test_y_channel", action="store_true", help="Use Y channel for PSNR/SSIM only.")
    parser.add_argument(
        "--max_size_diff", type=int, default=4,
        help="Maximum per-dimension mismatch to align by cropping right/bottom (default: 4).",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu. Default: auto.")
    parser.add_argument(
        "--csv_path", type=Path, default=Path("metrics_results.csv"),
        help="Output CSV path (default: metrics_results.csv). Existing CSV is overwritten.",
    )
    args = parser.parse_args(argv)
    if args.crop_border < 0:
        parser.error("--crop_border must be non-negative")
    if args.max_size_diff < 0:
        parser.error("--max_size_diff must be non-negative")

    pairs = make_pairs(args.pred, args.gt)
    if args.csv_path.resolve() in {path.resolve() for pair in pairs for path in pair}:
        parser.error("--csv_path must not overwrite an input SR/HR image")
    try:
        import torch
        from pyiqa import create_metric
        from pyiqa.utils.img_util import imread2tensor
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc.name}. Please install torch and pyiqa first.") from exc

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda_device = torch.device(device) if torch.device(device).type == "cuda" else None
    metrics = build_metrics(
        create_metric, args.all_metrics, device, args.crop_border, args.test_y_channel,
        metric_names=args.metrics,
    )
    print(f"Device: {device}; metrics={', '.join(metrics)}; paired images={len(pairs)}")
    totals = {name: 0.0 for name in metrics}
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM preserves Chinese filenames when opening the CSV in Excel.
    with args.csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file, torch.inference_mode():
        writer = csv.writer(csv_file)
        writer.writerow([
            "image", "gt_image",
            *(DISPLAY_NAMES.get(name, name.upper()) for name in metrics),
        ])
        for index, (pred_path, gt_path) in enumerate(pairs, 1):
            try:
                scores = evaluate_image_pair(
                    metrics, pred_path, gt_path, imread2tensor, device,
                    max_size_diff=args.max_size_diff, crop_border=args.crop_border,
                )
            finally:
                release_metric_buffers(metrics)
                if cuda_device is not None:
                    # Tensor references have gone out of scope; release cached
                    # allocations on the selected GPU before decoding another pair.
                    with torch.cuda.device(cuda_device):
                        torch.cuda.empty_cache()
            for name, value in scores.items():
                totals[name] += value
            writer.writerow([pred_path.name, gt_path.name, *(scores[name] for name in metrics)])
            csv_file.flush()
            print(f"[{index}/{len(pairs)}] {pred_path.name}: {format_scores(scores)}")

        averages = {name: total / len(pairs) for name, total in totals.items()}
        writer.writerow(["Average", "", *(averages[name] for name in metrics)])

    print(f"Average over {len(pairs)} images: {format_scores(averages)}")
    print(f"Per-image metrics and averages saved to: {args.csv_path.resolve()}")


if __name__ == "__main__":
    main()
