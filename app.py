"""Gradio demo for SIPE-SR checkpoints, example images, and user uploads."""

import argparse
import gc
import math
import threading
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
NO_EVALUATOR = ('None (Predictor only, faster)', '')
APP_FONT = ['Times New Roman', 'Times', 'serif']


def read_input_image(path):
    """Use the same orientation and RGB conversion for preview and inference."""
    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert('RGB')
    except (OSError, ValueError) as error:
        raise gr.Error(f'Failed to read image {Path(path).name}: {error}') from error


def integer(value, name, minimum, maximum):
    """Validate UI/API numbers before loading any model."""
    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer() or not minimum <= number <= maximum:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise gr.Error(f'{name} must be an integer between {minimum} and {maximum}.') from None
    return int(number)


class SIPEApplication:
    def __init__(self, images_dir=ROOT / 'images', experiments_dir=ROOT / 'experiments',
                 output_dir=None, device='auto'):
        self.images_dir = Path(images_dir).resolve()
        self.experiments_dir = Path(experiments_dir).resolve()
        # Keep downloads available during the app session without saving results in the project.
        self._temporary_output = TemporaryDirectory(prefix='sipe_gradio_') if output_dir is None else None
        self.output_dir = Path(self._temporary_output.name if self._temporary_output else output_dir).resolve()
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested, but no CUDA device is available. Use --device cpu instead.')
        self.device = torch.device(device)
        self.use_bf16 = self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        self.lock = threading.Lock()
        self.cache_key = None
        self.net_g = self.net_evaluator = self.opt = None

    def images(self):
        return {
            path.relative_to(self.images_dir).as_posix(): path
            for path in sorted(self.images_dir.rglob('*'))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            and not any(part.startswith('.') for part in path.relative_to(self.images_dir).parts)
            and path.resolve().is_relative_to(self.images_dir)
        }

    def checkpoints(self, scale, evaluator=False):
        stage = f'03_SIPE_evaluator_X{scale}' if evaluator else f'02_SIPE_predictor_X{scale}'
        prefix = 'net_evaluator' if evaluator else 'net_g'
        paths = (self.experiments_dir / stage / 'models').glob(f'{prefix}*.pth')
        return [
            path.relative_to(self.experiments_dir).as_posix()
            for path in sorted(paths, key=lambda path: (path.name != f'{prefix}_latest.pth', path.name))
            if path.is_file() and path.resolve().is_relative_to(self.experiments_dir)
        ]

    def config_path(self, scale, evaluator=False):
        stage = '03' if evaluator else '02'
        network = 'evaluator' if evaluator else 'predictor'
        filename = f'{stage}train_SIPE_{network}_X{scale}.yml'
        archived = self.experiments_dir / f'{stage}_SIPE_{network}_X{scale}' / filename
        return archived if archived.is_file() else ROOT / 'options' / filename

    def scale_controls(self, scale):
        predictors = self.checkpoints(scale)
        evaluators = self.checkpoints(scale, evaluator=True)
        return (
            gr.Dropdown(choices=predictors, value=next(iter(predictors), None)),
            gr.Dropdown(choices=[NO_EVALUATOR] + evaluators, value=next(iter(evaluators), '')),
            gr.Slider(value=4),
        )

    def preview(self, image_name):
        return self.preview_file(self.images().get(image_name))

    def preview_file(self, image_path):
        if not image_path:
            return None, 'Select an example image or upload your own image.'
        path = Path(image_path)
        image = read_input_image(path)
        return str(path), f'Input: {path.name} · {image.width} × {image.height} pixels'

    def environment_info(self):
        if self.device.type == 'cuda':
            precision = 'BF16' if self.use_bf16 else 'FP32'
            return f'Device: {torch.cuda.get_device_name(self.device)} · {precision}'
        return 'Device: CPU · FP32. Large images may take longer to process. CUDA is recommended.'

    def _load_networks(self, scale, predictor, evaluator):
        # Reuse the CLI inference implementation; training-only imports stay lazy.
        from inference_sipe import build_networks, load_state, load_yaml

        predictor_path = self.experiments_dir / predictor
        evaluator_path = self.experiments_dir / evaluator if evaluator else None
        configs = [self.config_path(scale)]
        if evaluator:
            configs.append(self.config_path(scale, evaluator=True))
        files = configs + [predictor_path] + ([evaluator_path] if evaluator_path else [])
        key = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
        if key == self.cache_key:
            return

        self.net_g = self.net_evaluator = self.opt = self.cache_key = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        opt = load_yaml(str(configs[0]))
        if evaluator:
            evaluator_opt = load_yaml(str(configs[1]))
            opt['network_evaluator'] = evaluator_opt['network_evaluator']
            opt['val'] = evaluator_opt.get('val') or {}
        if opt['network_g']['scale_factor'] != scale:
            raise ValueError('The scale factor in the configuration does not match the selected scale.')
        # A complete Stage-2 checkpoint contains the VQ prior. Never fetch a
        # training-machine checkpoint referenced by an archived YAML.
        opt['network_g']['vqgan_opt']['ckpt_path'] = None
        net_g, net_evaluator = build_networks(SimpleNamespace(evaluator_path=evaluator_path), opt)
        load_state(net_g, str(predictor_path), strict=True, name='predictor')
        if net_evaluator is not None:
            load_state(net_evaluator, str(evaluator_path), strict=True, name='evaluator')
            net_evaluator.prepare_token_embedding(net_g.vqgan.quantize.embedding.weight, checkpoint_loaded=True)
            net_evaluator = net_evaluator.to(self.device).eval().requires_grad_(False)
        self.net_g = net_g.to(self.device).eval().requires_grad_(False)
        self.net_evaluator, self.opt, self.cache_key = net_evaluator, opt, key

    def run(self, image_path, scale, predictor, evaluator, steps, tile, tile_pad, seed,
            progress=gr.Progress()):
        scale = integer(scale, 'Scale factor', 2, 4)
        if scale not in (2, 4):
            raise gr.Error('SIPE supports only x2 and x4 super-resolution.')
        steps = integer(steps, 'Refinement steps', 0, 16)
        tile = integer(tile, 'Tile size', 0, 1024)
        tile_pad = integer(tile_pad, 'Tile padding', 0, 128)
        seed = integer(seed, 'Random seed', 0, 2**32 - 1)
        if not image_path:
            raise gr.Error('Select an example image or upload your own image before running BSR.')
        path = Path(image_path)
        if not path.is_file():
            raise gr.Error('The input image is no longer available. Select or upload it again.')
        if predictor not in self.checkpoints(scale):
            raise gr.Error(f'Please select a Predictor checkpoint for x{scale} from experiments.')
        if evaluator and evaluator not in self.checkpoints(scale, evaluator=True):
            raise gr.Error(f'Please select an Evaluator checkpoint for x{scale}, or use Predictor only.')

        with self.lock:
            started = time.perf_counter()
            try:
                source_image = read_input_image(path)
                progress(0.02, desc='Loading SIPE models (the first run may take longer)')
                self._load_networks(scale, predictor, evaluator)
                from inference_sipe import infer_direct, infer_tiled, infer_with_evaluator
                from basicsr.utils import img2tensor, tensor2img

                torch.manual_seed(seed)
                image = img2tensor(np.asarray(source_image, dtype=np.float32) / 255.0,
                                   bgr2rgb=False).unsqueeze(0).to(self.device)
                height, width = image.shape[-2:]
                count = math.ceil(height / tile) * math.ceil(width / tile) if tile else 1
                completed = 0

                def infer_one(part):
                    nonlocal completed
                    if self.net_evaluator is None:
                        result = infer_direct(self.net_g, part)
                    else:
                        result = infer_with_evaluator(
                            self.net_g, self.net_evaluator, part, num_iter=steps,
                            val_opt=self.opt.get('val') or {})
                    completed += 1
                    progress(0.20 + 0.70 * completed / count, desc=f'Processing tiles: {completed}/{count}')
                    return result

                inference_started = time.perf_counter()
                progress(0.20, desc='Starting super-resolution inference')
                with torch.inference_mode(), torch.autocast(
                        device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_bf16):
                    output = infer_tiled(infer_one, image, scale, tile, tile_pad)
                restored = Image.fromarray(tensor2img(output.float(), rgb2bgr=False))
                inference_seconds = time.perf_counter() - inference_started
                del image, output

                progress(0.95, desc='Preparing the PNG download')
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Unique filenames keep downloads from repeated runs separate.
                destination = self.output_dir / f'{path.stem}_SIPE_x{scale}_{uuid4().hex[:12]}.png'
                if destination.resolve().is_relative_to(self.images_dir):
                    raise ValueError('The output directory must be outside the input image directory. Set --output-dir.')
                restored.save(destination)
                baseline = source_image.resize(restored.size, Image.Resampling.BICUBIC)
                mode = f'Predictor + Evaluator, {steps} refinement steps' if evaluator else 'Predictor only'
                details = (
                    f'Complete · x{scale} · {width} × {height} → {restored.width} × {restored.height} pixels\n\n'
                    f'{mode} · {self.device.type.upper()} · '
                    f'Inference: {inference_seconds:.2f} s · Total: {time.perf_counter() - started:.2f} s\n\n'
                    f'Random seed: {seed} · PNG: {destination.name}'
                )
                progress(1, desc='Complete')
                return str(destination), (baseline, restored), str(destination), details
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise gr.Error('GPU memory is insufficient. Try a smaller tile size, such as 64 or 128.') from error
            except Exception as error:
                traceback.print_exc()
                raise gr.Error(f'SIPE inference failed: {error}') from error


def create_app(application=None):
    application = application or SIPEApplication()
    images = application.images()
    first_image = next(iter(images), None)
    initial_image, initial_info = application.preview(first_image)
    default_scale = 2 if application.checkpoints(2) else 4
    predictor_control, evaluator_control, steps_control = application.scale_controls(default_scale)

    with gr.Blocks(title='SIPE Blind Image Super-Resolution', analytics_enabled=False) as demo:
        gr.Markdown('# SIPE Image Super-Resolution\nSelect an example or upload your own image for x2 or x4 blind super-resolution.')
        gr.Markdown(application.environment_info())
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                image_choice = gr.Dropdown(
                    choices=list(images), value=first_image, label='Select an example image',
                    info='Choose from the images folder or thumbnails, or upload your own image below.')
                preview = gr.Image(value=initial_image, label='Input image', interactive=True,
                                   type='filepath', sources=['upload'], format='png', height=220)
                input_info = gr.Markdown(initial_info)
                scale = gr.Radio(choices=[('x2', 2), ('x4', 4)], value=default_scale, label='Scale factor')
                predictor = gr.Dropdown(choices=predictor_control.choices, value=predictor_control.value,
                                        label='Predictor checkpoint')
                evaluator = gr.Dropdown(choices=evaluator_control.choices, value=evaluator_control.value,
                                        label='Evaluator checkpoint',
                                        info='The Evaluator iteratively refines predictions. Predictor only is faster.')
                with gr.Accordion('Inference settings', open=False):
                    steps = gr.Slider(0, 16, value=steps_control.value, step=1, label='Evaluator refinement steps',
                                      info='Ignored in Predictor-only mode. Set to 0 to generate only the initial prediction.')
                    tile = gr.Slider(0, 512, value=128, step=16, label='Tile size (input pixels)',
                                     info='Smaller tiles use less memory. Set to 0 for full-image inference.')
                    tile_pad = gr.Slider(0, 128, value=32, step=4, label='Tile padding (input pixels)')
                    seed = gr.Number(value=0, precision=0, minimum=0, maximum=2**32 - 1, label='Random seed')
                run = gr.Button('Run Super-Resolution', variant='primary', interactive=bool(images))
                refresh = gr.Button('Refresh images and checkpoints')
            with gr.Column(scale=2, min_width=400):
                with gr.Tabs():
                    with gr.Tab('Super-resolution result'):
                        result = gr.Image(label='SIPE super-resolution result', interactive=False, format='png', height=560)
                    with gr.Tab('Before / after comparison'):
                        comparison = gr.ImageSlider(
                            label='Left: Bicubic upscaling / Right: SIPE super-resolution', interactive=False, format='png', height=560)
                download = gr.File(label='Download the super-resolution PNG', interactive=False)
                status = gr.Markdown('Select an example or upload an image, then click Run Super-Resolution. Models are loaded on the first run and reused afterward.')

        gallery_names = list(images)
        gallery = gr.Gallery(value=[(str(path), name) for name, path in images.items()],
                             label='Example images', columns=6, height=180, object_fit='contain',
                             interactive=False, allow_preview=False, format='png')

        def select_gallery(evt: gr.SelectData):
            index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
            if isinstance(index, int) and 0 <= index < len(gallery_names):
                name = gallery_names[index]
                return name, *change_image(name)
            return (gr.skip(),) * 8

        def change_image(name):
            image, info = application.preview(name)
            return (image, info, None, None, None,
                    'Image updated. Click Run Super-Resolution to generate a result.',
                    gr.Button(interactive=image is not None))

        def change_upload(image_path):
            image, info = application.preview_file(image_path)
            message = ('Image updated. Click Run Super-Resolution to generate a result.'
                       if image else 'Select an example image or upload your own image.')
            return None, info, None, None, None, message, gr.Button(interactive=image is not None)

        def change_scale(value):
            return (*application.scale_controls(value), None, None, None, 'Scale factor updated. Click Run Super-Resolution to generate a result.')

        def refresh_files(name, value, image_path):
            current = application.images()
            gallery_names[:] = list(current)
            name = name if name in current else None
            return (
                gr.Dropdown(choices=list(current), value=name),
                [(str(path), item) for item, path in current.items()],
                *application.scale_controls(value), gr.Button(interactive=bool(image_path)),
            )

        image_outputs = [preview, input_info, result, comparison, download, status, run]
        gallery.select(select_gallery, outputs=[image_choice, *image_outputs], queue=False, api_visibility='private')
        image_choice.input(change_image, inputs=image_choice, outputs=image_outputs,
                           queue=False, api_name='preview')
        preview.input(change_upload, inputs=preview,
                      outputs=[image_choice, input_info, result, comparison, download, status, run],
                      queue=False, api_visibility='private')
        scale.change(change_scale, inputs=scale,
                     outputs=[predictor, evaluator, steps, result, comparison, download, status],
                     queue=False, api_visibility='private')
        refresh.click(refresh_files, inputs=[image_choice, scale, preview],
                      outputs=[image_choice, gallery, predictor, evaluator, steps, run],
                      queue=False, api_visibility='private')
        run.click(application.run, inputs=[preview, scale, predictor, evaluator, steps, tile, tile_pad, seed],
                  outputs=[result, comparison, download, status], concurrency_limit=1,
                  concurrency_id='sipe_inference', api_name='super_resolve')
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser(description='SIPE image super-resolution Gradio application')
    parser.add_argument('--images-dir', type=Path, default=ROOT / 'images', help='Example image directory')
    parser.add_argument('--experiments-dir', type=Path, default=ROOT / 'experiments', help='Model checkpoint directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Optional directory to keep PNG results; defaults to temporary storage')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--cpu-threads', type=int, default=8, help='Number of CPU inference threads')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true', help='Create a public Gradio sharing link')
    parser.add_argument('--inbrowser', action='store_true', help='Open the application in a browser on startup')
    args = parser.parse_args()
    if args.cpu_threads <= 0:
        parser.error('--cpu-threads must be greater than 0.')
    torch.set_num_threads(args.cpu_threads)
    try:
        application = SIPEApplication(args.images_dir, args.experiments_dir, args.output_dir, args.device)
    except ValueError as error:
        parser.error(str(error))
    create_app(application).launch(
        server_name=args.host, server_port=args.port, share=args.share, inbrowser=args.inbrowser,
        theme=gr.themes.Soft(primary_hue='blue', font=APP_FONT, font_mono=APP_FONT),
        allowed_paths=[str(application.images_dir), str(application.output_dir)],
    )


if __name__ == '__main__':
    main()
