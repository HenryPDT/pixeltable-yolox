import os
import onnx
import torch
import torch.nn as nn

# Imports from the pixeltable-yolox library
from yolox.models import YoloxModule
from yolox.config import YoloxConfig
from yolox.cli.utils import resolve_config, parse_model_config_opts
from yolox.utils import replace_module
from yolox.models.network_blocks import SiLU


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[:, :, :4]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype, device=boxes.device
        )
        boxes @= convert_matrix
        objectness = x[:, :, 4:5]
        scores, labels = torch.max(x[:, :, 5:], dim=-1, keepdim=True)
        scores *= objectness
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def yolox_export(weights: str, config_str: str, opts: dict | None = None) -> (nn.Module, YoloxConfig):
    """
    Loads a pixeltable-yolox model and prepares it for ONNX export.
    """
    config = resolve_config(config_str)
    # Apply runtime overrides to the config BEFORE creating the model
    if opts:
        config.update(opts)

    model = YoloxModule.from_pretrained(weights, config)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = True
    return model, config


def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)


def main(args):

    suppress_warnings()

    print(f'\nStarting: {args.weights}')

    print('Opening YOLOX model')

    # Parse the -D options from the command line
    opts = parse_model_config_opts(args.D)

    device = torch.device('cpu')
    model, config = yolox_export(args.weights, args.config, opts)

    # Wrap the model with the custom output layer
    model = nn.Sequential(model, DeepStreamOutput())
    model.to(device)

    img_size = config.input_size
    print(f'Using input size: {img_size}')
    print(f'Model has {config.num_classes} classes.') # You can add this to verify

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = f'{os.path.splitext(args.weights)[0]}.onnx'

    dynamic_axes = {
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    } if args.dynamic else None

    print(f'Exporting the model to ONNX at {onnx_output_file}')
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    if args.simplify:
        print('Simplifying the ONNX model')
        try:
            import onnxslim
            model_onnx = onnx.load(onnx_output_file)
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, onnx_output_file)
        except ImportError:
            print("onnxslim is not installed, skipping simplification. Please install with 'pip install onnxslim'.")
        except Exception as e:
            print(f"Error during ONNX simplification: {e}")


    print(f'Successfully exported to: {onnx_output_file}\n')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DeepStream YOLOX conversion for pixeltable-yolox')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pth) file path (required)')
    parser.add_argument('-cfg', '--config', required=True, help='YOLOX model config name (e.g., yolox-s) or path to a custom config class (required)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='Simplify the ONNX model using onnxslim')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch size in the ONNX model')
    parser.add_argument('--batch', type=int, default=1, help='Static batch size for the ONNX model')
    parser.add_argument('-D', type=str, metavar="OPT=VALUE", help="Override model configuration option (e.g., -D num_classes=5)", action="append")
    
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit(f'Invalid weights file: {args.weights}')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set --dynamic and --batch > 1 at the same time.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
