import os
import onnx
import torch
import torch.nn as nn

# Imports from the pixeltable-yolox library
from yolox.models import YoloxModule
from yolox.config import YoloxConfig
from yolox.cli.utils import resolve_config
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


def yolox_export(weights: str, config_str: str) -> (nn.Module, YoloxConfig):
    """
    Loads a pixeltable-yolox model and prepares it for ONNX export.
    """
    config = resolve_config(config_str)
    
    # Load the checkpoint to detect num_classes before creating the model
    print('Loading checkpoint to detect model configuration...')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    
    # Extract num_classes from the checkpoint by looking at the classification head
    # The cls_preds layers have shape [num_classes, ...], so we can get num_classes from there
    model_state = checkpoint.get('model', checkpoint)
    
    # Look for the first classification prediction layer to get num_classes
    cls_pred_key = None
    for key in model_state.keys():
        if 'head.cls_preds.0.weight' in key:
            cls_pred_key = key
            break
    
    if cls_pred_key is None:
        raise ValueError("Could not find classification prediction layer in checkpoint to detect num_classes")
    
    # Get num_classes from the shape of the classification layer
    detected_num_classes = model_state[cls_pred_key].shape[0]
    print(f'Auto-detected num_classes: {detected_num_classes}')
    
    # Update config with detected num_classes before creating the model
    config.num_classes = detected_num_classes

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

    device = torch.device('cpu')
    model, config = yolox_export(args.weights, args.config)

    # Wrap the model with the custom output layer
    model = nn.Sequential(model, DeepStreamOutput())
    model.to(device)

    # Use custom image size if provided, otherwise use config default
    img_size = args.imgsz if args.imgsz else config.input_size
    print(f'Using input size: {img_size}')
    print(f'Model has {config.num_classes} classes.')

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = f'{os.path.splitext(args.weights)[0]}.onnx'

    # Set dynamic_axes according to the selected flags
    dynamic_axes = None
    if args.dynamic or args.dynamic_shape:
        dynamic_axes = {
            'images': {},
            'output': {}
        }
        
        if args.dynamic:
            dynamic_axes['images'][0] = 'batch'
            dynamic_axes['output'][0] = 'batch'
            
        if args.dynamic_shape:
            dynamic_axes['images'][2] = 'height'
            dynamic_axes['images'][3] = 'width'

    print(f'Exporting the model to ONNX at {onnx_output_file}')
    torch.onnx.export(
        model,
        onnx_input_im,
        onnx_output_file,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['images'],
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
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, help='Image size (height, width) for ONNX export (default: use config input_size)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='Simplify the ONNX model using onnxslim')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch size in the ONNX model (can be combined with --dynamic-shape)')
    parser.add_argument('--dynamic-shape', action='store_true', help='Enable dynamic input size (height/width) in the ONNX model (can be combined with --dynamic)')
    parser.add_argument('--batch', type=int, default=1, help='Static batch size for the ONNX model')
    
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit(f'Invalid weights file: {args.weights}')
    
    # Process imgsz argument
    if args.imgsz:
        if len(args.imgsz) == 1:
            args.imgsz = (args.imgsz[0], args.imgsz[0])  # square
        elif len(args.imgsz) == 2:
            args.imgsz = tuple(args.imgsz)  # (height, width)
        else:
            raise SystemExit('--imgsz must be 1 or 2 integers (height, width)')
    
    # Validate dynamic batch size compatibility
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set --dynamic with --batch > 1 at the same time.')
    
    # Validate imgsz and dynamic-shape compatibility
    if args.imgsz and args.dynamic_shape:
        raise SystemExit('Cannot set --imgsz with --dynamic-shape. Use either fixed input size (--imgsz) or dynamic input size (--dynamic-shape), not both.')
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
