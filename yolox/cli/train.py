# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import sys
import warnings

import torch
from loguru import logger
from torch.backends import cudnn

from yolox.config import YoloxConfig
from yolox.core import launch
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

from .utils import parse_model_config_opts, resolve_config, get_unique_output_name


def make_parser():
    parser = argparse.ArgumentParser(
        "yolox train",
        description="Train YOLOX object detection models",
        epilog="""
Examples:
  # Basic training with a predefined model (auto-increments if out/yolox_s exists)
  yolox train -c yolox_s -b 16 --epochs 100

  # Custom training with specific parameters and output directory (saves to out/my_experiment)
  yolox train -c yolox_m --epochs 200 --lr 0.01 --num-classes 10 --data-dir /path/to/data --output-dir my_experiment

  # Training with augmentation settings (saves to out/augmented_exp)
  yolox train -c yolox_l --mosaic-prob 0.8 --mixup-prob 0.5 --flip-prob 0.5 --output-dir augmented_exp

  # Advanced configuration override (saves to out/custom_training)
  yolox train -c yolox_s -D scheduler=cosine -D momentum=0.95 --output-dir custom_training
  
  Note: Output directories are automatically incremented (e.g., yolox_s_2, yolox_s_3) to avoid overwriting existing experiments.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", type=str, help="A builtin config such as yolox_s, or a custom Python class given as {module}:{classname} such as yolox.config:YoloxS")
    parser.add_argument("-n", "--name", type=str, default=None, help="Model name; defaults to the model name specified in config")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        nargs="?",
        const="ram",
        help="Caching imgs to ram/disk for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics. \
                Implemented loggers include `tensorboard`, `mlflow` and `wandb`.",
        default="tensorboard"
    )

    # Training parameters - most commonly used ones
    training_group = parser.add_argument_group('Training Parameters', 'Common training parameters')
    training_group.add_argument(
        "--epochs", type=int, default=None,
        help="Maximum training epochs (default: 300)"
    )
    training_group.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        help="Learning rate per image (default: 0.01/64)"
    )
    training_group.add_argument(
        "--weight-decay", type=float, default=None,
        help="Weight decay (default: 5e-4)"
    )
    training_group.add_argument(
        "--warmup-epochs", type=int, default=None,
        help="Warmup epochs (default: 5)"
    )
    training_group.add_argument(
        "--no-aug-epochs", type=int, default=None,
        help="Number of epochs to disable augmentation at the end (default: 15)"
    )
    training_group.add_argument(
        "--eval-interval", type=int, default=None,
        help="Evaluation interval in epochs (default: 10)"
    )
    training_group.add_argument(
        "--save-history", action="store_true",
        help="Save checkpoint history (default: False)"
    )
    training_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Experiment name for output directory. Results will be saved to out/{name}. Auto-increments if directory exists (e.g., out/my_exp_2)"
    )

    # Data parameters
    data_group = parser.add_argument_group('Data Parameters', 'Dataset and augmentation parameters')
    data_group.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to dataset directory"
    )
    data_group.add_argument(
        "--num-classes", type=int, default=None,
        help="Number of classes (default: 80)"
    )
    # Add --imgsz argument
    data_group.add_argument(
        "--imgsz", type=int, nargs='+', default=None,
        help="Image size for both training and testing. Provide one value for square (e.g., 640) or two for (height width), e.g., 96 256. Sets both input_size and test_size."
    )
    data_group.add_argument(
        "--input-size", type=str, default=None,
        help="Input size as 'height,width' (default: 640,640)"
    )
    data_group.add_argument(
        "--mosaic-prob", type=float, default=None,
        help="Probability of applying mosaic augmentation (default: 1.0)"
    )
    data_group.add_argument(
        "--mixup-prob", type=float, default=None,
        help="Probability of applying mixup augmentation (default: 1.0)"
    )
    data_group.add_argument(
        "--hsv-prob", type=float, default=None,
        help="Probability of applying HSV augmentation (default: 1.0)"
    )
    data_group.add_argument(
        "--flip-prob", type=float, default=None,
        help="Probability of applying flip augmentation (default: 0.5)"
    )
    data_group.add_argument(
        '--random-size', type=int, nargs=2, default=None,
        help="Multi-scale range as 'min max' factors (e.g., 10 20). Overrides multiscale-range."
    )
    data_group.add_argument(
        '--multiscale-range', type=int, default=None,
        help="Simple multi-scale range factor (e.g., 5). Used if random-size is not set."
    )

    # Model parameters
    model_group = parser.add_argument_group('Model Parameters', 'Model architecture parameters')
    model_group.add_argument(
        "--depth", type=float, default=None,
        help="Model depth factor (default depends on model size)"
    )
    model_group.add_argument(
        "--width", type=float, default=None,
        help="Model width factor (default depends on model size)"
    )

    # Advanced configuration (keep existing -D option)
    parser.add_argument(
        "-D",
        type=str,
        metavar="OPT=VALUE",
        help="Override any model configuration option with custom value (example: -D num_classes=71)",
        action="append",
    )
    return parser


def train(config: YoloxConfig, args):
    if config.seed is not None:
        assert isinstance(config.seed, int)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = config.get_trainer(args)
    trainer.train()


def convert_args_to_config_opts(args):
    """Convert command-line arguments to configuration options."""
    config_opts = {}
    
    # Training parameters
    if args.epochs is not None:
        config_opts['max_epoch'] = str(args.epochs)
    if args.lr is not None:
        config_opts['basic_lr_per_img'] = str(args.lr)
    if args.weight_decay is not None:
        config_opts['weight_decay'] = str(args.weight_decay)
    if args.warmup_epochs is not None:
        config_opts['warmup_epochs'] = str(args.warmup_epochs)
    if args.no_aug_epochs is not None:
        config_opts['no_aug_epochs'] = str(args.no_aug_epochs)
    if args.eval_interval is not None:
        config_opts['eval_interval'] = str(args.eval_interval)
    if args.save_history:
        config_opts['save_history_ckpt'] = 'True'
    # Note: output_dir is handled separately in main() function
    
    # Data parameters
    if args.data_dir is not None:
        config_opts['data_dir'] = args.data_dir
    if args.num_classes is not None:
        config_opts['num_classes'] = str(args.num_classes)
    # Handle imgsz and input_size precedence
    if args.imgsz is not None:
        # Accept one or two ints
        if len(args.imgsz) == 1:
            h = w = int(args.imgsz[0])
        elif len(args.imgsz) == 2:
            h, w = map(int, args.imgsz)
        else:
            raise ValueError(f"--imgsz expects 1 or 2 values, got {args.imgsz}")
        
        # Set both input_size and test_size to the same value for consistency
        config_opts['input_size'] = f'({h}, {w})'
        config_opts['test_size'] = f'({h}, {w})'
    elif args.input_size is not None:
        # Parse input size from 'height,width' format
        try:
            height, width = map(int, args.input_size.split(','))
            config_opts['input_size'] = f'({height}, {width})'
        except ValueError:
            raise ValueError(f"Invalid input size format: {args.input_size}. Use 'height,width' format (e.g., '640,640')")
    if args.mosaic_prob is not None:
        config_opts['mosaic_prob'] = str(args.mosaic_prob)
    if args.mixup_prob is not None:
        config_opts['mixup_prob'] = str(args.mixup_prob)
    if args.hsv_prob is not None:
        config_opts['hsv_prob'] = str(args.hsv_prob)
    if args.flip_prob is not None:
        config_opts['flip_prob'] = str(args.flip_prob)
    
    # Handle multi-scale training arguments, with random_size taking precedence
    if args.random_size is not None:
        config_opts['random_size'] = f'({args.random_size[0]}, {args.random_size[1]})'
    elif args.multiscale_range is not None:
        config_opts['multiscale_range'] = str(args.multiscale_range)
    
    # Model parameters
    if args.depth is not None:
        config_opts['depth'] = str(args.depth)
    if args.width is not None:
        config_opts['width'] = str(args.width)
    
    return config_opts


def main(argv: list[str]) -> None:
    configure_module()
    args = make_parser().parse_args(argv)
    if args.config is None:
        raise AttributeError("Please specify a model configuration.")
    config = resolve_config(args.config)
    
    # Convert command-line arguments to config options
    arg_config_opts = convert_args_to_config_opts(args)
    
    # Parse -D options
    d_config_opts = parse_model_config_opts(args.D)
    
    # Merge config options (command-line args take precedence over defaults, -D options take precedence over command-line args)
    config_opts = {**arg_config_opts, **d_config_opts}
    
    config.update(config_opts)
    config.validate()

    if not args.name:
        args.name = config.name

    # Handle output directory logic
    if args.output_dir is not None:
        # User specified a custom output directory name
        # We want: out/train/{custom_name} instead of out/{custom_name}
        base_output_dir = os.path.join("out", "train")
        os.makedirs(base_output_dir, exist_ok=True)
        output_dir, experiment_name = get_unique_output_name(base_output_dir, args.output_dir)
        config.output_dir = output_dir
        args.name = experiment_name
    else:
        # No custom output dir specified, use default behavior with auto-increment
        # We want: out/train/{model_name} instead of out/{model_name}
        base_output_dir = os.path.join("out", "train")
        os.makedirs(base_output_dir, exist_ok=True)
        output_dir, experiment_name = get_unique_output_name(base_output_dir, args.name)
        config.output_dir = output_dir
        args.name = experiment_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    if args.cache is not None:
        config.dataset = config.get_dataset(cache=True, cache_type=args.cache)

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        train,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(config, args),
    )


if __name__ == "__main__":
    main(sys.argv[1:])