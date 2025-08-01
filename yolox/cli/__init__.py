# Copyright (c) Megvii, Inc. and its affiliates.

import sys
import yolox

from . import detect, eval, train, export_onnx


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == 'train':
        train.main(args[1:])
        return
    if args and args[0] == 'eval':
        eval.main(args[1:])
        return
    if args and args[0] == 'detect':
        detect.main(args[1:])
        return
    if args and args[0] == 'export_onnx':
        export_onnx.main(args[1:])
        return

    print(f'This is pixeltable-yolox, version {yolox.__version__}.\n')
    if args and args[0] in {'-h', '--help'}:
        print('Usage: yolox <command> [arguments]')
        print('Commands:')
        print('  train - Train a Yolo model')
        print('  eval  - Evaluate a Yolo model')
        print('  detect - Run inference on images or video')
        print('  export_onnx - Export a trained model to ONNX format')
        print('For help on an individual command: yolox <command> -h')
        return

    if args:
        print(f'Unrecognized command: {args[0]}')
    print(f'For help: yolox -h')
