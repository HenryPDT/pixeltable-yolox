<!-- file_path: README.md -->
<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/assets/logo.png"
     alt="YoloX" width="350"></div>
<br>

`pixeltable-yolox` is a lightweight, Apache-licensed object detection library built on PyTorch. It is a fork of the
[MegVii YOLOX package](https://github.com/Megvii-BaseDetection/YOLOX) originally authored by Zheng Ge et al,
modernized for recent versions of Python and refactored to be more easily usable as a Python library.

## Table of Contents
- [About this Fork](#about-this-fork)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Inference with Python](#inference-with-python)
  - [Pretrained Models](#pretrained-models)
- [Command-Line Interface](#command-line-interface)
  - [Detection](#detection)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Export to ONNX](#export-to-onnx)
- [Advanced Usage](#advanced-usage)
  - [Separate Module/Processor Steps](#separate-moduleprocessor-steps)
- [Contributing](#contributing)
- [In memory of Dr. Jian Sun](#in-memory-of-dr-jian-sun)

## About this Fork

The original YOLOX implementation, while powerful, has been updated only sporadically since 2022 and now faces
compatibility issues with current Python environments, dependencies, and platforms like Google Colab. This fork aims
to provide a reliable, up-to-date, and easy-to-use version of YOLOX that maintains its Apache license, ensuring it
remains accessible for academic and commercial use.

This fork is a work in progress. So far, it contains the following changes to the base YOLOX repo:

- `pip install`able with all versions of Python (3.9+)
- New `YoloxProcessor` class to simplify inference
- Refactored CLI for training and evaluation
- Improved test coverage

### Who are we?
Pixeltable, Inc. is a venture-backed AI infrastructure startup. Our core product is
[pixeltable](https://github.com/pixeltable/pixeltable), a database and orchestration system purpose-built for
multimodal AI workloads.

We chose to build upon YOLOX both to simplify our own integration, and also to give something back to the community
that will (hopefully) prove useful. The Pixeltable team has decades of collective experience in open source development.
Our backgrounds include companies such as Google, Cloudera, Twitter, Amazon, and Airbnb, that have a strong commitment
to open source development and collaboration. Thanks for your interest! For any questions or feedback, please contact us at `contact@pixeltable.com`.

## Installation

First, clone the repository to your local machine:
```bash
git clone https://github.com/HenryPDT/pixeltable-yolox.git
cd pixeltable-yolox
```

Create a new conda environment with Python 3.10:
```bash
conda create -n yolox python=3.10
conda activate yolox
```

Then install the package and its dependencies:
```bash
pip install -e .
```

## Getting Started

### Inference with Python

Here is a simple example of how to perform inference on an image using a pretrained model.

```python
import requests
from PIL import Image
from yolox.models import Yolox

url = "https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/tests/data/000000000001.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Load the pretrained 'yolox-s' model
model = Yolox.from_pretrained("yolox_s")

# Run inference
# Inputs can be a list of PIL images, file paths, or a mix of both.
result = model([image])
```

This yields a list of detections for each input image. Each detection is a dictionary containing bounding boxes, scores, and labels:

```python
[{'bboxes': [
   (272.36126708984375, 3.5648040771484375, 640.4871826171875, 223.2653350830078),
   (26.643890380859375, 118.68254089355469, 459.80706787109375, 315.089111328125),
   (259.41485595703125, 152.3223114013672, 295.37054443359375, 230.41783142089844)],
  'scores': [0.9417160943584335, 0.8170979975670818, 0.8095869439224117],
  'labels': [7, 2, 12]}]
```
The labels correspond to COCO category indices. You can map them to class names:
```python
from yolox.data.datasets import COCO_CLASSES

print(COCO_CLASSES[7])
# 'truck'
```

### Pretrained Models

This library automatically downloads and caches pretrained weights for standard models (e.g., `yolox-s`, `yolox-m`, etc.) when you use `Yolox.from_pretrained()`.

For a full list of available models and links to their weights, please see the [Model Zoo](docs/model_zoo.md). You can manually download these weights and pass the local file path to the `--weights` argument in the CLI commands.

## Command-Line Interface

`pixeltable-yolox` provides a command-line interface `yolox` for detection, training, evaluation, and exporting models.

### Detection

You can run inference on images, videos, or a webcam feed using the `yolox detect` command. The command can auto-detect the input type.

**Download pretrained weights**

The first time you run detection with a standard model (like `yolox-s`), the pretrained weights will be downloaded and cached automatically. You can also manually download weights from the [Model Zoo](docs/model_zoo.md) and use the `--weights` argument to specify the path.

**On an image:**
```bash
yolox detect --config yolox-s --path assets/dog.jpg --save_result
```
This will run detection with `yolox-s` model (weights are downloaded automatically) and save the output image with bounding boxes in the `out/detect/yolox_s/` directory.

**On a video:**
```bash
yolox detect --config yolox-s --path /path/to/your/video.mp4 --save_result
```

**Using custom weights and labels:**
```bash
yolox detect --config yolox-s --weights /path/to/your/weights.pth --path assets/dog.jpg --labels labels.txt --save_result
```

For more options, run:
```bash
yolox detect -h
```

### Training

You can train a YOLOX model on a COCO-style dataset using the `yolox train` command.

**1. Prepare your dataset**

First, organize your dataset in the COCO format. By default, the trainer looks for the dataset in `./datasets/COCO`. You can create a symbolic link:

```bash
ln -s /path/to/your/COCO ./datasets/COCO
```

The directory structure should be:
```
./datasets/COCO/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
    # image files
  val2017/
    # image files
```

For instructions on training with a custom dataset format, see [Train on Custom Data](docs/train_custom_data.md).

**2. Start Training**

To start training, run the `yolox train` command. For example, to train `yolox-s`:

```bash
# -c: model config (yolox-s)
# -d: number of devices (GPUs)
# -b: batch size
# --fp16: use mixed precision training
# -o: occupy GPU memory for faster training
yolox train -c yolox-s -d 1 -b 8 --fp16 -o
```
Checkpoints and logs will be saved to `out/train/yolox_s/`.

For a full list of training options:
```bash
yolox train -h
```

### Evaluation

To evaluate a trained model's performance on the COCO validation set:

```bash
yolox eval -c yolox-s --ckpt /path/to/your/yolox_s.pth -b 8 -d 1
```

For more evaluation options:
```bash
yolox eval -h
```

### Export to ONNX

You can export a trained model from a `.pth` checkpoint to the ONNX format. This is useful for deployment in various environments, including NVIDIA DeepStream.

```bash
# -w: path to input weights (.pth)
# -cfg: model config name (e.g., yolox-s)
yolox export_onnx -w /path/to/yolox_s.pth -cfg yolox-s
```
This will create a `yolox_s.onnx` file in the same directory as the weights.

The export script offers options for dynamic shapes and simplification with `onnxslim`:
```bash
# Export with dynamic batch size and shape, and simplify the model
yolox export_onnx -w /path/to/yolox_s.pth -cfg yolox-s --dynamic --dynamic-shape --simplify
```

For all export options:
```bash
yolox export_onnx -h
```

## Advanced Usage

### Separate Module/Processor Steps

To separate out the PyTorch module from image pre- and post-processing during inference (as is typical in the Hugging
Face transformers API):

```python
from yolox.models import YoloxModule, YoloxProcessor

module = YoloxModule.from_pretrained("yolox_s")
processor = YoloxProcessor("yolox_s")

# Pre-process images
tensor = processor([image])

# Run model
output = module(tensor)

# Post-process results
result = processor.postprocess([image], output)
```

## Contributing

We welcome contributions from the community! If you're interested in helping maintain and improve `pixeltable-yolox`,
check out the [contributors' guide](CONTRIBUTING.md).

## In memory of Dr. Jian Sun

Without the guidance of [Dr. Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ), YOLOX would not have
been released and open sourced to the community.
The passing away of Dr. Sun is a huge loss to the Computer Vision field. We add this section here to express our
remembrance and condolences to Dr. Sun.
It is hoped that every AI practitioner in the world will stick to the belief of "continuous innovation to expand
cognitive boundaries, and extraordinary technology to achieve product value" and move forward all the way.

<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable-yolox/main/assets/sunjian.png"
     alt="Dr. Jian Sun" width="200">
</div>