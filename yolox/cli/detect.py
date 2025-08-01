# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import sys
import time
from loguru import logger

import cv2

import torch
import onnxruntime
import torchvision
import numpy as np


from yolox.config import YoloxConfig
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    configure_module,
    fuse_model,
    get_model_info,
    postprocess,
    vis,
)

from .utils import resolve_config, get_unique_output_name

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"]


def is_image_file(filepath):
    """Check if a file is an image based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in IMAGE_EXT


def is_video_file(filepath):
    """Check if a file is a video based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in VIDEO_EXT


def detect_input_type(path):
    """Automatically detect the type of input (image, video, folder, or webcam)."""
    if path is None:
        return "webcam"
    
    if not os.path.exists(path):
        logger.error(f"Path {path} does not exist.")
        return None
    
    if os.path.isfile(path):
        if is_image_file(path):
            return "image"
        elif is_video_file(path):
            return "video"
        else:
            logger.error(f"Unsupported file type: {path}")
            return None
    elif os.path.isdir(path):
        return "folder"
    else:
        logger.error(f"Path {path} is neither a file nor a directory.")
        return None


def get_file_list(path):
    """Get all image and video files from a directory, sorted by type."""
    image_files = []
    video_files = []
    
    for maindir, _, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if is_image_file(apath):
                image_files.append(apath)
            elif is_video_file(apath):
                video_files.append(apath)
    
    image_files.sort()
    video_files.sort()
    
    return image_files, video_files


def make_parser():
    parser = argparse.ArgumentParser("yolox detect")
    parser.add_argument(
        "type", 
        nargs='?',  # Make type optional
        choices=["image", "video", "webcam", "auto"], 
        default="auto",
        help="type of detection (auto-detect if not specified)"
    )
    parser.add_argument("-c", "--config", required=True, type=str, help="A builtin config such as yolox-s")
    parser.add_argument("--weights", default=None, type=str, help="path to weights file. It can be a .pth checkpoint or a .onnx model.")
    parser.add_argument("--path", help="path to images/video/folder. For webcam, this is ignored.")
    parser.add_argument("--camid", type=int, default=0, help="webcam camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to a text file containing class labels (one per line). Overrides default COCO labels.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, _, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def onnx_postprocess(output, num_classes, conf_thre, nms_thre):
    """
    Postprocesses the output of an ONNX model.

    Args:
        output (np.ndarray): The raw output from the onnxruntime session.
        num_classes (int): The number of classes.
        conf_thre (float): The confidence threshold.
        nms_thre (float): The NMS threshold.

    Returns:
        list[torch.Tensor or None]: A list of detections for each image.
    """
    # The output of the ONNX model is expected to have gone through a DeepStream-style
    # post-processing step within the model itself.
    predictions = output[0]  # batch size 1, first output
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    class_ids = predictions[:, 5]

    # filter by score
    keep = scores > conf_thre
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    if len(boxes) == 0:
        return [None]

    # nms
    keep = torchvision.ops.batched_nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        torch.from_numpy(class_ids).long(),
        nms_thre
    )

    boxes_tensor = torch.from_numpy(boxes[keep])
    scores_tensor = torch.from_numpy(scores[keep])
    class_ids_tensor = torch.from_numpy(class_ids[keep])

    # Reconstruct to match pytorch predictor output format:
    # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    # For this ONNX model, score is already combined, so we use 1.0 for obj_conf.
    obj_conf = torch.ones_like(scores_tensor)

    detections = torch.cat(
        [boxes_tensor, obj_conf.unsqueeze(1), scores_tensor.unsqueeze(1), class_ids_tensor.unsqueeze(1).float()], 1
    )
    return [detections]


def process_mixed_content(predictor, folder_path, args, config):
    """Process a folder containing both images and videos."""
    image_files, video_files = get_file_list(folder_path)
    
    logger.info(f"Found {len(image_files)} image files and {len(video_files)} video files")
    
    # Process images first
    if image_files:
        logger.info("Processing images...")
        for image_name in image_files:
            outputs, img_info = predictor.inference(image_name)
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                save_folder = os.path.join(config.output_dir, args.experiment_name, "images")
                os.makedirs(save_folder, exist_ok=True)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                if not isinstance(predictor, ONNXPredictor):
                    logger.info("Saving detection result in {}", save_file_name)
                cv2.imwrite(save_file_name, result_image)
            else:
                cv2.imshow(os.path.basename(image_name), result_image)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
    
    # Process videos
    if video_files:
        logger.info("Processing videos...")
        for video_path in video_files:
            logger.info(f"Processing video: {video_path}")
            # Create a temporary args object for video processing
            temp_args = type('Args', (), {
                'type': 'video',
                'path': video_path,
                'save_result': args.save_result,
                'camid': args.camid
            })()
            
            # Process the video
            imageflow_demo(predictor, os.path.join(config.output_dir, args.experiment_name, "videos"), temp_args)


class Predictor:
    def __init__(
        self,
        model,
        config: YoloxConfig,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = config.num_classes
        self.confthre = config.test_conf
        self.nmsthre = config.nmsthre
        self.test_size = config.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, config.test_size[0], config.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            return [None], img_info

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s", time.time() - t0)
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

class ONNXPredictor:
    def __init__(self, model_path, config, cls_names=COCO_CLASSES, device='cpu'):
        providers = ['CUDAExecutionProvider' if 'gpu' in device else 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.cls_names = cls_names
        self.num_classes = config.num_classes
        self.confthre = config.test_conf
        self.nmsthre = config.nmsthre
        self.test_size = config.test_size
        self.device = device
        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = "webcam" if img is not None else None

        if img is None:
            return [None], img_info

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)

        ort_inputs = {self.input_name: img[None, :, :, :]}
        t0 = time.time()
        output = self.session.run(None, ort_inputs)
        logger.info("Infer time: {:.4f}s", time.time() - t0)
        return output, img_info

    def visual(self, output, img_info, conf_thre=None, nms_thre=None):
        if conf_thre is None:
            conf_thre = self.confthre
        if nms_thre is None:
            nms_thre = self.nmsthre
        predictions = output[0]  # remove batch dim

        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = predictions[:, 5]

        # filter by score
        keep = scores > conf_thre
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # nms
        if len(boxes) > 0:
            keep = torchvision.ops.batched_nms(
                torch.from_numpy(boxes),
                torch.from_numpy(scores),
                torch.from_numpy(class_ids).long(),
                nms_thre
            )
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        ratio = img_info["ratio"]
        boxes = boxes / ratio

        return vis(img_info["raw_img"], boxes, scores, class_ids, conf_thre, self.cls_names)

def imageflow_demo(predictor, vis_folder, args):
    cap = cv2.VideoCapture(args.path if args.type == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        os.makedirs(vis_folder, exist_ok=True)
        if args.type == "video":
            save_path = os.path.join(vis_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(vis_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(argv: list[str]) -> None:
    configure_module()
    args = make_parser().parse_args(argv)

    config = resolve_config(args.config)

    # Set output directory to out/detect and auto-increment experiment name if exists
    if not hasattr(args, 'name') or args.name is None:
        args.name = config.name if hasattr(config, 'name') else args.config
    base_output_dir = os.path.join("out", "detect")
    os.makedirs(base_output_dir, exist_ok=True)
    output_dir, experiment_name = get_unique_output_name(base_output_dir, args.name)
    config.output_dir = output_dir
    args.experiment_name = experiment_name

    # --- Begin: Load custom labels if provided ---
    custom_labels = None
    if args.labels is not None:
        if not os.path.isfile(args.labels):
            logger.error(f"Label file {args.labels} does not exist.")
            sys.exit(1)
        with open(args.labels, 'r') as f:
            custom_labels = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(custom_labels)} custom labels from {args.labels}")
    # --- End: Load custom labels if provided ---
    
    is_onnx = args.weights is not None and args.weights.endswith(".onnx")

    if args.weights is not None and not is_onnx:
        import torch
        logger.info("loading checkpoint from {}", args.weights)
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        model_state = ckpt.get('model', ckpt)
        cls_pred_key = None
        for key in model_state.keys():
            if 'head.cls_preds.0.weight' in key:
                cls_pred_key = key
                break
        if cls_pred_key is not None:
            detected_num_classes = model_state[cls_pred_key].shape[0]
            config.num_classes = detected_num_classes
            logger.info(f"Auto-detected num_classes from checkpoint: {detected_num_classes}")
        else:
            logger.warning("Could not auto-detect num_classes from checkpoint; using config default.")
        model = config.get_model()
        logger.info("Model Summary: {}", get_model_info(model, config.test_size))
        if args.device == "gpu":
            model.cuda()
            if args.fp16:
                model.half()
        model.eval()
        model.load_state_dict(ckpt.get("model", ckpt))
        predictor = Predictor(
            model,
            config,
            cls_names=custom_labels if custom_labels is not None else COCO_CLASSES,
            device=args.device,
            fp16=args.fp16,
            legacy=args.legacy,
        )
    elif is_onnx:
        model = config.get_model()
        logger.info("Model Summary: {}", get_model_info(model, config.test_size))
        predictor = ONNXPredictor(
            args.weights,
            config,
            cls_names=custom_labels if custom_labels is not None else COCO_CLASSES,
            device=args.device,
        )
    else:
        model = config.get_model()
        logger.info("Model Summary: {}", get_model_info(model, config.test_size))
        if args.device == "gpu":
            model.cuda()
            if args.fp16:
                model.half()
        model.eval()
        from yolox.models.yolox import YoloxModule
        model = YoloxModule.from_pretrained(args.config, config, args.device)
        predictor = Predictor(
            model,
            config,
            cls_names=custom_labels if custom_labels is not None else COCO_CLASSES,
            device=args.device,
            fp16=args.fp16,
            legacy=args.legacy,
        )

    # Auto-detect input type if not specified or set to auto
    if args.type == "auto" or args.type is None:
        input_type = detect_input_type(args.path)
        if input_type is None:
            sys.exit(1)
        logger.info(f"Auto-detected input type: {input_type}")
        args.type = input_type
    
    # Process based on detected or specified type
    if args.type == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            outputs, img_info = predictor.inference(image_name)
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                save_folder = os.path.join(config.output_dir, args.experiment_name)
                os.makedirs(save_folder, exist_ok=True)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                logger.info("Saving detection result in {}", save_file_name)
                cv2.imwrite(save_file_name, result_image)
            else:
                cv2.imshow(os.path.basename(image_name), result_image)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
    elif args.type == "video":
        imageflow_demo(predictor, os.path.join(config.output_dir, args.experiment_name), args)
    elif args.type == "webcam":
        imageflow_demo(predictor, os.path.join(config.output_dir, args.experiment_name), args)
    elif args.type == "folder":
        process_mixed_content(predictor, args.path, args, config)

if __name__ == "__main__":
    main(sys.argv[1:])
