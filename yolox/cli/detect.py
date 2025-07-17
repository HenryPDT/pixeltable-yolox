# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import sys
import time
from loguru import logger

import cv2

import torch

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

from .utils import resolve_config

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("yolox detect")
    parser.add_argument(
        "type", choices=["image", "video", "webcam"], help="type of detection"
    )
    parser.add_argument("-c", "--config", required=True, type=str, help="A builtin config such as yolox-s")
    parser.add_argument("--ckpt", default=None, type=str, help="path to checkpoint file")
    parser.add_argument("--path", help="path to images/video. For webcam, this is ignored.")
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

    # --- Begin: Auto-detect num_classes from checkpoint if provided ---
    if args.ckpt is not None:
        import torch
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
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
    # --- End: Auto-detect num_classes from checkpoint ---

    if args.conf is not None:
        config.test_conf = args.conf
    if args.nms is not None:
        config.nmsthre = args.nms
    if args.tsize is not None:
        config.test_size = (args.tsize, args.tsize)

    model = config.get_model()
    logger.info("Model Summary: {}", get_model_info(model, config.test_size))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    if args.ckpt is None:
        from yolox.models.yolox import YoloxModule
        model = YoloxModule.from_pretrained(args.config, config, args.device)
    else:
        logger.info("loading checkpoint from {}", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt))

    if args.fuse:
        model = fuse_model(model)

    predictor = Predictor(
        model,
        config,
        cls_names=custom_labels if custom_labels is not None else COCO_CLASSES,
        device=args.device,
        fp16=args.fp16,
        legacy=args.legacy,
    )

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
                save_folder = os.path.join(config.output_dir, args.config)
                os.makedirs(save_folder, exist_ok=True)
                save_file_name = os.path.join(save_folder, os.path.basename(image_name))
                logger.info("Saving detection result in {}", save_file_name)
                cv2.imwrite(save_file_name, result_image)
            else:
                cv2.imshow(os.path.basename(image_name), result_image)
                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
    elif args.type in ["video", "webcam"]:
        imageflow_demo(predictor, os.path.join(config.output_dir, args.config), args)

if __name__ == "__main__":
    main(sys.argv[1:])
