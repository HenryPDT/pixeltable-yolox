# Copyright (c) Megvii, Inc. and its affiliates.

import time

import torch
from loguru import logger


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        t0 = time.monotonic()
        logger.info(
            "DataPrefetcher: iter(dataloader) — if this hangs, workers are stuck starting "
            "or the first sample never returns."
        )
        self.loader = iter(loader)
        t1 = time.monotonic()
        logger.info(f"DataPrefetcher: iterator created in {t1 - t0:.2f}s")

        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image

        logger.info(
            "DataPrefetcher: first next(loader) + H2D — runs first __getitem__ in workers "
            "(Albumentations/OpenCV); can deadlock without cv2.setNumThreads(0) in worker_init."
        )
        t2 = time.monotonic()
        self.preload()
        t3 = time.monotonic()
        logger.info(f"DataPrefetcher: first batch ready in {t3 - t2:.2f}s (total init {t3 - t0:.2f}s)")

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        except Exception:
            logger.exception("DataPrefetcher: first next(loader) or CUDA staging failed")
            raise

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
