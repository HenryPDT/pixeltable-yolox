import unittest

import numpy as np

from yolox.data.data_augment import AlbumentationsTransform, TrainTransform


class TestEmptyBoxAugment(unittest.TestCase):
    def test_train_transform_applies_albumentations_for_empty_targets(self):
        albu = AlbumentationsTransform(hsv_prob=1.0, hflip_prob=0.0)
        transform = TrainTransform(max_labels=10, albu_transform=albu)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        original = image.copy()
        targets = np.zeros((0, 5), dtype=np.float32)

        out_image, out_targets = transform(image, targets, (64, 64))

        self.assertEqual(out_targets.shape, (10, 5))
        self.assertFalse(np.array_equal(out_image, original))

    def test_albumentations_empty_boxes_returns_image_only(self):
        albu = AlbumentationsTransform(hsv_prob=1.0, hflip_prob=0.0)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        original = image.copy()
        image, boxes, labels = albu(image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(labels.shape, (0,))
        self.assertFalse(np.array_equal(image, original))
