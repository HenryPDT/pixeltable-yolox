import unittest
from unittest.mock import patch

import torch

from yolox.config import YoloxS
from yolox.utils.auto_batch import _create_probe_model, auto_batch_size


class TestAutoBatchIsolation(unittest.TestCase):
    def test_probe_model_does_not_touch_cached_training_model(self):
        config = YoloxS()
        training_model = config.get_model()
        with torch.no_grad():
            for buffer in training_model.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.fill_(3.14)

        probe_model = _create_probe_model(config)
        self.assertIsNot(probe_model, training_model)
        self.assertTrue(hasattr(config, "model"))

        with torch.no_grad():
            for buffer in probe_model.buffers():
                if buffer.dtype.is_floating_point:
                    buffer.fill_(0.0)

        for before, after in zip(training_model.buffers(), config.model.buffers()):
            if before.dtype.is_floating_point:
                self.assertTrue(torch.allclose(before, after))

    @patch("yolox.utils.auto_batch.dist.get_rank", return_value=0)
    @patch("yolox.utils.auto_batch.dist.get_world_size", return_value=2)
    @patch("yolox.utils.auto_batch.dist.is_initialized", return_value=True)
    @patch("yolox.utils.auto_batch.dist.broadcast")
    @patch("yolox.utils.auto_batch._probe_batch_size", return_value=3)
    def test_auto_batch_broadcasts_global_size(self, mock_probe, mock_broadcast, *_args):
        config = YoloxS()

        class FakeTensor:
            def __init__(self, data):
                self._data = list(data)

            def __getitem__(self, idx):
                value = self._data[idx]
                return FakeTensor([value])

            def item(self):
                return self._data[-1]

        def fake_tensor(data, dtype=None, device=None):
            return FakeTensor(data)

        with patch("torch.cuda.is_available", return_value=True), patch(
            "yolox.utils.auto_batch.torch.tensor", side_effect=fake_tensor
        ):
            selected = auto_batch_size(config, device="cuda:0", is_distributed=True)
        self.assertEqual(selected, 6)
        mock_broadcast.assert_called_once()
        mock_probe.assert_called_once()

    @patch("yolox.utils.auto_batch._probe_batch_size", side_effect=RuntimeError("boom"))
    @patch("yolox.utils.auto_batch.dist.is_initialized", return_value=False)
    def test_probe_failure_propagates_when_not_recoverable(self, *_args):
        config = YoloxS()
        with patch("torch.cuda.is_available", return_value=True):
            with self.assertRaises(RuntimeError):
                auto_batch_size(config, device="cuda:0", is_distributed=False)

    @patch("yolox.utils.auto_batch._probe_batch_size", side_effect=RuntimeError("Could not load a valid sample"))
    @patch("yolox.utils.auto_batch.dist.is_initialized", return_value=False)
    def test_probe_dataset_failure_falls_back(self, *_args):
        config = YoloxS()
        with patch("torch.cuda.is_available", return_value=True):
            selected = auto_batch_size(config, device="cuda:0", is_distributed=False)
        self.assertEqual(selected, 4)
