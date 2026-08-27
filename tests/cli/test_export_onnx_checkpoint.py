import tempfile
import unittest
from pathlib import Path

import torch

from yolox.cli.export_onnx import yolox_export
from yolox.config import YoloxS


class TestExportOnnxCheckpoint(unittest.TestCase):
    def test_export_uses_deployable_model_key(self):
        config = YoloxS()
        config.num_classes = 80
        model = config.get_model()
        ema_model = config.get_model()
        with torch.no_grad():
            for (name, param), (_, ema_param) in zip(
                model.state_dict().items(), ema_model.state_dict().items()
            ):
                if param.dtype.is_floating_point:
                    param.fill_(1.0)
                    ema_param.fill_(2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ema_ckpt.pth"
            torch.save(
                {
                    "model": ema_model.state_dict(),
                    "model_raw": model.state_dict(),
                },
                ckpt_path,
            )
            exported, _ = yolox_export(str(ckpt_path), "yolox_s")
            weight = next(p for p in exported.parameters() if p.requires_grad or p.dim() > 0)
            self.assertTrue(torch.any(weight != 1.0))
