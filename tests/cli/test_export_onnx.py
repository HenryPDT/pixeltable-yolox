import subprocess
import sys

import pytest
import torch

from yolox.config import YoloxS


@pytest.fixture
def ema_checkpoint(tmp_path):
    config = YoloxS()
    model = config.get_model()
    ema_model = config.get_model()
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            if param.dtype.is_floating_point:
                param.fill_(1.0)
                ema_param.fill_(2.0)

    ckpt_path = tmp_path / "ema_ckpt.pth"
    torch.save(
        {
            "model": ema_model.state_dict(),
            "model_raw": model.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def test_export_onnx_with_checkpoint_fixture(ema_checkpoint):
    onnx_path = ema_checkpoint.with_suffix(".onnx")
    rs = subprocess.run(
        [
            sys.executable,
            "yolox/cli/export_onnx.py",
            "-w",
            str(ema_checkpoint),
            "-cfg",
            "yolox_s",
        ],
        capture_output=True,
        text=True,
    )
    assert rs.returncode == 0, rs.stderr
    assert onnx_path.exists()
