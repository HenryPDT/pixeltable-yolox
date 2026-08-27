import os
import subprocess
from pathlib import Path

import pytest
import torch


def _coco_data_dir() -> Path | None:
    """Return a COCO root that contains train annotations, if one exists."""
    candidates = []
    for env_name in ("YOLOX_SMOKE_DATA_DIR", "YOLOX_DATADIR"):
        env_dir = os.getenv(env_name)
        if env_dir:
            path = Path(env_dir)
            candidates.append(path)
            candidates.append(path / "COCO")
    candidates.append(Path("datasets/COCO"))

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "annotations" / "instances_train2017.json").is_file():
            return resolved
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class TestTraining:
    def test_training_checkpoint_contains_expected_keys(self) -> None:
        data_dir = _coco_data_dir()
        if data_dir is None:
            pytest.skip(
                "COCO annotations not found. Symlink datasets/COCO, set "
                "YOLOX_DATADIR, or point YOLOX_SMOKE_DATA_DIR at a COCO-style root."
            )

        output_name = "gpu_smoke_test"
        subprocess.run(
            [
                "yolox",
                "train",
                "-c",
                "yolox-s",
                "-d",
                "1",
                "-b",
                "8",
                "--fp16",
                "-o",
                "--output-dir",
                output_name,
                "--data-dir",
                str(data_dir),
                "-D",
                "max_epoch=2",
                "-D",
                "no_aug_epochs=0",
                "-D",
                "seed=4171780",
                "-D",
                "deterministic=True",
                "-D",
                "data_num_workers=1",
            ],
            check=True,
        )

        candidates = sorted(Path("out/train").glob(f"*/{output_name}/latest_ckpt.pth"))
        if not candidates:
            candidates = sorted(Path("out/train").glob(f"*/{output_name}_*/latest_ckpt.pth"))
        assert candidates, "Training output checkpoint not found"

        ckpt = torch.load(candidates[-1], map_location="cpu", weights_only=False)
        assert "model" in ckpt
        assert "model_raw" in ckpt
        assert "optimizer" in ckpt
        assert "rng_state" in ckpt
        assert "progress" in ckpt
        assert "meta" in ckpt
        assert ckpt["start_epoch"] == 2
        assert ckpt["meta"]["num_classes"] > 0
