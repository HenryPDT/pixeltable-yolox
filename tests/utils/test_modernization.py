import tempfile
import unittest
from pathlib import Path
import torch
import torch.nn as nn

from yolox.config import YoloxS, YoloxConfig
from yolox.utils.optimizers import build_yolox_optimizer, Adan
from yolox.utils.auto_batch import auto_batch_size
from yolox.utils.confidence_analysis import find_best_confidence_threshold


class TestModernizationFeatures(unittest.TestCase):
    def test_optimizer_parameter_groups(self):
        """Verify that build_yolox_optimizer correctly separates backbone and head parameter groups."""
        config = YoloxS()
        model = config.get_model()

        # Test SGD with decoupled learning rate
        opt = build_yolox_optimizer(
            model,
            optimizer_type="sgd",
            base_lr=0.01,
            backbone_lr_ratio=0.2,
            momentum=0.9,
            weight_decay=5e-4,
        )

        assert len(opt.param_groups) == 4
        group_names = [g["name"] for g in opt.param_groups]
        assert "backbone_weights" in group_names
        assert "backbone_no_decay" in group_names
        assert "head_weights" in group_names
        assert "head_no_decay" in group_names

        # Check learning rates and weight decays
        for g in opt.param_groups:
            if "backbone" in g["name"]:
                assert g["lr_multiplier"] == 0.2
                assert abs(g["lr"] - 0.002) < 1e-6
            else:
                assert g["lr_multiplier"] == 1.0
                assert abs(g["lr"] - 0.01) < 1e-6

            if "no_decay" in g["name"]:
                assert g["weight_decay"] == 0.0
            else:
                assert g["weight_decay"] == 5e-4

    def test_adamw_and_adan_optimizers(self):
        """Verify that AdamW and Adan optimizers build and step correctly."""
        config = YoloxS()
        model = config.get_model()

        # AdamW test
        opt_adamw = build_yolox_optimizer(model, optimizer_type="adamw", base_lr=1e-3, backbone_lr_ratio=0.5)
        assert isinstance(opt_adamw, torch.optim.AdamW)

        # Adan test
        opt_adan = build_yolox_optimizer(model, optimizer_type="adan", base_lr=1e-3, backbone_lr_ratio=0.5)
        assert isinstance(opt_adan, Adan)

        # Run a dummy step with Adan
        x = torch.randn(1, 3, 64, 64)
        targets = torch.zeros(1, 10, 5)
        model.train()
        outputs = model(x, targets)
        loss = outputs["total_loss"]
        loss.backward()
        opt_adan.step()
        opt_adan.zero_grad()

    def test_confidence_analysis_fallback(self):
        """Verify find_best_confidence_threshold handles None evalImgs with mock coco_eval."""
        class MockFastCocoEval:
            def __init__(self):
                self.evalImgs = None
                self.params = type("Params", (), {
                    "imgIds": [1, 2],
                    "catIds": [1],
                    "iouThrs": [0.5],
                    "maxDets": [100],
                })()

        mock_eval = MockFastCocoEval()
        # When evalImgs is None and no GT/DT is passed, it logs warning and returns None gracefully
        res = find_best_confidence_threshold(mock_eval, class_names=["cat"])
        assert res is None

    def test_checkpoint_meta_handling(self):
        """Verify that checkpoint metadata is properly constructed and readable."""
        config = YoloxS()
        config.num_classes = 5

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pth"
            model = config.get_model()

            meta = {
                "model_name": "yolox_s",
                "num_classes": 5,
                "class_names": ["c1", "c2", "c3", "c4", "c5"],
                "input_size": [640, 640],
                "depth": 0.33,
                "width": 0.50,
                "act": "silu",
                "depthwise": False,
                "decision_metric": "ap50_95",
            }
            ckpt_data = {
                "model": model.state_dict(),
                "meta": meta,
            }
            torch.save(ckpt_data, ckpt_path)

            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert "meta" in loaded
            assert loaded["meta"]["num_classes"] == 5
            assert loaded["meta"]["model_name"] == "yolox_s"
            assert loaded["meta"]["class_names"] == ["c1", "c2", "c3", "c4", "c5"]

    def test_config_options_override(self):
        """Verify that YoloxConfig correctly updates new modernization parameters."""
        config = YoloxS()
        config.update({
            "opt_type": "adamw",
            "backbone_lr_ratio": "0.25",
            "amp_dtype": "bfloat16",
            "decision_metric": "ap50",
            "base_lr": "0.001",
        })

        assert config.opt_type == "adamw"
        assert config.backbone_lr_ratio == 0.25
        assert config.amp_dtype == "bfloat16"
        assert config.decision_metric == "ap50"
        assert config.base_lr == 0.001

        # Test optimizer built from updated config
        opt = config.get_optimizer(batch_size=16)
        assert isinstance(opt, torch.optim.AdamW)
        for g in opt.param_groups:
            if "backbone" in g["name"]:
                assert abs(g["lr"] - 0.00025) < 1e-6
            else:
                assert abs(g["lr"] - 0.001) < 1e-6

    def test_lr_multiplier_trainer_logic(self):
        """Simulate Trainer LR update loop to verify decoupled group scaling."""
        config = YoloxS()
        config.backbone_lr_ratio = 0.2
        config.base_lr = 0.01
        opt = config.get_optimizer(batch_size=16)

        # Simulate LR scheduler outputting a decayed learning rate
        decayed_base_lr = 0.005
        for param_group in opt.param_groups:
            param_group["lr"] = decayed_base_lr * param_group.get("lr_multiplier", 1.0)

        for g in opt.param_groups:
            if "backbone" in g["name"]:
                # Backbone should be 0.005 * 0.2 = 0.001
                assert abs(g["lr"] - 0.001) < 1e-6
            else:
                # Head should be 0.005 * 1.0 = 0.005
                assert abs(g["lr"] - 0.005) < 1e-6

    def test_cli_lr_mapping_for_adamw(self):
        """Verify that CLI --lr maps to base_lr for adamw/adan and basic_lr_per_img for sgd."""
        from yolox.cli.train import make_parser, convert_args_to_config_opts

        parser = make_parser()
        args_adamw = parser.parse_args(["-c", "yolox_s", "--optimizer", "adamw", "--lr", "0.001", "--backbone-lr-ratio", "0.2"])
        opts = convert_args_to_config_opts(args_adamw)
        assert opts["base_lr"] == "0.001"
        assert "basic_lr_per_img" not in opts

        args_sgd = parser.parse_args(["-c", "yolox_s", "--optimizer", "sgd", "--lr", "0.00015625"])
        opts_sgd = convert_args_to_config_opts(args_sgd)
        assert opts_sgd["basic_lr_per_img"] == "0.00015625"
        assert "base_lr" not in opts_sgd

