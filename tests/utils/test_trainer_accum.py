import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

from yolox.cli.train import apply_lr_override, convert_args_to_config_opts, make_parser, merge_train_cli_config
from yolox.config import YoloxConfig, YoloxS
from yolox.core.trainer import Trainer
from yolox.utils.confidence_analysis import find_best_confidence_threshold
from yolox.utils.lr_scheduler import LRScheduler


class TestTrainerAccumulation(unittest.TestCase):
    def _make_trainer_stub(self):
        trainer = Trainer.__new__(Trainer)
        trainer.grad_accum_steps = 4
        trainer.clip_max_norm = 0.0
        trainer.use_model_ema = False
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        return trainer

    def test_infer_ema_updates_counts_trailing_flushes(self):
        trainer = self._make_trainer_stub()
        trainer.start_epoch = 3
        trainer.max_iter = 101
        expected = 3 * math.ceil(101 / 4)
        self.assertEqual(trainer._infer_ema_updates(), expected)
        self.assertNotEqual(trainer._infer_ema_updates(), (101 * 3) // 4)

    def test_trailing_gradient_rescale(self):
        model = nn.Linear(2, 2, bias=False)
        for param in model.parameters():
            param.grad = torch.ones_like(param.data)

        trainer = self._make_trainer_stub()
        trainer.model = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        captured = {}

        def capture_step(*_args, **_kwargs):
            captured["grad"] = model.weight.grad.clone()

        trainer.optimizer.step = capture_step
        trainer.optimizer.zero_grad = lambda: None
        trainer._optimizer_step(trailing_micro_steps=2)

        self.assertTrue(torch.allclose(captured["grad"], torch.full_like(captured["grad"], 2.0)))

    def test_optimizer_step_skips_ema_on_scaler_overflow(self):
        model = nn.Linear(2, 2, bias=False)
        for param in model.parameters():
            param.grad = torch.ones_like(param.data)

        trainer = self._make_trainer_stub()
        trainer.model = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer.use_model_ema = True
        trainer.ema_model = MagicMock()

        scaler = MagicMock()
        scaler.is_enabled.return_value = True
        scaler.get_scale.side_effect = [1024.0, 512.0]
        trainer.scaler = scaler

        trainer._optimizer_step()
        trainer.ema_model.update.assert_not_called()

    def test_scheduler_initial_lr_before_first_step(self):
        config = YoloxS()
        config.warmup_epochs = 5
        config.max_epoch = 10
        base_lr = config.get_base_lr(batch_size=16)
        scheduler = LRScheduler(
            config.scheduler,
            base_lr,
            iters_per_epoch=100,
            total_epochs=config.max_epoch,
            warmup_epochs=config.warmup_epochs,
            warmup_lr_start=config.warmup_lr,
            no_aug_epochs=config.no_aug_epochs,
            min_lr_ratio=config.min_lr_ratio,
        )
        init_lr = scheduler.update_lr(1)
        self.assertLess(init_lr, base_lr)

    def test_merge_train_cli_config_with_d_opt_type_and_lr(self):
        config = YoloxS()
        parser = make_parser()
        args = parser.parse_args(["-c", "yolox_s", "--lr", "0.002", "-D", "opt_type=adamw"])
        merge_train_cli_config(config, args)
        self.assertEqual(config.opt_type, "adamw")
        self.assertEqual(config.base_lr, 0.002)

    def test_merge_train_cli_config_d_lr_overrides_cli_lr(self):
        config = YoloxS()
        parser = make_parser()
        args = parser.parse_args(["-c", "yolox_s", "--lr", "0.002", "-D", "base_lr=0.005"])
        merge_train_cli_config(config, args)
        self.assertEqual(config.base_lr, 0.005)

    def test_apply_lr_override_with_cli_optimizer(self):
        config = YoloxS()
        parser = make_parser()
        args = parser.parse_args(["-c", "yolox_s", "--optimizer", "adan", "--lr", "0.003"])
        opts = convert_args_to_config_opts(args)
        config.update(opts)
        apply_lr_override(config, args)
        self.assertEqual(config.base_lr, 0.003)

    def test_confidence_dtignore_excluded(self):
        eval_img = {
            "dtScores": [0.9, 0.8],
            "dtMatches": np.array([[1, 0], [0, 1]]),
            "dtIgnore": np.array([[False, True], [False, False]]),
            "gtIgnore": [False],
        }
        mock_eval = SimpleNamespace(
            evalImgs=[eval_img],
            params=SimpleNamespace(
                imgIds=[1],
                catIds=[1],
                areaRng=[[0, 1e5]],
            ),
            cocoGt=None,
            cocoDt=None,
        )
        result = find_best_confidence_threshold(mock_eval, class_names=["cat"])
        self.assertIsNotNone(result)
        idx = result["thresholds"].index(0.1)
        self.assertEqual(result["precisions"][idx], 1.0)

    def test_config_validation_rejects_zero_resize_interval(self):
        config = YoloxConfig(name="test")
        config.random_resize_interval = 0
        with self.assertRaises(AssertionError):
            config.validate()

    def test_resume_iter_restored_from_checkpoint(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(resume=True, start_epoch=None, ckpt=None)
        trainer.device = "cpu"
        trainer.file_name = tempfile.mkdtemp()
        trainer.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.rank = 0

        model = nn.Linear(2, 2)
        ckpt_path = Path(trainer.file_name) / "latest_ckpt.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "model_raw": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scaler": trainer.scaler.state_dict(),
                "start_epoch": 2,
                "progress": {"epoch": 2, "iter": 7},
            },
            ckpt_path,
        )

        trainer.resume_train(model)
        self.assertEqual(trainer.start_epoch, 2)
        self.assertEqual(trainer._resume_iter, 7)
