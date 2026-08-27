import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from yolox.config import YoloxS
from yolox.core.trainer import Trainer


class TestCheckpointCompat(unittest.TestCase):
    def test_save_ckpt_prefers_ema_for_model_key(self):
        config = YoloxS()
        trainer = Trainer.__new__(Trainer)
        trainer.rank = 0
        trainer.use_model_ema = True
        trainer.exp = config
        trainer.exp.dataset = MagicMock(class_ids=[0, 1], _classes=["a", "b"])
        trainer.epoch = 0
        trainer.iter = 3
        trainer.best_ap = 0.0
        trainer.best_epoch = 0
        trainer._early_stopping_counter = 0
        trainer.decision_metric = "ap50_95"
        trainer.args = MagicMock(logger="tensorboard")
        trainer.file_name = tempfile.mkdtemp()
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)

        student = nn.Linear(2, 2, bias=False)
        ema = nn.Linear(2, 2, bias=False)
        student.weight.data.fill_(1.0)
        ema.weight.data.fill_(9.0)
        trainer.model = student
        trainer.ema_model = MagicMock()
        trainer.ema_model.ema = ema
        trainer.ema_model.updates = 42

        trainer.save_ckpt("unit_test", update_best_ckpt=False, ap=0.1)

        ckpt_path = Path(trainer.file_name) / "unit_test_ckpt.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertTrue(torch.allclose(ckpt["model"]["weight"], torch.full((2, 2), 9.0)))
        self.assertTrue(torch.allclose(ckpt["model_raw"]["weight"], torch.full((2, 2), 1.0)))
        self.assertEqual(ckpt["ema_updates"], 42)
        self.assertEqual(ckpt["progress"], {"epoch": 1, "iter": 0})
        self.assertIn("rng_state", ckpt)
        self.assertEqual(ckpt["rng_state"][0]["rank"], 0)

    def test_resume_prefers_model_raw_over_deployable_model(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = MagicMock(resume=True, start_epoch=None, ckpt=None)
        trainer.device = "cpu"
        trainer.file_name = tempfile.mkdtemp()
        trainer.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.rank = 0

        model = nn.Linear(2, 2, bias=False)
        deployable = nn.Linear(2, 2, bias=False)
        model.weight.data.fill_(1.0)
        deployable.weight.data.fill_(9.0)

        ckpt_path = Path(trainer.file_name) / "latest_ckpt.pth"
        torch.save(
            {
                "model": deployable.state_dict(),
                "model_raw": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scaler": trainer.scaler.state_dict(),
                "start_epoch": 1,
                "progress": {"epoch": 1, "iter": 2},
            },
            ckpt_path,
        )

        loaded = trainer.resume_train(nn.Linear(2, 2, bias=False))
        self.assertTrue(torch.allclose(loaded.weight, torch.full((2, 2), 1.0)))
        self.assertEqual(trainer._resume_iter, 2)

    def test_legacy_checkpoint_resumes_from_model_key(self):
        trainer = Trainer.__new__(Trainer)
        trainer.args = MagicMock(resume=True, start_epoch=None, ckpt=None)
        trainer.device = "cpu"
        trainer.file_name = tempfile.mkdtemp()
        trainer.optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)
        trainer.rank = 0

        model = nn.Linear(2, 2, bias=False)
        model.weight.data.fill_(5.0)
        ckpt_path = Path(trainer.file_name) / "latest_ckpt.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scaler": trainer.scaler.state_dict(),
                "start_epoch": 0,
            },
            ckpt_path,
        )

        loaded = trainer.resume_train(nn.Linear(2, 2, bias=False))
        self.assertTrue(torch.allclose(loaded.weight, torch.full((2, 2), 5.0)))


class TestBestCheckpointAnalysis(unittest.TestCase):
    def _make_trainer(self, tmp, use_ema=True):
        trainer = Trainer.__new__(Trainer)
        trainer.rank = 0
        trainer.is_distributed = False
        trainer.use_model_ema = use_ema
        trainer.device = torch.device("cpu")
        trainer.file_name = tmp
        trainer.best_ap = 0.42
        trainer.best_epoch = 3
        trainer.args = MagicMock(logger="tensorboard")
        student = nn.Linear(2, 2, bias=False)
        ema = nn.Linear(2, 2, bias=False)
        student.weight.data.fill_(1.0)
        ema.weight.data.fill_(0.0)
        trainer.model = student
        trainer.ema_model = MagicMock()
        trainer.ema_model.ema = ema
        trainer.exp = MagicMock()
        trainer.evaluator = MagicMock(nmsthre=0.65)
        trainer.evaluator._last_threshold_result = {
            "thresholds": [0.25, 0.5],
            "best_threshold": 0.5,
            "best_f1": 0.8,
            "precisions": [0.2, 0.7],
            "recalls": [0.9, 0.6],
        }
        return trainer, student, ema

    def _write_best_ckpt(self, tmp, deployable_fill=9.0, raw_fill=1.0, wrapped=True):
        deployable = nn.Linear(2, 2, bias=False)
        raw = nn.Linear(2, 2, bias=False)
        deployable.weight.data.fill_(deployable_fill)
        raw.weight.data.fill_(raw_fill)
        payload = (
            {
                "model": deployable.state_dict(),
                "model_raw": raw.state_dict(),
            }
            if wrapped
            else deployable.state_dict()
        )
        torch.save(payload, Path(tmp) / "best_ckpt.pth")

    def test_missing_best_ckpt_skips_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, _, _ = self._make_trainer(tmp)
            trainer.after_train()
            trainer.exp.eval.assert_not_called()

    def test_after_train_loads_deployable_model_not_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, student, ema = self._make_trainer(tmp)
            self._write_best_ckpt(tmp)
            trainer.after_train()

            trainer.exp.eval.assert_called_once()
            eval_model, evaluator, distributed = trainer.exp.eval.call_args.args[:3]
            self.assertIs(eval_model, ema)
            self.assertIs(evaluator, trainer.evaluator)
            self.assertFalse(distributed)
            self.assertTrue(torch.allclose(ema.weight, torch.full((2, 2), 9.0)))
            self.assertTrue(torch.allclose(student.weight, torch.full((2, 2), 1.0)))

    def test_after_train_without_ema_loads_unwrapped_student(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, student, _ = self._make_trainer(tmp, use_ema=False)
            self._write_best_ckpt(tmp)
            trainer.after_train()

            eval_model = trainer.exp.eval.call_args.args[0]
            self.assertIs(eval_model, student)
            self.assertTrue(torch.allclose(student.weight, torch.full((2, 2), 9.0)))

    def test_legacy_unwrapped_checkpoint_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, _, ema = self._make_trainer(tmp)
            self._write_best_ckpt(tmp, wrapped=False)
            trainer.after_train()
            self.assertTrue(torch.allclose(ema.weight, torch.full((2, 2), 9.0)))
            trainer.exp.eval.assert_called_once()

    def test_eval_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, _, _ = self._make_trainer(tmp)
            self._write_best_ckpt(tmp)
            trainer.exp.eval.side_effect = RuntimeError("eval boom")
            trainer.after_train()

    def test_distributed_non_zero_rank_still_runs_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer, _, ema = self._make_trainer(tmp)
            trainer.is_distributed = True
            trainer.rank = 1
            self._write_best_ckpt(tmp)

            def fake_broadcast(tensor, src=0):
                if tensor.numel() == 1:
                    tensor.fill_(1.0)

            with patch("torch.distributed.broadcast", side_effect=fake_broadcast):
                trainer.after_train()

            trainer.exp.eval.assert_called_once()
            self.assertTrue(trainer.exp.eval.call_args.args[2])
            self.assertIs(trainer.exp.eval.call_args.args[0], ema)
