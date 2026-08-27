import datetime
import math
import os
import random
import time
from contextlib import nullcontext
from loguru import logger

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.config import YoloxConfig
from yolox.data import DataPrefetcher
from yolox.data.datasets import ConcatDataset, MixConcatDataset
from yolox.utils import (
    MeterBuffer,
    MlflowLogger,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    average_gradients,
    analyze_dataset_stats,
    auto_batch_size,
    get_deployable_state_dict,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    log_dataset_stats,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, config: YoloxConfig, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = config
        self.args = args

        # training related attr
        self.max_epoch = config.max_epoch
        amp_dtype_str = getattr(args, "amp_dtype", None) or getattr(config, "amp_dtype", "float16")
        explicit_amp_dtype = getattr(args, "amp_dtype", None) is not None
        self.amp_training = bool(getattr(args, "fp16", False) or explicit_amp_dtype)
        if amp_dtype_str == "bfloat16":
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                logger.warning(
                    "bfloat16 not supported on this GPU architecture, falling back to float16 AMP"
                )
                self.amp_dtype = "float16"
                self.amp_training = True
            else:
                self.amp_dtype = "bfloat16"
                self.amp_training = True
        else:
            self.amp_dtype = "float16"
            if explicit_amp_dtype:
                self.amp_training = True
        # bfloat16 has full FP32 dynamic range and does not require loss scaling
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.amp_training and self.amp_dtype == "float16"))
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = config.ema
        self.save_history_ckpt = config.save_history_ckpt
        self.grad_accum_steps = max(config.grad_accum_steps, 1)
        self.clip_max_norm = config.clip_max_norm
        self.early_stopping_patience = config.early_stopping_patience
        self.decision_metric = getattr(config, "decision_metric", "ap50_95")
        self._early_stopping_counter = 0
        self._stop_training = False
        self._accum_micro_steps = 0
        self._resume_ema_state = None
        self._resume_ema_updates = None
        self._resume_iter = 0
        self._last_decision_ap = None

        # data/dataloader related attr
        self.data_type = torch.float16 if (self.amp_training and self.amp_dtype == "float16") else torch.float32
        self.input_size = config.input_size
        self.best_ap = 0
        self.best_epoch = 0

        # metric record
        self.meter = MeterBuffer(window_size=config.print_interval)
        self.file_name = os.path.join(config.output_dir, args.name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error("Exception in training: ", e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            if self._stop_training:
                break
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        start_iter = 0
        if self.epoch == self.start_epoch and self._resume_iter > 0:
            start_iter = self._resume_iter
            logger.info(
                f"Resuming epoch {self.epoch + 1} at iter {start_iter + 1}/{self.max_iter}"
            )
            for _ in range(start_iter):
                self.prefetcher.next()
            self._resume_iter = 0

        for self.iter in range(start_iter, self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        base_lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = base_lr * param_group.get("lr_multiplier", 1.0)

        is_accum_boundary = (self.iter + 1) % self.grad_accum_steps == 0
        use_no_sync = (
            self.is_distributed
            and self.grad_accum_steps > 1
            and not is_accum_boundary
        )
        sync_cm = self.model.no_sync() if use_no_sync else nullcontext()

        torch_amp_dtype = torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16
        with sync_cm:
            with torch.amp.autocast('cuda', enabled=self.amp_training, dtype=torch_amp_dtype):
                outputs = self.model(inps, targets)

            loss = outputs["total_loss"]
            if self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

        self._accum_micro_steps += 1

        if is_accum_boundary:
            self._optimizer_step()
            self._accum_micro_steps = 0

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=base_lr,
            **outputs,
        )

    def _optimizer_step(self, trailing_micro_steps=None):
        """Run gradient clipping, optimizer step, and EMA update on accumulation boundaries."""
        if trailing_micro_steps is not None and trailing_micro_steps < self.grad_accum_steps:
            scale = self.grad_accum_steps / trailing_micro_steps
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

        if self.clip_max_norm > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_max_norm)

        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        optimizer_stepped = (not self.scaler.is_enabled()) or (
            self.scaler.get_scale() >= scale_before
        )

        self.optimizer.zero_grad()

        if optimizer_stepped and self.use_model_ema:
            self.ema_model.update(self.model)

    def _sync_ema_model(self):
        """Broadcast rank-0 EMA weights/buffers so distributed eval uses consistent state."""
        if not self.use_model_ema or not self.is_distributed:
            return
        self._broadcast_module(self.ema_model.ema)

    def _broadcast_module(self, module):
        """Broadcast parameters and buffers from rank 0. No-op when not distributed."""
        if not self.is_distributed:
            return
        import torch.distributed as dist

        for param in module.parameters():
            dist.broadcast(param.data, src=0)
        for buffer in module.buffers():
            dist.broadcast(buffer.data, src=0)

    def _get_eval_model(self):
        """Return the unwrapped deployable model used for evaluation."""
        if self.use_model_ema:
            return self.ema_model.ema
        evalmodel = self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module
        return evalmodel

    def _infer_ema_updates(self):
        """Legacy fallback when checkpoints omit the exact EMA update counter."""
        if self.max_iter <= 0:
            return 0
        steps_per_epoch = math.ceil(self.max_iter / self.grad_accum_steps)
        return self.start_epoch * steps_per_epoch

    def before_train(self):
        logger.info("args: {}".format(self.args))

        # Get dataset and num_classes before initializing model
        if self.exp.dataset is None:
            self.exp.dataset = self.exp.get_dataset(cache=bool(self.args.cache), cache_type=self.args.cache)

        base_dataset = self.exp.dataset
        if isinstance(base_dataset, (ConcatDataset, MixConcatDataset)):
            base_dataset = base_dataset.datasets[0]

        if hasattr(base_dataset, "class_ids"):
            num_classes = len(base_dataset.class_ids)
        elif hasattr(base_dataset, "_classes"):
            num_classes = len(base_dataset._classes)
        else:
            raise ValueError("Cannot determine number of classes from dataset.")

        # Always use the number of classes from the dataset
        if self.exp.num_classes != num_classes:
            logger.info(f"Setting num_classes to {num_classes} from the dataset (previously {self.exp.num_classes}).")
            self.exp.num_classes = num_classes
            # reset model cache in exp if it was created with wrong num_classes
            if hasattr(self.exp, 'model'):
                delattr(self.exp, 'model')

        logger.info("exp value:\n{}".format(self.exp))

        if self.exp.seed is not None:
            seed = self.exp.seed + self.rank if self.is_distributed else self.exp.seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            if self.exp.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.info(f"Set random seed to {seed} (deterministic={self.exp.deterministic})")

        # Auto batch size probing if requested
        if self.args.batch_size == -1:
            self.args.batch_size = auto_batch_size(
                self.exp,
                device=self.device,
                is_distributed=self.is_distributed,
                amp_enabled=self.amp_training,
                amp_dtype=self.amp_dtype,
            )

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        self.optimizer.zero_grad()  # Initialize gradients to zero for accumulation

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        logger.info(
            f"Train dataloader ready: max_iter={len(self.train_loader)} "
            f"(num_workers={self.exp.data_num_workers})."
        )
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        base_lr = (
            self.exp.get_base_lr(self.args.batch_size)
            if hasattr(self.exp, "get_base_lr")
            else (self.exp.base_lr if self.exp.base_lr is not None else self.exp.basic_lr_per_img * self.args.batch_size)
        )
        self.lr_scheduler = self.exp.get_lr_scheduler(base_lr, self.max_iter)
        init_lr = self.lr_scheduler.update_lr(1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = init_lr * param_group.get("lr_multiplier", 1.0)
        logger.info("LR scheduler initialized.")
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            if self.args.batch_size < 4 and torch.cuda.is_available():
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                logger.info("Converted BatchNorm to SyncBatchNorm for small batch DDP (<4)")
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            if getattr(self, "_resume_ema_state", None) is not None:
                self.ema_model.ema.load_state_dict(self._resume_ema_state)
                self.ema_model.updates = (
                    self._resume_ema_updates
                    if self._resume_ema_updates is not None
                    else self._infer_ema_updates()
                )
            else:
                self.ema_model.updates = self._infer_ema_updates()

        self.model = model
        self._accum_micro_steps = 0

        logger.info("Building validation evaluator / dataloader...")
        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed, save_dir=self.file_name
        )
        logger.info("Evaluator ready.")

        # Log dataset statistics only on main process
        if self.rank == 0:
            # Analyze and log training dataset statistics
            train_dataset = self.train_loader.dataset
            logger.info("Computing training dataset statistics (sampled)...")
            train_stats = analyze_dataset_stats(train_dataset, "Training Dataset")
            log_dataset_stats(train_stats, "Training Dataset")

            # Analyze and log validation dataset statistics
            val_dataset = self.evaluator.dataloader.dataset
            logger.info("Computing validation dataset statistics (sampled)...")
            val_stats = analyze_dataset_stats(val_dataset, "Validation Dataset")
            log_dataset_stats(val_stats, "Validation Dataset")
        
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("logger must be either 'tensorboard', 'mlflow' or 'wandb'")

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

        if self.grad_accum_steps > 1:
            logger.info(
                f"Using gradient accumulation: {self.grad_accum_steps} steps "
                f"(effective batch size = batch_size * {self.grad_accum_steps})"
            )
        if self.clip_max_norm > 0:
            logger.info(f"Using gradient clipping with max_norm={self.clip_max_norm}")
        if self.early_stopping_patience > 0:
            logger.info(f"Early stopping enabled with patience={self.early_stopping_patience} epochs")

    def after_train(self):
        if self.best_epoch > 0:
            logger.info(
                f"Training of experiment is done and the best AP is {self.best_ap * 100:.2f} (mAP: {self.best_ap:.4f}) achieved at epoch {self.best_epoch}"
            )
        else:
            logger.info(
                "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
            )

        self._analyze_best_checkpoint()

        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
            elif self.args.logger == "mlflow":
                metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
                self.mlflow_logger.on_train_end(self.args, file_name=self.file_name,
                                                metadata=metadata)

    def _analyze_best_checkpoint(self):
        """Re-evaluate deployable `best_ckpt.pth` weights on every rank.

        Rank-0-only eval would hang under DDP (`gather`/`synchronize`) and would
        only see that rank's val shard. Load `model` (EMA/deployable), not
        `model_raw`.
        """
        best_ckpt_path = os.path.join(self.file_name, "best_ckpt.pth")
        should_run = torch.zeros(1, device=self.device)
        if self.rank == 0 and os.path.isfile(best_ckpt_path):
            should_run.fill_(1.0)
        if self.is_distributed:
            import torch.distributed as dist

            dist.broadcast(should_run, src=0)
        if should_run.item() < 1.0:
            return

        evalmodel = self._get_eval_model()
        load_ok = torch.ones(1, device=self.device)
        if self.rank == 0:
            logger.info(
                f"Running final threshold analysis on BEST checkpoint "
                f"(Epoch {self.best_epoch})..."
            )
            try:
                ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
                load_ckpt(evalmodel, get_deployable_state_dict(ckpt))
            except Exception as e:
                logger.warning(f"Failed to load best checkpoint for final analysis: {e}")
                load_ok.zero_()
        if self.is_distributed:
            import torch.distributed as dist

            dist.broadcast(load_ok, src=0)
        if load_ok.item() < 1.0:
            synchronize()
            return

        self._broadcast_module(evalmodel)
        try:
            with adjust_status(evalmodel, training=False):
                self.exp.eval(
                    evalmodel, self.evaluator, self.is_distributed, return_outputs=False
                )
            if self.rank == 0:
                self._log_deployment_summary()
        except Exception as e:
            if self.rank == 0:
                logger.warning(f"Failed to run final analysis on best checkpoint: {e}")
        synchronize()

    def _log_deployment_summary(self):
        tr = getattr(self.evaluator, "_last_threshold_result", None)
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE — DEPLOYMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(
            f"  Best AP:    {self.best_ap * 100:.2f}%  (mAP50:95)  at Epoch {self.best_epoch}"
        )
        if tr is not None:
            prec = rec = None
            try:
                best_idx = tr["thresholds"].index(tr["best_threshold"])
                prec = tr["precisions"][best_idx]
                rec = tr["recalls"][best_idx]
            except (KeyError, ValueError, IndexError, TypeError):
                pass
            if prec is not None and rec is not None:
                logger.info(
                    f"  Confidence: {tr['best_threshold']:.2f}  "
                    f"(F1={tr['best_f1']:.3f}, Precision={prec:.3f}, Recall={rec:.3f})"
                )
            else:
                logger.info(
                    f"  Confidence: {tr['best_threshold']:.2f}  (F1={tr['best_f1']:.3f})"
                )
        logger.info(f"  NMS IoU:    {self.evaluator.nmsthre:.2f}  (used during training)")
        logger.info("=" * 60)

    def before_epoch(self):
        self._accum_micro_steps = 0
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def _force_grad_sync(self):
        """Manually average param grads across DDP ranks for trailing accumulation windows."""
        average_gradients(self.model)

    def after_epoch(self):
        # Flush leftover accumulated gradients at the end of the epoch
        if self.grad_accum_steps > 1 and self._accum_micro_steps > 0:
            if self.is_distributed:
                self._force_grad_sync()
            self._optimizer_step(trailing_micro_steps=self._accum_micro_steps)
            self._accum_micro_steps = 0

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            if self.use_model_ema:
                self._sync_ema_model()
            self.evaluate_and_save_model()

        self.save_ckpt(ckpt_name="latest", ap=self._last_decision_ap)

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {}x{}, {}".format(self.input_size[0], self.input_size[1], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
                if self.args.logger == 'mlflow':
                    logs = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    logs.update({"train/lr": self.meter["lr"].latest})
                    self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)

            self.meter.clear_meters()

        # random resizing
        if not self.exp.deterministic:
            resize_interval = getattr(self.exp, "random_resize_interval", 10)
            if resize_interval > 0 and (self.progress_in_iter + 1) % resize_interval == 0:
                self.input_size = self.exp.random_resize(
                    self.train_loader, self.epoch, self.rank, self.is_distributed
                )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            student_state = ckpt.get("model_raw", ckpt["model"])
            model.load_state_dict(student_state)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])
            self.best_ap = ckpt.pop("best_ap", 0)
            self.best_epoch = ckpt.pop("best_epoch", 0)
            self._early_stopping_counter = ckpt.pop("early_stopping_counter", 0)
            progress = ckpt.get("progress", {})
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            self._resume_ema_state = ckpt.get("ema")
            self._resume_ema_updates = ckpt.get("ema_updates")
            progress_epoch = int(progress.get("epoch", start_epoch))
            self._resume_iter = 0
            if self.args.start_epoch is None and progress_epoch == start_epoch:
                self._resume_iter = int(progress.get("iter", 0))
            elif progress and self.args.start_epoch is None:
                logger.warning(
                    "Ignoring checkpoint progress.iter because progress.epoch "
                    f"({progress_epoch}) does not match start_epoch ({start_epoch})."
                )
            self._restore_rng_state(ckpt.get("rng_state"))
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    model = load_ckpt(model, ckpt["model"])
                else:
                    model = load_ckpt(model, ckpt)
            self.start_epoch = 0
            self._resume_ema_state = None
            self._resume_ema_updates = None
            self._resume_iter = 0

        return model

    def evaluate_and_save_model(self):
        evalmodel = self._get_eval_model()

        with adjust_status(evalmodel, training=False):
            eval_outputs = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        ap50_95, ap50, summary, predictions = 0.0, 0.0, "", {}
        decision_score = 0.0
        update_best_ckpt = False

        if self.rank == 0:
            (ap50_95, ap50, summary), predictions = eval_outputs
            decision_score = ap50 if self.decision_metric == "ap50" else ap50_95
            update_best_ckpt = decision_score > self.best_ap
            if update_best_ckpt:
                self.best_ap = decision_score
                self.best_epoch = self.epoch + 1

            if self.early_stopping_patience > 0:
                if update_best_ckpt:
                    self._early_stopping_counter = 0
                else:
                    self._early_stopping_counter += 1
                    logger.info(
                        f"Early stopping counter: {self._early_stopping_counter}"
                        f"/{self.early_stopping_patience}"
                    )
                    if self._early_stopping_counter >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {self._early_stopping_counter} "
                            f"epochs without improvement. Best metric: {self.best_ap:.4f}"
                        )
                        self._stop_training = True

        if self.is_distributed:
            import torch.distributed as dist

            state = torch.tensor(
                [
                    float(ap50_95),
                    float(ap50),
                    float(decision_score),
                    float(update_best_ckpt),
                    float(self._early_stopping_counter),
                    float(self._stop_training),
                    float(self.best_ap),
                    float(self.best_epoch),
                ],
                device=self.device,
            )
            dist.broadcast(state, src=0)
            if self.rank != 0:
                ap50_95 = state[0].item()
                ap50 = state[1].item()
                decision_score = state[2].item()
                update_best_ckpt = bool(state[3].item())
                self._early_stopping_counter = int(state[4].item())
                self._stop_training = bool(state[5].item())
                self.best_ap = state[6].item()
                self.best_epoch = int(state[7].item())

        self._last_decision_ap = decision_score

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            if self.args.logger == "mlflow":
                logs = {
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "val/best_ap": round(self.best_ap, 3),
                    "train/epoch": self.epoch + 1,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)
            if summary:
                logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=decision_score)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=decision_score)

        if self.args.logger == "mlflow" and self.rank == 0:
            metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
            self.mlflow_logger.save_checkpoints(self.args, self.exp, self.file_name, self.epoch,
                                                metadata, update_best_ckpt)

    def _capture_rng_state(self):
        state = {
            "rank": self.rank,
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, rng_state):
        if not rng_state:
            return
        if isinstance(rng_state, list):
            if self.rank >= len(rng_state):
                logger.warning(
                    f"No RNG state for rank {self.rank}; checkpoint has {len(rng_state)} ranks."
                )
                return
            rng_state = rng_state[self.rank]
        if rng_state.get("rank", self.rank) != self.rank:
            logger.warning(
                f"Checkpoint RNG state was saved on rank {rng_state.get('rank')}; "
                f"restoring on rank {self.rank} for best-effort resume."
            )
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "torch_cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        rng_states = [self._capture_rng_state()]
        if getattr(self, "is_distributed", False):
            import torch.distributed as dist

            rng_states = [None] * get_world_size()
            dist.all_gather_object(rng_states, self._capture_rng_state())

        if self.rank == 0:
            student_model = self.model.module if is_parallel(self.model) else self.model
            deployable_model = (
                self.ema_model.ema if self.use_model_ema else student_model
            )
            logger.info("Save weights to {}".format(self.file_name))

            base_dataset = self.exp.dataset
            if isinstance(base_dataset, (ConcatDataset, MixConcatDataset)):
                base_dataset = base_dataset.datasets[0]
            class_ids = getattr(base_dataset, "class_ids", None)
            class_names = getattr(base_dataset, "_classes", None) or class_ids

            meta = {
                "model_name": self.exp.name,
                "num_classes": self.exp.num_classes,
                "class_names": list(class_names) if class_names is not None else None,
                "class_ids": list(class_ids) if class_ids is not None else None,
                "input_size": list(self.exp.input_size),
                "depth": self.exp.depth,
                "width": self.exp.width,
                "act": self.exp.act,
                "depthwise": self.exp.depthwise,
                "decision_metric": self.decision_metric,
            }

            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": deployable_model.state_dict(),
                "model_raw": student_model.state_dict(),
                "ema": self.ema_model.ema.state_dict() if self.use_model_ema else None,
                "ema_updates": self.ema_model.updates if self.use_model_ema else 0,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_ap": self.best_ap,
                "best_epoch": self.best_epoch,
                "early_stopping_counter": self._early_stopping_counter,
                "curr_ap": ap,
                "rng_state": rng_states,
                "progress": {
                    # Checkpoints are written at epoch boundaries; resume at the
                    # first micro-batch of the next epoch.
                    "epoch": self.epoch + 1,
                    "iter": 0,
                },
                "meta": meta,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )