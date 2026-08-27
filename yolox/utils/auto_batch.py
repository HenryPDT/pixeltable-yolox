# Copyright (c) Megvii, Inc. and its affiliates.
# Auto batch size search utility inspired by D-FINE-seg.

from __future__ import annotations

import copy

import torch
import torch.distributed as dist
from loguru import logger


def auto_batch_size(
    config,
    device: torch.device | str = "cuda:0",
    is_distributed: bool = False,
    target_fraction: float = 0.7,
    amp_enabled: bool = True,
    amp_dtype: str = "float16",
) -> int:
    """Probe the GPU to find the largest global batch size that fits in VRAM.

    Returns the total batch size across all devices. ``get_data_loader()`` divides
    this by world size to obtain the per-rank micro-batch.
    """
    device = torch.device(device) if isinstance(device, str) else device
    rank = dist.get_rank() if (is_distributed and dist.is_initialized()) else 0
    world_size = dist.get_world_size() if (is_distributed and dist.is_initialized()) else 1
    selected_bs = 4
    probe_ok = True

    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("Auto batch size only works on CUDA devices, defaulting to batch_size=4")
        selected_bs = 4 * world_size
    elif rank == 0:
        try:
            selected_bs = _probe_batch_size(
                config,
                device,
                target_fraction=target_fraction,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            selected_bs = max(1, selected_bs) * world_size
            logger.info(f"Optimal global batch size selected: {selected_bs}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"Auto batch size probing failed with OOM: {e}, defaulting to batch_size=4"
                )
                selected_bs = 4 * world_size
            elif "Could not load a valid sample" in str(e):
                logger.warning(
                    f"Auto batch size probing failed: {e}, defaulting to batch_size=4"
                )
                selected_bs = 4 * world_size
            else:
                probe_ok = False
        except OSError as e:
            logger.warning(f"Auto batch size probing failed: {e}, defaulting to batch_size=4")
            selected_bs = 4 * world_size
    else:
        selected_bs = 4 * world_size

    if is_distributed and dist.is_initialized():
        status = torch.tensor(
            [int(probe_ok), selected_bs],
            dtype=torch.long,
            device=device if device.type == "cuda" and torch.cuda.is_available() else "cpu",
        )
        dist.broadcast(status, src=0)
        probe_ok = bool(status[0].item())
        selected_bs = int(status[1].item())
        if not probe_ok:
            raise RuntimeError("Auto batch size probing failed on rank 0")
        if rank != 0:
            logger.info(f"Worker rank received optimal global batch size: {selected_bs}")

    if not probe_ok:
        raise RuntimeError("Auto batch size probing failed on rank 0")

    return selected_bs


def _build_probe_dataset(config):
    """Build the production training dataset path used during real training."""
    from yolox.data import MosaicDetection, TrainTransform

    base_dataset = config.get_dataset(cache=False)
    return MosaicDetection(
        dataset=base_dataset,
        mosaic=True,
        img_size=config.input_size,
        preproc=TrainTransform(
            max_labels=120,
            albu_transform=config._build_albu_transform(),
        ),
        degrees=config.degrees,
        translate=config.translate,
        mosaic_scale=config.mosaic_scale,
        mixup_scale=config.mixup_scale,
        shear=config.shear,
        enable_mixup=config.enable_mixup,
        mosaic_prob=config.mosaic_prob,
        mixup_prob=config.mixup_prob,
    )


def _create_probe_model(config):
    """Create an isolated model without touching the cached training model."""
    # A shallow config copy gives get_model() an independent attribute cache
    # without duplicating a potentially large cached dataset.
    probe_config = copy.copy(config)
    if hasattr(probe_config, "model"):
        delattr(probe_config, "model")
    return probe_config.get_model()


def _probe_batch_size(
    config,
    device: torch.device,
    target_fraction: float,
    amp_enabled: bool,
    amp_dtype: str,
) -> int:
    logger.info("Searching for the optimal batch size via VRAM probing...")

    base_h, base_w = config.input_size
    multiscale_range = getattr(config, "multiscale_range", 5)
    if getattr(config, "random_size", None) is not None:
        max_h = max(config.random_size) * 32
        max_w = max(config.random_size) * 32
    else:
        max_h = base_h + multiscale_range * 32
        max_w = base_w + multiscale_range * 32
    max_input_size = (max_h, max_w)

    probe_dataset = _build_probe_dataset(config)
    probe_loader = torch.utils.data.DataLoader(
        probe_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    max_boxes = 0
    sample_img, sample_targets = None, None
    for i, batch_data in enumerate(probe_loader):
        img = batch_data[0]
        targets = batch_data[1]
        if img is None:
            continue

        num_boxes = (targets[..., 4] > 0).sum().item() if targets.ndim >= 2 else targets.shape[1]
        if num_boxes >= max_boxes:
            max_boxes = num_boxes
            sample_img, sample_targets = img, targets

        if i >= 99:
            break

    if sample_img is None:
        raise RuntimeError("Could not load a valid sample for auto batch size probing")

    model = _create_probe_model(config)
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    dtype = torch.float16 if (amp_enabled and amp_dtype == "float16") else torch.float32
    sample_img = sample_img.to(device).to(dtype)
    sample_targets = sample_targets.to(device).to(dtype)
    sample_targets.requires_grad = False
    sample_img, sample_targets = config.preprocess(sample_img, sample_targets, max_input_size)

    total_mem = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total_mem * target_fraction)

    torch_amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == "float16"))

    def _try_batch(bs: int) -> bool:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch_inps = sample_img.repeat(bs, 1, 1, 1)
            batch_targets = sample_targets.repeat(bs, 1, 1)
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch_amp_dtype):
                    outputs = model(batch_inps, batch_targets)
                loss = outputs["total_loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_inps, batch_targets)
                loss = outputs["total_loss"]
                loss.backward()
                optimizer.step()

            peak = torch.cuda.max_memory_reserved(device)
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            return peak <= target_mem
        except RuntimeError as e:
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                return False
            raise

    best_bs = 1
    fail_bs = None
    for bs in (2**i for i in range(1, 9)):
        if _try_batch(bs):
            best_bs = bs
        else:
            fail_bs = bs
            break

    if fail_bs is not None and fail_bs - best_bs > 1:
        lo, hi = best_bs + 1, fail_bs - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _try_batch(mid):
                best_bs = mid
                lo = mid + 1
            else:
                hi = mid - 1

    torch.cuda.empty_cache()

    logger.info(
        f"Optimal per-device batch size selected: {best_bs} "
        f"(targeted {target_fraction:.0%} of {total_mem / (1024**3):.1f} GB VRAM at "
        f"{max_input_size[0]}x{max_input_size[1]})"
    )
    return best_bs
