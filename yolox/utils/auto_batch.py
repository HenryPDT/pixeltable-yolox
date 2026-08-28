# Copyright (c) Megvii, Inc. and its affiliates.
# Auto batch size search utility inspired by D-FINE-seg.

from __future__ import annotations

from typing import Optional
from loguru import logger
import torch
import torch.distributed as dist


def auto_batch_size(
    config,
    device: torch.device | str = "cuda:0",
    is_distributed: bool = False,
    target_fraction: float = 0.7,
    amp_enabled: bool = True,
    amp_dtype: str = "float16",
) -> int:
    """Probe the GPU to find the largest batch size that fits within *target_fraction* of total VRAM.

    Uses real forward + backward passes with the model and dataset at maximum multi-scale size
    to account for activations, SimOTA matching cost matrix, loss backward, etc.

    Args:
        config: YoloxConfig instance
        device: target torch device
        is_distributed: whether DDP is active
        target_fraction: fraction of total VRAM to target (default: 0.70)
        amp_enabled: whether mixed precision is active
        amp_dtype: 'float16' or 'bfloat16'

    Returns:
        int: optimal physical batch size per device
    """
    device = torch.device(device) if isinstance(device, str) else device

    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("Auto batch size only works on CUDA devices, defaulting to batch_size=4")
        return 4

    rank = dist.get_rank() if (is_distributed and dist.is_initialized()) else 0

    if rank != 0:
        # Worker ranks wait for broadcast from rank 0
        result_tensor = torch.tensor([4], dtype=torch.long, device=device)
        dist.broadcast(result_tensor, src=0)
        selected_bs = int(result_tensor.item())
        logger.info(f"Worker rank received optimal batch size: {selected_bs}")
        return selected_bs

    logger.info("Searching for the optimal batch size via VRAM probing...")

    # Determine maximum multi-scale input resolution for memory headroom
    base_h, base_w = config.input_size
    multiscale_range = getattr(config, "multiscale_range", 5)
    max_h = base_h + multiscale_range * 32
    max_w = base_w + multiscale_range * 32
    max_input_size = (max_h, max_w)

    # Build a single-threaded probe dataloader
    try:
        probe_dataset = config.get_dataset(cache=False)
    except Exception as e:
        logger.warning(f"Could not initialize probe dataset: {e}, defaulting to batch_size=4")
        return 4

    probe_loader = torch.utils.data.DataLoader(
        probe_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Scan up to 100 samples to find the one with the most ground truth boxes (worst-case for SimOTA matching)
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
        logger.warning("Could not load a valid sample for auto batch size, defaulting to batch_size=4")
        return 4

    # Build a lightweight probe model on the target device
    model = config.get_model()
    model.to(device)
    model.train()

    # Preprocess the sample to maximum multi-scale dimensions
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
        """Run forward + backward at batch size `bs`. Return True if within target VRAM."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        batch_inps = sample_img.repeat(bs, 1, 1, 1)
        batch_targets = sample_targets.repeat(bs, 1, 1)

        try:
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch_amp_dtype):
                    outputs = model(batch_inps, batch_targets)
                loss = outputs["total_loss"]
                scaler.scale(loss).backward()
            else:
                outputs = model(batch_inps, batch_targets)
                loss = outputs["total_loss"]
                loss.backward()

            peak = torch.cuda.max_memory_reserved(device)
            model.zero_grad(set_to_none=True)
            return peak <= target_mem

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return False
            raise

    # Phase 1: Escalate through powers of 2 (2, 4, 8, 16, 32, 64, 128, 256)
    best_bs = 1
    fail_bs = None
    for bs in (2**i for i in range(1, 9)):  # 2 up to 256
        if _try_batch(bs):
            best_bs = bs
        else:
            fail_bs = bs
            break

    # Phase 2: Binary search within [best_bs + 1, fail_bs - 1] to find maximum batch size
    if fail_bs is not None and fail_bs - best_bs > 1:
        lo, hi = best_bs + 1, fail_bs - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _try_batch(mid):
                best_bs = mid
                lo = mid + 1
            else:
                hi = mid - 1

    # Cleanup probing resources
    del model, sample_img, sample_targets, probe_loader, probe_dataset, scaler
    torch.cuda.empty_cache()

    logger.info(
        f"Optimal batch size selected: {best_bs} "
        f"(targeted {target_fraction:.0%} of {total_mem / (1024**3):.1f} GB VRAM at {max_input_size[0]}x{max_input_size[1]})"
    )

    if is_distributed and dist.is_initialized():
        result_tensor = torch.tensor([best_bs], dtype=torch.long, device=device)
        dist.broadcast(result_tensor, src=0)

    return best_bs
