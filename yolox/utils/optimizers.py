# Copyright (c) Megvii, Inc. and its affiliates.
# Modern optimizer implementations for YOLOX, including Adan and parameter group builders.

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


def _adan_update(p, grad, st, betas, eps, lr, wd):
    """Adan (Xie et al., arXiv:2208.06677): adaptive Nesterov momentum. Updates p in place.

    betas=(b1,b2,b3) are decay coeffs; m = grad momentum, v = grad-diff momentum,
    n = second moment of (grad + b2*diff). Decoupled post-prox weight decay.
    """
    b1, b2, b3 = betas
    step = st["step"]
    m, v, n, neg = st["exp_avg"], st["exp_avg_diff"], st["exp_avg_sq"], st["neg_pre_grad"]
    neg.add_(grad)  # neg held -prev_grad -> now diff = grad - prev_grad (0 on step 1)
    m.mul_(b1).add_(grad, alpha=1 - b1)
    v.mul_(b2).add_(neg, alpha=1 - b2)
    neg.mul_(b2).add_(grad)  # grad + b2*diff
    n.mul_(b3).addcmul_(neg, neg, value=1 - b3)
    denom = (n.sqrt() / (1 - b3**step) ** 0.5).add_(eps)
    p.addcdiv_(m, denom, value=-lr / (1 - b1**step))
    p.addcdiv_(v, denom, value=-lr * b2 / (1 - b2**step))
    p.div_(1 + lr * wd)
    neg.zero_().add_(grad, alpha=-1.0)  # store -grad for next step


class Adan(torch.optim.Optimizer):
    """Adan optimizer: Adaptive Nesterov Momentum Algorithm (arXiv:2208.06677)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0 and 0.0 <= betas[2] < 1.0):
            raise ValueError(f"Invalid beta parameters: {betas}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            betas = group.get("betas", (0.98, 0.92, 0.99))
            eps = group.get("eps", 1e-8)

            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                    st["exp_avg_diff"] = torch.zeros_like(p)
                    st["neg_pre_grad"] = -p.grad.clone()
                st["step"] += 1
                _adan_update(p, p.grad, st, betas, eps, lr, wd)

        return loss


def build_yolox_optimizer(
    model: nn.Module,
    optimizer_type: str = "sgd",
    base_lr: float = 0.01,
    backbone_lr_ratio: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    adan_betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
) -> torch.optim.Optimizer:
    """Build optimizer with separated parameter groups:

    - Backbone Conv weights: lr = base_lr * backbone_lr_ratio, weight_decay active
    - Backbone Norm & Biases: lr = base_lr * backbone_lr_ratio, weight_decay = 0.0
    - Head & Neck Conv weights: lr = base_lr, weight_decay active
    - Head & Neck Norm & Biases: lr = base_lr, weight_decay = 0.0
    """
    backbone_weights = []
    backbone_no_decay = []
    head_weights = []
    head_no_decay = []

    # Identify backbone module names (CspDarknet is model.backbone.backbone)
    # The neck is model.backbone.lateral_conv0, model.backbone.C3_p4, etc.
    # The head is model.head.*
    for name, module in model.named_modules():
        is_backbone = name.startswith("backbone.backbone")

        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.SyncBatchNorm)):
            if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
                if is_backbone:
                    backbone_no_decay.append(module.weight)
                else:
                    head_no_decay.append(module.weight)
            if hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
                if is_backbone:
                    backbone_no_decay.append(module.bias)
                else:
                    head_no_decay.append(module.bias)
        elif hasattr(module, "bias") and isinstance(module.bias, nn.Parameter):
            # standalone module biases (e.g. Conv2d.bias or Linear.bias)
            if is_backbone:
                backbone_no_decay.append(module.bias)
            else:
                head_no_decay.append(module.bias)

    # Collect weights with weight decay (excluding norm weights and biases already captured)
    no_decay_set = set(backbone_no_decay + head_no_decay)
    for name, param in model.named_parameters():
        if param in no_decay_set:
            continue
        if name.startswith("backbone.backbone"):
            backbone_weights.append(param)
        else:
            head_weights.append(param)

    backbone_lr = base_lr * backbone_lr_ratio

    param_groups = [
        {
            "name": "backbone_weights",
            "params": backbone_weights,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "lr_multiplier": backbone_lr_ratio,
        },
        {
            "name": "backbone_no_decay",
            "params": backbone_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
            "lr_multiplier": backbone_lr_ratio,
        },
        {
            "name": "head_weights",
            "params": head_weights,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "lr_multiplier": 1.0,
        },
        {
            "name": "head_no_decay",
            "params": head_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "lr_multiplier": 1.0,
        },
    ]

    # Filter out empty parameter groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    opt_type = optimizer_type.lower()
    if opt_type == "sgd":
        return optim.SGD(param_groups, lr=base_lr, momentum=momentum, nesterov=True)
    elif opt_type == "adamw":
        return optim.AdamW(param_groups, lr=base_lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif opt_type == "adan":
        return Adan(param_groups, lr=base_lr, betas=adan_betas, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}. Choose from ['sgd', 'adamw', 'adan']")
