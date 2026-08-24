

from dataclasses import dataclass

import torch
import torch.nn.functional as F

def _as_class_indices(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    if targets.ndim > 0 and targets.shape[-1] == num_classes:
        return targets.argmax(dim=-1)
    return targets.long()

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:

    num_classes = logits.shape[-1]
    class_targets = _as_class_indices(targets, num_classes)
    flat_logits = logits.reshape(-1, num_classes)
    flat_targets = class_targets.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    pt = flat_logits.softmax(dim=-1).gather(1, flat_targets[:, None]).squeeze(1)
    return (((1.0 - pt) ** gamma) * ce).mean()

def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:

    num_classes = logits.shape[-1]
    class_targets = _as_class_indices(targets, num_classes)
    one_hot = F.one_hot(class_targets, num_classes=num_classes).to(logits.dtype)
    probabilities = logits.softmax(dim=-1)
    reduce_dims = tuple(range(probabilities.ndim - 1))
    intersection = (probabilities * one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + one_hot.sum(dim=reduce_dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()

@dataclass(frozen=True)
class LossWeights:
    lambda_dice: float = 1.0
    lambda_future: float = 1.0
    focal_gamma: float = 2.0

def ag_dsn_loss(
    current_logits: torch.Tensor,
    future_logits: torch.Tensor,
    current_targets: torch.Tensor,
    future_targets: torch.Tensor,
    weights: LossWeights = LossWeights(),
) -> dict[str, torch.Tensor]:
    current_focal = focal_loss(current_logits, current_targets, weights.focal_gamma)
    current_dice = dice_loss(current_logits, current_targets)
    future_focal = focal_loss(future_logits, future_targets, weights.focal_gamma)
    future_dice = dice_loss(future_logits, future_targets)
    current_total = current_focal + weights.lambda_dice * current_dice
    future_total = future_focal + weights.lambda_dice * future_dice
    total = current_total + weights.lambda_future * future_total
    return {
        "total": total,
        "current_focal": current_focal,
        "current_dice": current_dice,
        "future_focal": future_focal,
        "future_dice": future_dice,
    }
