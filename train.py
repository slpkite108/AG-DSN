from collections.abc import Iterable

import torch

from loss import LossWeights, ag_dsn_loss
from model import AGDSN
import config as cfg

def build_model() -> AGDSN:
    return AGDSN(
        num_classes=cfg.NUM_OUTPUT_CLASSES,
        d_model=cfg.FEATURE_DIM,
        nhead=cfg.ATTENTION_HEADS,
        num_encoder_layers=cfg.TRANSFORMER_LAYERS,
        num_decoder_layers=cfg.TRANSFORMER_LAYERS,
        use_backbone=False,
        frame_memory_len=cfg.FRAME_MEMORY_LENGTH,
        clip_memory_len=cfg.CLIP_MEMORY_LENGTH,
        prediction_len=cfg.PREDICTION_LENGTH,
    )

def build_optimizer_and_scheduler(model: torch.nn.Module):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.SCHEDULER_T_MAX,
        eta_min=cfg.SCHEDULER_ETA_MIN,
    )
    return optimizer, scheduler

def compute_training_loss(
    outputs: dict[str, torch.Tensor],
    current_targets: torch.Tensor,
    future_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weights = LossWeights(
        lambda_dice=cfg.LAMBDA_DICE,
        lambda_future=cfg.LAMBDA_FUTURE,
        focal_gamma=cfg.FOCAL_GAMMA,
    )
    return ag_dsn_loss(
        outputs["current_logits"],
        outputs["future_logits"],
        current_targets,
        future_targets,
        weights,
    )

def slice_subclip_targets(labels, history_len=cfg.FRAME_MEMORY_LENGTH,
                          prediction_len=cfg.PREDICTION_LENGTH):
    """Split labels into current t and ordered predictions t through t+F-1."""
    if labels.size(1) < history_len + prediction_len:
        raise ValueError("Subclip labels must cover history and the full prediction interval.")
    return labels[:, history_len:history_len + 1], labels[:, history_len:history_len + prediction_len]


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    required = ("features", "current_targets", "future_targets")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing required fields: {missing}")
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}

def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_window(batch["features"][:, :model.frame_memory_len + 1])
        losses = compute_training_loss(
            outputs,
            batch["current_targets"],
            batch["future_targets"],
        )
        losses["total"].backward()
        optimizer.step()
        total_loss += float(losses["total"].detach().cpu())
        batches += 1
    return total_loss / max(1, batches)

@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        outputs = model.forward_window(batch["features"][:, :model.frame_memory_len + 1])
        losses = compute_training_loss(
            outputs,
            batch["current_targets"],
            batch["future_targets"],
        )
        total_loss += float(losses["total"].cpu())
        batches += 1
    return total_loss / max(1, batches)

def fit(train_loader, validation_loader=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model)
    history = []
    for epoch in range(cfg.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_loss = None
        if validation_loader is not None:
            validation_loss = evaluate_one_epoch(model, validation_loader, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
    return model, history
