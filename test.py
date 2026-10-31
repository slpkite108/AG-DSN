import torch

import config as cfg
from train import build_model, compute_training_loss, slice_subclip_targets

def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = build_model().eval()
    features = torch.randn(1, cfg.SUBCLIP_LENGTH, cfg.FEATURE_DIM)
    targets = torch.randint(
        low=0,
        high=cfg.NUM_OUTPUT_CLASSES,
        size=(1, cfg.SUBCLIP_LENGTH),
    )

    current_targets, future_targets = slice_subclip_targets(targets)
    outputs = model.forward_window(features[:, :cfg.FRAME_MEMORY_LENGTH + 1])
    losses = compute_training_loss(outputs, current_targets, future_targets)
    losses["total"].backward()

    assert tuple(outputs["current_logits"].shape) == (1, 1, cfg.NUM_OUTPUT_CLASSES)
    assert tuple(outputs["future_logits"].shape) == (1, cfg.PREDICTION_LENGTH, cfg.NUM_OUTPUT_CLASSES)
    assert torch.isfinite(losses["total"])
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    print("AG-DSN synthetic check: PASS")
    print({key: tuple(value.shape) for key, value in outputs.items() if torch.is_tensor(value)})

if __name__ == "__main__":
    main()
