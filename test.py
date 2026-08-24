import torch

import config as cfg
from train import build_model, compute_training_loss

def main() -> None:
    torch.manual_seed(7)
    model = build_model().eval()
    features = torch.randn(1, cfg.SUBCLIP_LENGTH, cfg.FEATURE_DIM)
    targets = torch.randint(
        low=0,
        high=cfg.NUM_OUTPUT_CLASSES,
        size=(1, cfg.SUBCLIP_LENGTH),
    )

    with torch.no_grad():
        outputs = model(features)
        losses = compute_training_loss(outputs, targets, targets)

    expected_shape = (1, cfg.SUBCLIP_LENGTH, cfg.NUM_OUTPUT_CLASSES)
    assert tuple(outputs["current_logits"].shape) == expected_shape
    assert tuple(outputs["future_logits"].shape) == expected_shape
    assert torch.isfinite(losses["total"])
    print("AG-DSN synthetic check: PASS")
    print({key: tuple(value.shape) for key, value in outputs.items() if torch.is_tensor(value)})

if __name__ == "__main__":
    main()
