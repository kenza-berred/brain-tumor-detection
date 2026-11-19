"""Model definition — YOLO11 detector initialised from COCO weights (transfer learning).

YOLO11 = CSP backbone (C3k2 blocks) + SPPF + C2PSA attention block, PAN neck, and an
anchor-free decoupled detection head with 3 output scales (strides 8, 16, 32).
Only the classification branch of the head changes size for our 2 classes
(negative / positive); every other layer starts from the COCO pre-trained weights and
is fine-tuned end to end.

Two variants are compared (params.yaml -> train.model):
    yolo11n.pt   ~2.6 M parameters, ~6.5 GFLOPs   (served model: fastest on CPU)
    yolo11s.pt   ~9.4 M parameters, ~21.5 GFLOPs
"""

from __future__ import annotations

from ultralytics import YOLO

# Augmentations switched off for the "no augmentation" ablation run (train.augment: false)
NO_AUGMENT = dict(
    mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, fliplr=0.0, translate=0.0, scale=0.0
)


def build_model(weights: str = "yolo11n.pt") -> YOLO:
    """Load a YOLO11 detector with COCO pre-trained weights (downloaded on first use)."""
    return YOLO(weights)


def describe(model: YOLO) -> dict[str, float]:
    """Number of parameters (millions) and layers, logged to MLflow with each run."""
    params = sum(p.numel() for p in model.model.parameters())
    return {"params_millions": round(params / 1e6, 3), "layers": len(list(model.model.modules()))}
