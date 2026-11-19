"""YOLO11 inference with ONNX Runtime — no PyTorch / Ultralytics needed in the API image.

Pipeline:  image -> letterbox (640x640) -> ONNX model -> decode -> NMS -> boxes in
original image coordinates.

The YOLO11 detection head outputs a tensor of shape (1, 4 + num_classes, 8400):
for each of the 8400 anchor points: (cx, cy, w, h, score_class0, score_class1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping the aspect ratio and pad to a square (gray 114, like Ultralytics)."""
    h, w = img.shape[:2]
    ratio = min(size / h, size / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, ratio, (pad_x, pad_y)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Intersection over Union between one box and N boxes (x1, y1, x2, y2)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Non-Maximum Suppression: keep the best box, drop boxes overlapping it too much."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        overlaps = iou(boxes[best], boxes[order[1:]])
        order = order[1:][overlaps < iou_thres]
    return keep


def postprocess(
    output: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    labels: dict[int, str],
) -> list[Detection]:
    preds = output[0].T  # (8400, 4 + nc)
    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores.max(axis=1)

    mask = scores >= conf_thres
    preds, class_ids, scores = preds[mask], class_ids[mask], scores[mask]
    if not len(preds):
        return []

    # (cx, cy, w, h) -> (x1, y1, x2, y2), then undo the letterbox
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_shape[0])

    # Class-agnostic NMS: a brain region is either "positive" or "negative", never both,
    # so overlapping boxes of different classes compete and only the most confident survives.
    detections = [
        Detection(
            *map(float, boxes[i]),
            float(scores[i]),
            int(class_ids[i]),
            labels.get(int(class_ids[i]), str(class_ids[i])),
        )
        for i in nms(boxes, scores, iou_thres)
    ]
    return sorted(detections, key=lambda d: d.confidence, reverse=True)


# One colour per class (BGR for OpenCV), the same as the frontend canvas
CLASS_COLORS = {"positive": (96, 69, 255), "tumor": (96, 69, 255), "negative": (230, 225, 92)}

# Class names that mean "tumor": `positive` (2-class dataset negative/positive) or `tumor`
# (1-class version of the dataset published on 2026-11-15). The API works with both models.
TUMOR_LABELS = frozenset({"positive", "tumor"})


def draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of the image with each box, its label and its score drawn on it."""
    out = image_bgr.copy()
    scale = max(out.shape[:2]) / 640
    thick = max(1, round(2 * scale))
    font = 0.5 * scale
    for d in detections:
        color = CLASS_COLORS.get(d.label, (255, 123, 46))
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, thick)
        text = f"{'tumor' if d.label in TUMOR_LABELS else 'no tumor'} {d.confidence:.0%}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        y = p1[1] - 4 if p1[1] - th - base - 4 > 0 else p2[1] + th + base + 4
        cv2.rectangle(out, (p1[0], y - th - base), (p1[0] + tw + 6, y + base), color, -1)
        cv2.putText(out, text, (p1[0] + 3, y), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), thick, cv2.LINE_AA)
    return out


class TumorDetector:
    def __init__(self, model_path: Path, labels_path: Path | None = None):
        import onnxruntime as ort

        # Use the GPU when ONNX Runtime has CUDA, otherwise the CPU
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2])
        self.labels = {0: "negative", 1: "positive"}
        if labels_path and labels_path.exists():
            self.labels = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou_thres: float = 0.45) -> list[Detection]:
        img, ratio, pad = letterbox(image_bgr, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->NCHW
        output = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        return postprocess(output, conf, iou_thres, ratio, pad, image_bgr.shape[:2], self.labels)
