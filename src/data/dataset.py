"""Stage 1 — download the Ultralytics Brain Tumor dataset and write its YOLO config.

Run with:  python -m src.data.dataset   (or `dvc repro prepare`)

Output (versioned by DVC): data/brain-tumor/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt   (YOLO format: class cx cy w h, normalized)
    data.yaml
"""

import hashlib
import io
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.utils.common import ROOT, load_params


def download(url: str, expected_md5: str | None) -> bytes:
    """Download the archive and check it is exactly the version this project was built on.

    The upstream zip can change without notice (the Ultralytics archive was replaced on
    2026-11-15: 2 classes / 893+223 images -> 1 class / 994+121 images). Pinning its MD5 makes
    such a change fail loudly instead of silently training on different data.
    """
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        payload = resp.read()
    md5 = hashlib.md5(payload).hexdigest()
    if expected_md5 and md5 != expected_md5:
        raise SystemExit(
            f"Dataset archive changed upstream: md5 {md5} != pinned {expected_md5}.\n"
            "Check what changed, then update data.md5 in params.yaml on purpose (and retrain)."
        )
    print(f"md5 {md5} OK")
    return payload


def main() -> None:
    params = load_params()["data"]
    out_dir = ROOT / params["dir"]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    archive = zipfile.ZipFile(io.BytesIO(download(params["url"], params.get("md5"))))
    archive.extractall(out_dir)

    # Class names come from the archive's own YAML, so they always match the label files.
    with open(out_dir / "brain-tumor.yaml", encoding="utf-8") as f:
        class_names = {int(k): v for k, v in yaml.safe_load(f)["names"].items()}

    # Relative paths: train.py resolves them against `path` at runtime.
    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"classes: {class_names}")
    for split in ("train", "val"):
        n = len(list((out_dir / "images" / split).glob("*")))
        print(f"{split}: {n} images")


@dataclass
class Sample:
    """One image of a split and its ground-truth boxes (pixels, x1 y1 x2 y2) + class ids."""

    image: Path
    boxes: np.ndarray  # (n, 4) float32
    classes: np.ndarray  # (n,) int


def load_split(data_dir: Path, split: str, image_size: tuple[int, int] | None = None) -> list[Sample]:
    """Read a YOLO split (images/<split> + labels/<split>) into pixel-space boxes.

    Used by src/models/evaluate.py to compute mAP with the same pre/post-processing as the API.
    `image_size` (w, h) skips reading each image when all images share one size.
    """
    import cv2

    samples = []
    for img_path in sorted((data_dir / "images" / split).glob("*")):
        label_path = data_dir / "labels" / split / f"{img_path.stem}.txt"
        rows = np.loadtxt(label_path, ndmin=2) if label_path.exists() else np.zeros((0, 5))
        if rows.size == 0:
            rows = np.zeros((0, 5))
        if image_size:
            w, h = image_size
        else:
            h, w = cv2.imread(str(img_path)).shape[:2]
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).astype(np.float32)
        samples.append(Sample(img_path, boxes, rows[:, 0].astype(int)))
    return samples


if __name__ == "__main__":
    main()
