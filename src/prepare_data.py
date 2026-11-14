"""Stage 1 — download the Ultralytics Brain Tumor dataset and write its YOLO config.

Output (versioned by DVC): data/brain-tumor/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt   (YOLO format: class cx cy w h, normalized)
    data.yaml
"""

import io
import shutil
import urllib.request
import zipfile

import yaml

from common import ROOT, load_params

CLASS_NAMES = {0: "negative", 1: "positive"}


def main() -> None:
    params = load_params()["data"]
    out_dir = ROOT / params["dir"]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"Downloading {params['url']} ...")
    with urllib.request.urlopen(params["url"]) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    archive.extractall(out_dir)

    # Relative paths: train.py resolves them against `path` at runtime.
    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    for split in ("train", "val"):
        n = len(list((out_dir / "images" / split).glob("*")))
        print(f"{split}: {n} images")


if __name__ == "__main__":
    main()
