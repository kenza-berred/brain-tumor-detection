"""Stage 2 — fine-tune YOLO11 on the brain tumor dataset and track everything in MLflow.

Transfer learning: we start from weights pre-trained on COCO (yolo11n.pt) and
fine-tune all layers on our 2 classes (negative / positive).

Outputs (versioned by DVC):
    models/best.pt          best checkpoint (highest fitness on val)
    reports/metrics.json    final validation metrics
    reports/run_id.txt      MLflow run id, reused by export.py
"""

import csv
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

from common import ROOT, load_params, setup_mlflow

# We log to MLflow ourselves (cleaner runs, full control) -> disable the built-in callback.
settings.update({"mlflow": False})

NO_AUGMENT = dict(mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                  fliplr=0.0, translate=0.0, scale=0.0)


def resolved_data_yaml(data_dir: Path) -> str:
    """Copy data.yaml with an absolute `path` so YOLO finds the images from any cwd."""
    with open(data_dir / "data.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(data_dir.resolve())
    tmp = Path(tempfile.mkdtemp()) / "data.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(tmp)


def clean(name: str) -> str:
    """'metrics/mAP50(B)' -> 'mAP50' (MLflow-friendly metric names)."""
    return name.strip().split("/")[-1].replace("(B)", "").replace("-", "_")


def log_epoch_curves(mlflow, results_csv: Path) -> None:
    """Log every epoch so MLflow draws the learning curves."""
    with open(results_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            step = int(float(row.pop("epoch")))
            for key, value in row.items():
                try:
                    mlflow.log_metric(clean(key) if "metrics" in key else key.replace("/", "_"),
                                      float(value), step=step)
                except ValueError:
                    pass


def main() -> None:
    params = load_params()
    p = params["train"]
    data_dir = ROOT / params["data"]["dir"]
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)

    mlflow = setup_mlflow(params["mlflow"]["experiment"])
    run_name = f"{Path(p['model']).stem}-e{p['epochs']}-{'aug' if p['augment'] else 'noaug'}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({k: v for k, v in p.items()})
        mlflow.set_tags({"task": "detection", "dataset": "ultralytics/brain-tumor"})

        model = YOLO(p["model"])  # COCO pre-trained weights = transfer learning
        extra = {} if p["augment"] else NO_AUGMENT
        results = model.train(
            data=resolved_data_yaml(data_dir),
            epochs=p["epochs"],
            imgsz=p["imgsz"],
            batch=p["batch"],
            patience=p["patience"],
            seed=p["seed"],
            deterministic=True,
            device=p["device"] or None,
            project=str(ROOT / "runs"),
            name="train",
            exist_ok=True,
            plots=True,
            verbose=False,
            **extra,
        )

        save_dir = Path(results.save_dir)
        log_epoch_curves(mlflow, save_dir / "results.csv")

        # Final metrics of the best checkpoint on the validation set
        final = {clean(k): float(v) for k, v in results.results_dict.items()}
        mlflow.log_metrics({f"final_{k}": v for k, v in final.items()})

        # Plots (confusion matrix, PR curve, ...) + weights
        for plot in save_dir.glob("*.png"):
            mlflow.log_artifact(str(plot), artifact_path="plots")
        best = save_dir / "weights" / "best.pt"
        mlflow.log_artifact(str(best), artifact_path="weights")
        shutil.copy(best, ROOT / "models" / "best.pt")

        with open(ROOT / "reports" / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2)
        (ROOT / "reports" / "run_id.txt").write_text(run.info.run_id)

        print(json.dumps(final, indent=2))
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
