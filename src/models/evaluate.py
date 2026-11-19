"""Stage 3 — export, quantize, evaluate, benchmark and register the model.

Run with:  python -m src.models.evaluate   (or `dvc repro export`)
Options:   --skip-export  reuse models/best.onnx (no PyTorch / Ultralytics needed)
           --no-mlflow    do not log or register anything (quick local check)

1. Export models/best.pt -> models/best.onnx (FP32, the file the API serves)
2. Quantize it: FP16 (weights + activations in half precision) and INT8 (static, QDQ,
   calibrated on training images) -> models/best_fp16.onnx, models/best_int8.onnx
3. Evaluate the three ONNX models on the validation set with the API's own inference
   code (backend/app/inference.py) and NumPy metrics (src/utils/metrics.py):
   precision, recall, mAP50, mAP50-95
4. Benchmark CPU latency (pre-processing + model + NMS, median per image), and PyTorch
   when the .pt checkpoint is exported in the same run
5. MLflow: log everything in the training run, register the FP32 ONNX model in the
   Model Registry and promote it: alias `production` + stage "Production"
6. Write models/metadata.json, read by the API's GET /model/metadata
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnx.version_converter

from backend.app.inference import TumorDetector, letterbox
from src.data.dataset import load_split
from src.utils.common import ROOT, load_params, setup_mlflow
from src.utils.metrics import detection_metrics, match_predictions

MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
VARIANTS = {"fp32": "best.onnx", "fp16": "best_fp16.onnx", "int8": "best_int8.onnx"}
METRIC_NAMES = ("precision", "recall", "mAP50", "mAP50_95")
EVAL_CONF, EVAL_IOU = 0.001, 0.7  # same thresholds as Ultralytics validation


# ---------------------------------------------------------------- export + quantization
def export_onnx(pt_path: Path, imgsz: int, opset: int, simplify: bool) -> tuple[Path, dict[int, str]]:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False)
    onnx_path = MODELS / VARIANTS["fp32"]
    shutil.move(exported, onnx_path)
    onnx.checker.check_model(str(onnx_path))
    return onnx_path, {int(k): v for k, v in model.names.items()}


def to_fp16(src: Path, dst: Path) -> Path:
    """Half-precision weights and activations; inputs/outputs stay float32 for the API."""
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    # Stale shape annotations make the converter insert mistyped Casts around Resize: drop them,
    # ONNX Runtime infers the shapes again at load time.
    del model.graph.value_info[:]
    model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model, str(dst))
    return dst


class CalibrationReader:
    """Feeds pre-processed training images to the INT8 calibrator (same pre-processing as the API)."""

    def __init__(self, images: list[Path], input_name: str, imgsz: int):
        self.batches = iter([{input_name: preprocess(cv2.imread(str(p)), imgsz)} for p in images])

    def get_next(self):
        return next(self.batches, None)


def to_int8(src: Path, dst: Path, calib_images: list[Path], imgsz: int) -> Path:
    """Static INT8 quantization (QDQ, per-channel weights) calibrated on training images."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared = dst.with_suffix(".prep.onnx")
    # Per-channel QDQ needs opset >= 13 (DequantizeLinear `axis`); the API model is exported at 12.
    upgraded = dst.with_suffix(".op13.onnx")
    onnx.save(onnx.version_converter.convert_version(onnx.load(str(src)), 13), str(upgraded))
    quant_pre_process(str(upgraded), str(prepared))
    upgraded.unlink(missing_ok=True)
    graph = onnx.load(str(prepared)).graph
    input_name = graph.input[0].name
    # Keep the box decoding of the Detect head (DFL, anchors, concat of boxes in pixels + scores in
    # [0, 1]) in FP32: one INT8 scale cannot cover both ranges and the scores collapse.
    head = max(int(n.name.split("/")[1].split(".")[1]) for n in graph.node if n.name.startswith("/model."))
    prefix = f"/model.{head}/"
    keep_fp32 = [
        n.name
        for n in graph.node
        if n.name.startswith(prefix) and not any(b in n.name for b in ("/cv2", "/cv3"))
    ]
    quantize_static(
        str(prepared),
        str(dst),
        CalibrationReader(calib_images, input_name, imgsz),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=keep_fp32,
    )
    prepared.unlink(missing_ok=True)
    return dst


def preprocess(image_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    img, _, _ = letterbox(image_bgr, imgsz)
    return np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


# ---------------------------------------------------------------- evaluation + benchmark
def evaluate(detector: TumorDetector, samples, classes: list[int]) -> dict[str, float]:
    tps, confs, pred_cls, gt_cls = [], [], [], []
    for s in samples:
        dets = detector.predict(cv2.imread(str(s.image)), EVAL_CONF, EVAL_IOU)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32).reshape(-1, 4)
        cls = np.array([d.class_id for d in dets], dtype=int)
        tps.append(match_predictions(boxes, cls, s.boxes, s.classes))
        confs.append(np.array([d.confidence for d in dets]))
        pred_cls.append(cls)
        gt_cls.append(s.classes)
    return detection_metrics(
        np.concatenate(tps), np.concatenate(confs), np.concatenate(pred_cls), np.concatenate(gt_cls), classes
    )


def latency_ms(detector: TumorDetector, images: list[Path]) -> float:
    """Median end-to-end time per image (pre-processing + ONNX Runtime + NMS) after warm-up."""
    arrays = [cv2.imread(str(p)) for p in images]
    for a in arrays[:3]:
        detector.predict(a)
    times = []
    for a in arrays:
        t0 = time.perf_counter()
        detector.predict(a)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def pytorch_latency_ms(pt_path: Path, images: list[Path], imgsz: int) -> float:
    from ultralytics import YOLO

    model = YOLO(str(pt_path))
    for img in images[:3]:
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for img in images:
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


# ---------------------------------------------------------------- MLflow
def register(params: dict, report: dict, metadata: dict) -> str | None:
    """Log to the training run, register the FP32 ONNX model and promote it to production."""
    mlflow = setup_mlflow(params["mlflow"]["experiment"])
    from mlflow import MlflowClient
    from mlflow import onnx as mlflow_onnx

    name = params["mlflow"]["registered_model"]
    run_id = (REPORTS / "run_id.txt").read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({f"export_{k}": v for k, v in params["export"].items()})
        mlflow.log_metrics(report)
        mlflow.log_artifact(str(MODELS / "labels.json"))
        for variant in ("fp16", "int8"):
            mlflow.log_artifact(str(MODELS / VARIANTS[variant]), artifact_path="quantized")
        info = mlflow_onnx.log_model(
            onnx.load(str(MODELS / VARIANTS["fp32"])), artifact_path="onnx_model", registered_model_name=name
        )
        version = str(getattr(info, "registered_model_version", "") or "")
        client = MlflowClient()
        if not version:
            versions = client.search_model_versions(f"name='{name}'")
            version = max(versions, key=lambda v: int(v.version)).version
        client.set_registered_model_alias(name, "production", version)
        try:  # classic stages, still shown by the MLflow 2.x UI
            client.transition_model_version_stage(name, version, "Production", archive_existing_versions=True)
        except Exception as e:  # noqa: BLE001 - stages are deprecated, the alias is what matters
            print(f"Stage transition skipped: {e}")
        metadata["version"] = f"v{version}"
        (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(MODELS / "metadata.json"))
    print(f"Registered '{name}' v{version} from run {run_id} -> alias 'production'")
    return version


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-export", action="store_true", help="reuse models/best.onnx")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log / register")
    args = parser.parse_args()

    params = load_params()
    p = params["export"]
    data_dir = ROOT / params["data"]["dir"]
    pt_path = MODELS / "best.pt"
    REPORTS.mkdir(exist_ok=True)

    # 1. Export
    if args.skip_export:
        labels = {int(k): v for k, v in json.loads((MODELS / "labels.json").read_text()).items()}
    else:
        _, labels = export_onnx(pt_path, p["imgsz"], p["opset"], p["simplify"])
        (MODELS / "labels.json").write_text(json.dumps(labels, indent=2))

    # 2. Quantization
    train_images = sorted((data_dir / "images" / "train").glob("*"))
    n_calib = p["calibration_images"]
    calib = train_images[:: max(1, len(train_images) // n_calib)][:n_calib]  # spread over the split
    to_fp16(MODELS / VARIANTS["fp32"], MODELS / VARIANTS["fp16"])
    to_int8(MODELS / VARIANTS["fp32"], MODELS / VARIANTS["int8"], calib, p["imgsz"])

    # 3-4. Accuracy + speed of each variant, with the API's inference code
    samples = load_split(data_dir, "val")
    bench_images = [s.image for s in samples[: p["benchmark_images"]]]
    report: dict[str, float] = {}
    for variant, filename in VARIANTS.items():
        detector = TumorDetector(MODELS / filename, MODELS / "labels.json")
        metrics = evaluate(detector, samples, sorted(labels))
        report.update({f"{variant}_{k}": v for k, v in metrics.items()})
        report[f"{variant}_cpu_ms"] = latency_ms(detector, bench_images)
        report[f"{variant}_size_mb"] = (MODELS / filename).stat().st_size / 1e6
        print(f"{variant}: {metrics} | {report[f'{variant}_cpu_ms']:.1f} ms")
    if not args.skip_export:
        report["pytorch_cpu_ms"] = pytorch_latency_ms(pt_path, bench_images, p["imgsz"])
        report["pytorch_size_mb"] = pt_path.stat().st_size / 1e6
    with open(REPORTS / "export.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 6. Model card for the API
    train_metrics = json.loads((REPORTS / "metrics.json").read_text())
    metadata = {
        "name": params["mlflow"]["registered_model"],
        "version": None,
        "architecture": Path(params["train"]["model"]).stem.replace("yolo", "YOLO"),
        "dataset": f"Ultralytics Brain Tumor ({len(train_images)} train / {len(samples)} val images)",
        "served_precision": "fp32",
        "validation_metrics": {k: round(report[f"fp32_{k}"], 4) for k in METRIC_NAMES},
        "training_metrics": {k: round(v, 4) for k, v in train_metrics.items() if k != "fitness"},
        "benchmark": {k: round(v, 2) for k, v in report.items() if k.endswith(("_cpu_ms", "_size_mb"))},
    }
    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # 5. MLflow
    if not args.no_mlflow:
        register(params, report, metadata)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
