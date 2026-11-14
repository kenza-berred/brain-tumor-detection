"""Stage 3 — export the best model to ONNX, check it, benchmark it, register it.

1. Export models/best.pt -> models/best.onnx (the file the API serves)
2. Validate the ONNX model on the val set (accuracy must not drop)
3. Benchmark CPU latency: PyTorch vs ONNX Runtime
4. Log everything in the same MLflow run and register the ONNX model
   in the MLflow Model Registry.
"""

import json
import shutil
import statistics
import time
from pathlib import Path

import onnx
from ultralytics import YOLO, settings

from common import ROOT, load_params, setup_mlflow
from train import clean, resolved_data_yaml

settings.update({"mlflow": False})


def latency_ms(model: YOLO, images: list[Path], imgsz: int) -> float:
    """Median end-to-end inference time per image on CPU (after warm-up)."""
    for img in images[:3]:
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for img in images:
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    params = load_params()
    p = params["export"]
    data_dir = ROOT / params["data"]["dir"]
    pt_path = ROOT / "models" / "best.pt"
    onnx_path = ROOT / "models" / "best.onnx"

    # 1. Export
    pt_model = YOLO(str(pt_path))
    exported = pt_model.export(format="onnx", imgsz=p["imgsz"], opset=p["opset"],
                               simplify=p["simplify"], dynamic=False)
    shutil.move(exported, onnx_path)
    onnx.checker.check_model(str(onnx_path))
    labels = {int(k): v for k, v in pt_model.names.items()}
    (ROOT / "models" / "labels.json").write_text(json.dumps(labels, indent=2))

    # 2. Accuracy of the ONNX model
    onnx_model = YOLO(str(onnx_path), task="detect")
    val = onnx_model.val(data=resolved_data_yaml(data_dir), imgsz=p["imgsz"],
                         batch=1, device="cpu", plots=False, verbose=False)
    onnx_metrics = {f"onnx_{clean(k)}": float(v) for k, v in val.results_dict.items()}

    # 3. Speed: PyTorch vs ONNX Runtime (CPU, same 30 val images)
    images = sorted((data_dir / "images" / "val").glob("*"))[:30]
    bench = {
        "pytorch_cpu_ms": latency_ms(YOLO(str(pt_path)), images, p["imgsz"]),
        "onnx_cpu_ms": latency_ms(onnx_model, images, p["imgsz"]),
    }
    bench["onnx_speedup"] = bench["pytorch_cpu_ms"] / bench["onnx_cpu_ms"]
    bench["onnx_size_mb"] = onnx_path.stat().st_size / 1e6
    bench["pytorch_size_mb"] = pt_path.stat().st_size / 1e6

    report = {**onnx_metrics, **bench}
    with open(ROOT / "reports" / "export.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    # 4. MLflow: same run as training + Model Registry
    mlflow = setup_mlflow(params["mlflow"]["experiment"])
    run_id = (ROOT / "reports" / "run_id.txt").read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({f"export_{k}": v for k, v in p.items()})
        mlflow.log_metrics(report)
        mlflow.log_artifact(str(ROOT / "models" / "labels.json"))
        import mlflow.onnx  # noqa: E402

        mlflow.onnx.log_model(
            onnx.load(str(onnx_path)),
            artifact_path="onnx_model",
            registered_model_name=params["mlflow"]["registered_model"],
        )
    print(f"Registered '{params['mlflow']['registered_model']}' from run {run_id}")


if __name__ == "__main__":
    main()
