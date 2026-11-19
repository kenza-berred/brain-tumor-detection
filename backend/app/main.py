"""Brain Tumor Detection API — FastAPI (async) + ONNX Runtime.

Swagger UI: http://localhost:8000/docs

Endpoints: GET /health, GET /model/metadata, POST /predict, POST /predict/visualize
"""

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .inference import TUMOR_LABELS, TumorDetector, draw_detections
from .inference import Detection as RawDetection
from .schemas import (
    BoundingBox,
    Detection,
    HealthResponse,
    ModelMetadata,
    PredictionResponse,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.onnx"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "models/labels.json"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "models/metadata.json"))
MAX_UPLOAD_MB = 10

state: dict = {"detector": None, "metadata": {}, "version": None}


def load_metadata() -> dict:
    """models/metadata.json is written by src/export.py (architecture, metrics, registry version)."""
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def model_version(metadata: dict) -> str:
    """Registry version from metadata.json when known, plus a short hash of the ONNX file."""
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:8]
    version = metadata.get("version")
    return f"{version}-{digest}" if version else f"sha-{digest}"


def gpu_providers() -> tuple[bool, list[str]]:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    return "CUDAExecutionProvider" in providers, providers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ONNX model once at startup (not on every request)
    if MODEL_PATH.exists():
        state["detector"] = TumorDetector(MODEL_PATH, LABELS_PATH)
        state["metadata"] = load_metadata()
        state["version"] = model_version(state["metadata"])
        print(f"Model loaded: {MODEL_PATH} ({state['version']})")
    else:
        print(f"WARNING: model not found at {MODEL_PATH} — /predict will return 503")
    yield
    state.update(detector=None, metadata={}, version=None)


app = FastAPI(
    title="Brain Tumor Detection API",
    description="Detects brain tumors on MRI/CT scans with a fine-tuned YOLO11 model "
    "served with ONNX Runtime. Research/education demo — not a medical device.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_detector() -> TumorDetector:
    if state["detector"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state["detector"]


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Server status, GPU availability and the version of the loaded model."""
    gpu, providers = gpu_providers()
    return HealthResponse(
        status="ok",
        model_loaded=state["detector"] is not None,
        model_version=state["version"],
        gpu_available=gpu,
        execution_providers=providers,
    )


@app.get("/model/metadata", response_model=ModelMetadata, tags=["system"])
async def model_metadata():
    """Model card: name, architecture, classes, validation metrics and benchmark."""
    det = get_detector()
    meta = state["metadata"]
    return ModelMetadata(
        name=meta.get("name", "brain-tumor-detector"),
        version=state["version"],
        architecture=meta.get("architecture", "YOLO11"),
        input_size=det.imgsz,
        classes=det.labels,
        dataset=meta.get("dataset"),
        validation_metrics=meta.get("validation_metrics", {}),
        benchmark=meta.get("benchmark", {}),
    )


@app.get("/model-info", response_model=ModelMetadata, tags=["system"], deprecated=True)
async def model_info():
    """Old name of /model/metadata, kept so existing clients keep working."""
    return await model_metadata()


async def read_image(file: UploadFile) -> np.ndarray:
    """Validate the upload and decode it: 415 not an image, 413 too big, 400 unreadable."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image larger than {MAX_UPLOAD_MB} MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


async def run_model(image: np.ndarray, conf: float, iou: float) -> tuple[list[RawDetection], float]:
    detector = get_detector()
    t0 = time.perf_counter()
    # CPU-bound inference runs in a worker thread so the event loop stays responsive
    dets = await run_in_threadpool(detector.predict, image, conf, iou)
    return dets, round((time.perf_counter() - t0) * 1000, 2)


CONF_QUERY = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold")
IOU_QUERY = Query(0.45, ge=0.01, le=1.0, description="NMS IoU threshold")
IMAGE_FILE = File(..., description="MRI/CT image (jpg, png)")


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = IMAGE_FILE,
    conf: float = CONF_QUERY,
    iou: float = IOU_QUERY,
):
    """Detect tumors and return the boxes as JSON."""
    get_detector()
    image = await read_image(file)
    dets, elapsed = await run_model(image, conf, iou)

    tumor_conf = [d.confidence for d in dets if d.label in TUMOR_LABELS]
    return PredictionResponse(
        tumor_detected=bool(tumor_conf),
        num_detections=len(dets),
        max_tumor_confidence=max(tumor_conf, default=0.0),
        image_width=image.shape[1],
        image_height=image.shape[0],
        inference_ms=elapsed,
        detections=[
            Detection(
                label=d.label,
                class_id=d.class_id,
                confidence=round(d.confidence, 4),
                box=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2),
            )
            for d in dets
        ],
    )


@app.post(
    "/predict/visualize",
    tags=["inference"],
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "The image with the boxes drawn on it"}},
)
async def predict_visualize(
    file: UploadFile = IMAGE_FILE,
    conf: float = CONF_QUERY,
    iou: float = IOU_QUERY,
):
    """Detect tumors and return the annotated image (PNG): one colour per class, label and score."""
    get_detector()
    image = await read_image(file)
    dets, elapsed = await run_model(image, conf, iou)
    annotated = draw_detections(image, dets)
    ok, png = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the annotated image")
    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={"X-Inference-Ms": str(elapsed), "X-Num-Detections": str(len(dets))},
    )
