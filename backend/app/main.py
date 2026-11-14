"""Brain Tumor Detection API — FastAPI (async) + ONNX Runtime.

Swagger UI: http://localhost:8000/docs
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from .inference import TumorDetector
from .schemas import BoundingBox, Detection, HealthResponse, ModelInfo, PredictionResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.onnx"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "models/labels.json"))
MAX_UPLOAD_MB = 10

state: dict = {"detector": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ONNX model once at startup (not on every request)
    if MODEL_PATH.exists():
        state["detector"] = TumorDetector(MODEL_PATH, LABELS_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    else:
        print(f"WARNING: model not found at {MODEL_PATH} — /predict will return 503")
    yield
    state["detector"] = None


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
    return HealthResponse(status="ok", model_loaded=state["detector"] is not None)


@app.get("/model-info", response_model=ModelInfo, tags=["system"])
async def model_info():
    det = get_detector()
    return ModelInfo(model_path=str(MODEL_PATH), input_size=det.imgsz, labels=det.labels)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="MRI/CT image (jpg, png)"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.01, le=1.0, description="NMS IoU threshold"),
):
    detector = get_detector()
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")

    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image larger than {MAX_UPLOAD_MB} MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    t0 = time.perf_counter()
    # CPU-bound inference runs in a worker thread so the event loop stays responsive
    dets = await run_in_threadpool(detector.predict, image, conf, iou)
    elapsed = (time.perf_counter() - t0) * 1000

    tumor_conf = [d.confidence for d in dets if d.label == "positive"]
    return PredictionResponse(
        tumor_detected=bool(tumor_conf),
        num_detections=len(dets),
        max_tumor_confidence=max(tumor_conf, default=0.0),
        image_width=image.shape[1],
        image_height=image.shape[0],
        inference_ms=round(elapsed, 2),
        detections=[
            Detection(label=d.label, class_id=d.class_id, confidence=round(d.confidence, 4),
                      box=BoundingBox(x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2))
            for d in dets
        ],
    )
