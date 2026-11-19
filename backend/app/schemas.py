"""Pydantic schemas — they validate the API responses and generate the Swagger docs."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left, in pixels of the original image")
    y1: float = Field(..., description="Top")
    x2: float = Field(..., description="Right")
    y2: float = Field(..., description="Bottom")


class Detection(BaseModel):
    label: str = Field(..., examples=["positive"])
    class_id: int = Field(..., examples=[1])
    confidence: float = Field(..., ge=0, le=1, examples=[0.87])
    box: BoundingBox


class PredictionResponse(BaseModel):
    tumor_detected: bool = Field(..., description="True if at least one tumor box ('positive' or 'tumor')")
    num_detections: int
    max_tumor_confidence: float = Field(0.0, ge=0, le=1)
    image_width: int
    image_height: int
    inference_ms: float = Field(..., description="Server-side model time (preprocess + ONNX + NMS)")
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    model_version: str | None = Field(
        None, description="Model version (registry version or file hash)", examples=["v2-3f9a1c2b"]
    )
    gpu_available: bool = Field(..., description="True if ONNX Runtime can use a CUDA GPU")
    execution_providers: list[str] = Field(..., examples=[["CPUExecutionProvider"]])


class ModelMetadata(BaseModel):
    name: str = Field(..., examples=["brain-tumor-detector"])
    version: str | None = Field(None, examples=["v2-3f9a1c2b"])
    architecture: str = Field(..., examples=["YOLO11n"])
    task: str = Field("object-detection", examples=["object-detection"])
    framework: str = Field("onnxruntime", examples=["onnxruntime"])
    input_size: int = Field(..., examples=[640])
    classes: dict[int, str] = Field(..., examples=[{0: "negative", 1: "positive"}])
    dataset: str | None = Field(None, examples=["Ultralytics Brain Tumor (893 train / 223 val)"])
    validation_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metrics of the exported ONNX model on the validation set",
        examples=[{"precision": 0.45, "recall": 0.84, "mAP50": 0.49, "mAP50_95": 0.35}],
    )
    benchmark: dict[str, float] = Field(default_factory=dict, description="CPU latency / size benchmark")
