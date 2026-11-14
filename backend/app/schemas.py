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
    tumor_detected: bool = Field(..., description="True if at least one 'positive' box")
    num_detections: int
    max_tumor_confidence: float = Field(0.0, ge=0, le=1)
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool


class ModelInfo(BaseModel):
    model_path: str
    input_size: int
    labels: dict[int, str]
    runtime: str = "onnxruntime"
