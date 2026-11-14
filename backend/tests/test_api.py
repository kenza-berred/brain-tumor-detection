"""API tests with a fake detector, so they run in CI without the trained model."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.inference import Detection
from app.main import app, state


class FakeDetector:
    imgsz = 640
    labels = {0: "negative", 1: "positive"}

    def predict(self, image, conf=0.25, iou_thres=0.45):
        return [Detection(10, 20, 110, 120, 0.91, 1, "positive")]


def png_bytes() -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def with_model(client):
    state["detector"] = FakeDetector()
    yield client
    state["detector"] = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_without_model_returns_503(client):
    state["detector"] = None
    r = client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 503


def test_predict_ok(with_model):
    r = with_model.post("/predict?conf=0.3",
                        files={"file": ("scan.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["tumor_detected"] is True
    assert body["num_detections"] == 1
    assert body["image_width"] == 64
    assert body["detections"][0]["box"] == {"x1": 10, "y1": 20, "x2": 110, "y2": 120}


def test_predict_rejects_non_image(with_model):
    r = with_model.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_predict_rejects_corrupted_image(with_model):
    r = with_model.post("/predict", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_predict_validates_threshold(with_model):
    r = with_model.post("/predict?conf=5",
                        files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 422


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]
