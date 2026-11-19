"""Tests for /health (GPU + version), /model/metadata and /predict/visualize.

The ONNX model is replaced by a fake detector, so these tests run in CI without the weights.
"""

from typing import ClassVar

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.inference import Detection, draw_detections


class FakeDetector:
    imgsz = 640
    labels: ClassVar[dict[int, str]] = {0: "negative", 1: "positive"}

    def predict(self, image, conf=0.25, iou=0.45):
        return [Detection(10, 20, 60, 80, 0.9, 1, "positive"), Detection(70, 70, 90, 95, 0.4, 0, "negative")]


def png_bytes(w=128, h=96) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 80, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def fake_client(monkeypatch):
    client = TestClient(main.app)  # no `with`: the real model is not loaded
    monkeypatch.setitem(main.state, "detector", FakeDetector())
    monkeypatch.setitem(main.state, "version", "v2-test")
    monkeypatch.setitem(
        main.state,
        "metadata",
        {
            "architecture": "YOLO11n",
            "validation_metrics": {"mAP50": 0.49, "mAP50_95": 0.35},
        },
    )
    return client


@pytest.fixture
def empty_client(monkeypatch):
    monkeypatch.setitem(main.state, "detector", None)
    monkeypatch.setitem(main.state, "version", None)
    return TestClient(main.app)


def test_health_reports_gpu_and_version(fake_client):
    body = fake_client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "v2-test"
    assert isinstance(body["gpu_available"], bool)
    assert "CPUExecutionProvider" in body["execution_providers"]


def test_health_without_model(empty_client):
    body = empty_client.get("/health").json()
    assert body["model_loaded"] is False and body["model_version"] is None


def test_model_metadata(fake_client):
    body = fake_client.get("/model/metadata").json()
    assert body["architecture"] == "YOLO11n"
    assert body["classes"] == {"0": "negative", "1": "positive"}
    assert body["input_size"] == 640
    assert body["validation_metrics"]["mAP50_95"] == 0.35


def test_model_metadata_without_model_returns_503(empty_client):
    assert empty_client.get("/model/metadata").status_code == 503


def test_visualize_returns_annotated_png(fake_client):
    res = fake_client.post("/predict/visualize", files={"file": ("scan.png", png_bytes(), "image/png")})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert res.headers["x-num-detections"] == "2"
    assert float(res.headers["x-inference-ms"]) >= 0
    img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (96, 128, 3)
    assert (img != 80).any()  # something was drawn


def test_visualize_rejects_non_image(fake_client):
    res = fake_client.post("/predict/visualize", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_visualize_rejects_corrupted_image(fake_client):
    res = fake_client.post("/predict/visualize", files={"file": ("a.png", b"not a png", "image/png")})
    assert res.status_code == 400


def test_visualize_without_model_returns_503(empty_client):
    res = empty_client.post("/predict/visualize", files={"file": ("scan.png", png_bytes(), "image/png")})
    assert res.status_code == 503


def test_draw_detections_uses_one_colour_per_class():
    img = np.zeros((100, 100, 3), np.uint8)
    out = draw_detections(
        img, [Detection(5, 30, 40, 60, 0.9, 1, "positive"), Detection(55, 30, 95, 60, 0.8, 0, "negative")]
    )
    left, right = out[45, 5], out[45, 95]  # points on each box's border
    assert tuple(left) != tuple(right)
    assert img.sum() == 0  # the input image is not modified
