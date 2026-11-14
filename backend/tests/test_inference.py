"""Unit tests for pre/post-processing — no model file needed."""

import numpy as np

from app.inference import iou, letterbox, nms, postprocess


def test_letterbox_keeps_ratio_and_pads():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    out, ratio, (pad_x, pad_y) = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert ratio == 640 / 400
    assert pad_x == 0 and pad_y == (640 - 320) // 2


def test_iou_identical_and_disjoint():
    a = np.array([0, 0, 10, 10], dtype=float)
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    res = iou(a, boxes)
    assert np.isclose(res[0], 1.0)
    assert np.isclose(res[1], 0.0)


def test_nms_removes_overlapping_box():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, iou_thres=0.5) == [0, 2]


def test_postprocess_decodes_and_filters():
    # 3 anchors, 2 classes -> output shape (1, 6, 3)
    preds = np.array([
        # cx, cy,  w,  h, negative, positive
        [100, 100, 20, 20, 0.05, 0.90],   # kept: positive
        [101, 101, 20, 20, 0.05, 0.80],   # removed by NMS (overlaps the first)
        [300, 300, 40, 40, 0.10, 0.10],   # removed: below threshold
    ], dtype=np.float32)
    output = preds.T[None]
    dets = postprocess(output, conf_thres=0.25, iou_thres=0.45, ratio=1.0, pad=(0, 0),
                       orig_shape=(640, 640), labels={0: "negative", 1: "positive"})
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "positive"
    assert np.isclose(d.confidence, 0.90)
    assert (d.x1, d.y1, d.x2, d.y2) == (90, 90, 110, 110)


def test_postprocess_nms_is_class_agnostic():
    preds = np.array([
        [100, 100, 20, 20, 0.80, 0.10],   # negative 0.80
        [100, 100, 20, 20, 0.10, 0.60],   # positive 0.60, same region -> suppressed
    ], dtype=np.float32)
    dets = postprocess(preds.T[None], 0.25, 0.45, 1.0, (0, 0), (640, 640),
                       {0: "negative", 1: "positive"})
    assert [d.label for d in dets] == ["negative"]


def test_postprocess_undoes_letterbox():
    preds = np.array([[320, 320, 64, 64, 0.0, 0.9]], dtype=np.float32)
    dets = postprocess(preds.T[None], 0.25, 0.45, ratio=2.0, pad=(0, 160),
                       orig_shape=(160, 320), labels={1: "positive"})
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (144, 64, 176, 96)
