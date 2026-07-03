"""Decoding of raw YOLOv8-seg ONNX outputs into instance masks and polygons.

The exported graph has two outputs:
  output0: (1, 4 + nc + 32, 8400) — box (cx, cy, w, h), class scores, mask coefficients
  output1: (1, 32, 160, 160)      — mask prototypes

This module re-implements the ultralytics post-processing (NMS + prototype
mask assembly) in pure numpy/OpenCV so the serving image needs neither torch
nor the ultralytics package. One deliberate improvement over the training
notebook: letterbox padding is cropped *before* resizing masks back to the
original resolution, instead of resizing the padded mask directly.
"""

from __future__ import annotations

import cv2
import numpy as np

NUM_MASK_COEFFS = 32
PROTO_SIZE = 160

DEFAULT_CONF = 0.5   # backed by the notebook's ablation (conf 0.5 vs 0.7)
DEFAULT_IOU = 0.7    # ultralytics default, matches the training config
MAX_DETECTIONS = 300

# Minimum blob area (fraction of image) kept when polygonizing a mask, and
# polygon simplification tolerance (fraction of contour perimeter).
MIN_BLOB_AREA_FRAC = 0.0005
POLY_EPSILON_FRAC = 0.002


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    half = boxes[:, 2:4] / 2
    out[:, 0:2] = boxes[:, 0:2] - half
    out[:, 2:4] = boxes[:, 0:2] + half
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy IoU-based non-maximum suppression. Returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def non_max_suppression(
    output0: np.ndarray,
    conf_thres: float = DEFAULT_CONF,
    iou_thres: float = DEFAULT_IOU,
    max_det: int = MAX_DETECTIONS,
) -> np.ndarray:
    """Filter raw head output down to final detections.

    Args:
        output0: (1, 4 + nc + 32, 8400) raw output.

    Returns:
        (N, 6 + 32) array: x1, y1, x2, y2, conf, class_id, 32 mask coeffs —
        boxes in letterboxed 640-space.
    """
    pred = output0[0].T  # (8400, 4 + nc + 32)
    num_classes = pred.shape[1] - 4 - NUM_MASK_COEFFS

    class_scores = pred[:, 4 : 4 + num_classes]
    conf = class_scores.max(axis=1)
    mask = conf >= conf_thres
    if not mask.any():
        return np.empty((0, 6 + NUM_MASK_COEFFS), dtype=np.float32)

    pred, conf = pred[mask], conf[mask]
    class_ids = pred[:, 4 : 4 + num_classes].argmax(axis=1).astype(np.float32)
    boxes = _xywh2xyxy(pred[:, :4])
    coeffs = pred[:, 4 + num_classes :]

    # Class-aware NMS via the standard per-class box offset trick.
    offset_boxes = boxes + class_ids[:, None] * 7680.0
    keep = _nms(offset_boxes, conf, iou_thres)[:max_det]

    return np.concatenate(
        [boxes[keep], conf[keep, None], class_ids[keep, None], coeffs[keep]], axis=1
    ).astype(np.float32)


def process_masks(
    protos: np.ndarray,
    detections: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    img_size: int = 640,
) -> np.ndarray:
    """Assemble per-instance binary masks at the original image resolution.

    Args:
        protos: (1, 32, 160, 160) prototype output.
        detections: (N, 6 + 32) rows from non_max_suppression.
        ratio: letterbox scale factor (original -> 640-space).
        pad: (left, top) letterbox padding in 640-space.
        orig_shape: (H0, W0) of the original image.

    Returns:
        (N, H0, W0) uint8 binary masks.
    """
    h0, w0 = orig_shape
    n = detections.shape[0]
    if n == 0:
        return np.empty((0, h0, w0), dtype=np.uint8)

    proto = protos[0].reshape(NUM_MASK_COEFFS, -1)  # (32, 160*160)
    coeffs = detections[:, 6:]                      # (N, 32)
    masks = coeffs @ proto                          # (N, 160*160)
    masks = 1.0 / (1.0 + np.exp(-masks))
    masks = masks.reshape(n, PROTO_SIZE, PROTO_SIZE)

    left, top = pad
    scale = PROTO_SIZE / img_size
    # Content region inside the padded square, expressed in proto resolution.
    # Derived from the ratio (not from symmetric padding, which can be off by
    # one pixel on the right/bottom edge).
    px1, py1 = round(left * scale), round(top * scale)
    px2 = min(px1 + max(1, round(w0 * ratio * scale)), PROTO_SIZE)
    py2 = min(py1 + max(1, round(h0 * ratio * scale)), PROTO_SIZE)

    out = np.zeros((n, h0, w0), dtype=np.uint8)
    for i in range(n):
        # Crop to the detection box (in proto space) so prototypes leaking
        # outside the box do not contaminate the instance mask.
        x1, y1, x2, y2 = detections[i, :4] * scale
        boxed = np.zeros_like(masks[i])
        bx1, by1 = max(int(x1), 0), max(int(y1), 0)
        bx2, by2 = min(int(np.ceil(x2)), PROTO_SIZE), min(int(np.ceil(y2)), PROTO_SIZE)
        boxed[by1:by2, bx1:bx2] = masks[i, by1:by2, bx1:bx2]

        # Drop letterbox padding first, then resize to the original resolution.
        content = boxed[py1:py2, px1:px2]
        resized = cv2.resize(content, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out[i] = (resized >= 0.5).astype(np.uint8)
    return out


def scale_boxes(
    boxes: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """Map xyxy boxes from letterboxed 640-space to normalized [0, 1] original coords."""
    h0, w0 = orig_shape
    left, top = pad
    out = boxes.copy().astype(np.float64)
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / ratio
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0) / w0
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0) / h0
    return out


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    """Vectorize a binary mask into simplified polygon rings.

    Returns a list of rings; each ring is [[x, y], ...] normalized to [0, 1].
    A fragmented mask yields one ring per blob; blobs smaller than
    MIN_BLOB_AREA_FRAC of the image are dropped as noise. Holes are ignored
    (RETR_EXTERNAL), which is acceptable for a translucent overlay.
    """
    h, w = mask.shape
    min_area = MIN_BLOB_AREA_FRAC * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rings: list[list[list[float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area or len(contour) < 3:
            continue
        epsilon = POLY_EPSILON_FRAC * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(approx) < 3:
            continue
        ring = [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in approx[:, 0, :]]
        rings.append(ring)
    return rings
