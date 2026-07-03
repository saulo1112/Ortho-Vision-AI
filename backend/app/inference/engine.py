"""ONNX Runtime segmentation engine — the single inference entry point.

Loaded once at application startup (see app.main lifespan) and shared across
requests; ONNX Runtime sessions are thread-safe for run().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .postprocess import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    mask_to_polygons,
    non_max_suppression,
    process_masks,
    scale_boxes,
)
from .preprocess import IMG_SIZE, letterbox, to_tensor

CLASS_NAMES = {
    0: "clavo_intramedular",
    1: "placa_atornillada",
    2: "protesis_articular",
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized [0, 1]
    polygons: list[list[list[float]]]        # rings of [x, y] normalized [0, 1]
    mask: np.ndarray                          # (H0, W0) uint8, original resolution


@dataclass
class EngineResult:
    detections: list[Detection]
    timing_ms: dict[str, int] = field(default_factory=dict)


class SegmentationEngine:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(
        self,
        image_bgr: np.ndarray,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> EngineResult:
        """Run segmentation on a BGR uint8 image at its original resolution."""
        t0 = time.perf_counter()
        orig_shape = image_bgr.shape[:2]
        padded, ratio, pad = letterbox(image_bgr, IMG_SIZE)
        tensor = to_tensor(padded)

        t1 = time.perf_counter()
        output0, output1 = self.session.run(None, {self.input_name: tensor})

        t2 = time.perf_counter()
        dets = non_max_suppression(output0, conf_thres=conf, iou_thres=iou)
        masks = process_masks(output1, dets, ratio, pad, orig_shape, IMG_SIZE)
        boxes = scale_boxes(dets[:, :4], ratio, pad, orig_shape)

        detections = []
        for i in range(dets.shape[0]):
            class_id = int(dets[i, 5])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES.get(class_id, str(class_id)),
                    confidence=round(float(dets[i, 4]), 4),
                    bbox=tuple(round(v, 4) for v in boxes[i]),
                    polygons=mask_to_polygons(masks[i]),
                    mask=masks[i],
                )
            )
        t3 = time.perf_counter()

        timing = {
            "preprocess": round((t1 - t0) * 1000),
            "inference": round((t2 - t1) * 1000),
            "postprocess": round((t3 - t2) * 1000),
            "total": round((t3 - t0) * 1000),
        }
        return EngineResult(detections=detections, timing_ms=timing)
