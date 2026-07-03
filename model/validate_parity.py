"""Parity validation: best.pt (ultralytics) vs best.onnx (production engine).

Runs both pipelines over the held-out test set and verifies that the ONNX
path served in production is metrically equivalent to the academic .pt model:

  1. GT metrics — per-class IoU/Dice against ground truth for BOTH pipelines,
     reproducing the notebook's evaluation (conf=0.5). Acceptance: |delta| <= 0.01.
  2. Direct agreement — mask-to-mask IoU between .pt and ONNX outputs per
     image/class. Acceptance: mean >= 0.99.
  3. CPU latency of the production engine.

The ONNX side imports the *actual backend code* (backend/app/inference), so
what is validated here is byte-for-byte what serves in production.

Usage:
    python validate_parity.py            # expects dataset downloaded (download_dataset.py)

Writes ../docs/onnx-parity.md.
"""

from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.inference.engine import CLASS_NAMES, SegmentationEngine  # noqa: E402

NUM_CLASSES = 3
CONF = 0.5
DATASET_DIR = MODEL_DIR / "datasets" / "v3"
REPORT_PATH = REPO_ROOT / "docs" / "onnx-parity.md"

# Reference numbers reported by the training notebook (test set, conf=0.5).
NOTEBOOK_IOU = {0: 0.9597, 1: 0.9099, 2: 0.9368}
NOTEBOOK_DICE = {0: 0.9772, 1: 0.9369, 2: 0.9530}


# --- Metrics: same logic as the training notebook -------------------------

def load_gt_mask(label_path: str, img_shape: tuple[int, int]) -> dict[int, np.ndarray]:
    """Rebuild per-class GT binary masks from a YOLOv8-seg label file."""
    h, w = img_shape
    masks = {c: np.zeros((h, w), dtype=np.uint8) for c in range(NUM_CLASSES)}
    if not os.path.exists(label_path):
        return masks
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            if len(parts) == 5:  # box-only label
                x_c, y_c, bw, bh = map(float, parts[1:5])
                x1, y1 = int((x_c - bw / 2) * w), int((y_c - bh / 2) * h)
                x2, y2 = int((x_c + bw / 2) * w), int((y_c + bh / 2) * h)
                cv2.rectangle(masks[cls_id], (x1, y1), (x2, y2), 1, -1)
                continue
            coords = np.array(list(map(float, parts[5:])))
            pts = np.stack([(coords[0::2] * w).astype(int), (coords[1::2] * h).astype(int)], axis=1)
            cv2.fillPoly(masks[cls_id], [pts], 1)
    return masks


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else 2 * np.logical_and(a, b).sum() / total


# --- Prediction paths ------------------------------------------------------

def pt_class_masks(model, img_path: str, img_shape: tuple[int, int]) -> dict[int, np.ndarray]:
    """Per-class union masks from the .pt model, replicating the notebook's get_pred_masks."""
    h, w = img_shape
    pred = {c: np.zeros((h, w), dtype=np.uint8) for c in range(NUM_CLASSES)}
    results = model(img_path, conf=CONF, imgsz=640, verbose=False)[0]
    if results.masks is None:
        return pred
    for i, box in enumerate(results.boxes.data.tolist()):
        *_, score, cls_id = box
        mask_i = results.masks.data[i].cpu().numpy()
        mask_i = (cv2.resize(mask_i, (w, h)) >= 0.5).astype(np.uint8)
        pred[int(cls_id)] = np.logical_or(pred[int(cls_id)], mask_i).astype(np.uint8)
    return pred


def onnx_class_masks(engine: SegmentationEngine, img_bgr: np.ndarray) -> dict[int, np.ndarray]:
    """Per-class union masks from the production ONNX engine."""
    h, w = img_bgr.shape[:2]
    pred = {c: np.zeros((h, w), dtype=np.uint8) for c in range(NUM_CLASSES)}
    for det in engine.predict(img_bgr, conf=CONF).detections:
        pred[det.class_id] = np.logical_or(pred[det.class_id], det.mask).astype(np.uint8)
    return pred


# --- Main -------------------------------------------------------------------

def main() -> None:
    from ultralytics import YOLO

    test_images = DATASET_DIR / "test" / "images"
    test_labels = DATASET_DIR / "test" / "labels"
    if not test_images.exists():
        sys.exit(f"Test set not found at {test_images}. Run download_dataset.py first.")

    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(str(test_images / ext))
    )
    print(f"Test images: {len(image_paths)}")

    pt_model = YOLO(str(MODEL_DIR / "weights" / "best.pt"))
    engine = SegmentationEngine(MODEL_DIR / "weights" / "best.onnx")

    acc = {  # per class accumulators
        c: {"pt_iou": [], "pt_dice": [], "ox_iou": [], "ox_dice": [], "agree": []}
        for c in range(NUM_CLASSES)
    }
    latencies = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        label_path = str(test_labels / (Path(img_path).stem + ".txt"))

        gt = load_gt_mask(label_path, (h, w))
        pt = pt_class_masks(pt_model, img_path, (h, w))

        t0 = time.perf_counter()
        ox = onnx_class_masks(engine, img)
        latencies.append((time.perf_counter() - t0) * 1000)

        for c in range(NUM_CLASSES):
            acc[c]["pt_iou"].append(iou(gt[c], pt[c]))
            acc[c]["pt_dice"].append(dice(gt[c], pt[c]))
            acc[c]["ox_iou"].append(iou(gt[c], ox[c]))
            acc[c]["ox_dice"].append(dice(gt[c], ox[c]))
            acc[c]["agree"].append(iou(pt[c], ox[c]))

    # --- Report ---
    rows, all_pass = [], True
    for c in range(NUM_CLASSES):
        m = {k: float(np.mean(v)) for k, v in acc[c].items()}
        d_iou, d_dice = m["ox_iou"] - m["pt_iou"], m["ox_dice"] - m["pt_dice"]
        ok = abs(d_iou) <= 0.01 and abs(d_dice) <= 0.01 and m["agree"] >= 0.99
        all_pass &= ok
        rows.append((CLASS_NAMES[c], m, d_iou, d_dice, ok))
        print(
            f"{CLASS_NAMES[c]:22s} IoU pt={m['pt_iou']:.4f} onnx={m['ox_iou']:.4f} (d{d_iou:+.4f}) | "
            f"Dice pt={m['pt_dice']:.4f} onnx={m['ox_dice']:.4f} (d{d_dice:+.4f}) | "
            f"agreement={m['agree']:.4f} | {'PASS' if ok else 'FAIL'}"
        )
    lat_mean, lat_p95 = float(np.mean(latencies)), float(np.percentile(latencies, 95))
    print(f"ONNX CPU latency: mean {lat_mean:.0f} ms, p95 {lat_p95:.0f} ms")
    print("RESULT:", "PASS" if all_pass else "FAIL")

    import onnxruntime
    import ultralytics

    lines = [
        "# ONNX Parity Report",
        "",
        "Equivalence between the academic model (`best.pt`, ultralytics "
        f"{ultralytics.__version__}) and the production inference path "
        f"(`best.onnx` + `backend/app/inference`, ONNX Runtime {onnxruntime.__version__}, CPU).",
        "",
        f"- Test set: {len(image_paths)} images (Roboflow v3, held out from training)",
        f"- Confidence threshold: {CONF} (same as the notebook's ablation)",
        "- Metrics computed with the notebook's per-class mask IoU/Dice methodology",
        "- *Agreement* = mask-to-mask IoU between the .pt and ONNX outputs (target >= 0.99)",
        "",
        "| Class | IoU .pt | IoU ONNX | ΔIoU | Dice .pt | Dice ONNX | ΔDice | Agreement | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m, d_iou, d_dice, ok in rows:
        lines.append(
            f"| {name} | {m['pt_iou']:.4f} | {m['ox_iou']:.4f} | {d_iou:+.4f} "
            f"| {m['pt_dice']:.4f} | {m['ox_dice']:.4f} | {d_dice:+.4f} "
            f"| {m['agree']:.4f} | {'✅ pass' if ok else '❌ fail'} |"
        )
    lines += [
        "",
        f"**Notebook reference (same protocol, GPU):** IoU "
        + " / ".join(f"{NOTEBOOK_IOU[c]:.4f}" for c in range(NUM_CLASSES))
        + ", Dice "
        + " / ".join(f"{NOTEBOOK_DICE[c]:.4f}" for c in range(NUM_CLASSES))
        + " (clavo / placa / prótesis).",
        "",
        f"**Production engine latency (CPU):** mean {lat_mean:.0f} ms, p95 {lat_p95:.0f} ms per image.",
        "",
        f"**Overall: {'PASS — the ONNX path is metrically equivalent to the original model.' if all_pass else 'FAIL — investigate before deploying.'}**",
        "",
        "_Generated by `model/validate_parity.py`._",
    ]
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
