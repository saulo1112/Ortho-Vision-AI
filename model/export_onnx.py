"""Export best.pt (YOLOv8s-seg) to ONNX for the production backend.

Fixed batch=1 and 640x640 input (dynamic=False): the serving path always
letterboxes to 640, and a static graph keeps the export surface small.

Usage:
    python export_onnx.py
"""

from pathlib import Path

import numpy as np

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def main() -> None:
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS_DIR / "best.pt"))
    exported = model.export(format="onnx", opset=12, imgsz=640, simplify=True, dynamic=False)
    print(f"Exported: {exported}")

    # Smoke check: load with ONNX Runtime and verify output signatures.
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(WEIGHTS_DIR / "best.onnx"), providers=["CPUExecutionProvider"]
    )
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: dummy})
    for meta, out in zip(session.get_outputs(), outputs):
        print(f"  {meta.name}: {out.shape}")
    assert outputs[0].shape == (1, 39, 8400), "unexpected detection head shape"
    assert outputs[1].shape == (1, 32, 160, 160), "unexpected prototype shape"
    print("ONNX Runtime smoke check passed.")


if __name__ == "__main__":
    main()
