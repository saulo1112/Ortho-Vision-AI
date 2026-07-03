"""Download the Roboflow dataset (v3) used for training and parity validation.

Requires a Roboflow API key:
    ROBOFLOW_API_KEY=xxxx python download_dataset.py

The dataset lands in model/datasets/ (gitignored).
"""

import os
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
WORKSPACE = "proyecto-visin-computacional"
PROJECT = "segmentacion-dispositivos-medicos-itkfw"
VERSION = 3


def main() -> Path:
    target = DATASETS_DIR / f"v{VERSION}"
    if (target / "test" / "images").exists():
        print(f"Dataset already present at {target}")
        return target

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Set ROBOFLOW_API_KEY to download the dataset.")

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download("yolov8", location=str(target))
    print(f"Downloaded to {target}")
    return target


if __name__ == "__main__":
    main()
