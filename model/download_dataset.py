"""Download the Roboflow dataset (v3) used for training and parity validation.

Requires a Roboflow API key, read from the environment (never hardcoded).
Either export it inline:
    ROBOFLOW_API_KEY=xxxx python download_dataset.py
or copy .env.example to .env (gitignored) and fill in ROBOFLOW_API_KEY — it is
loaded automatically via python-dotenv.

The dataset lands in model/datasets/ (gitignored).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

MODEL_DIR = Path(__file__).resolve().parent
load_dotenv(MODEL_DIR / ".env")

DATASETS_DIR = MODEL_DIR / "datasets"
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
        sys.exit(
            "Set ROBOFLOW_API_KEY (env var or model/.env, see .env.example) "
            "to download the dataset."
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    project.version(VERSION).download("yolov8", location=str(target))
    print(f"Downloaded to {target}")
    return target


if __name__ == "__main__":
    main()
