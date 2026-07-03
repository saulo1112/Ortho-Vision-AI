# OrthoVision AI

> Instance segmentation of orthopedic implants in radiographs — from a trained
> YOLOv8s-seg model to a deployed API and a native-feeling mobile app.

OrthoVision AI takes a research segmentation model and productizes it end to
end: export to ONNX with a **documented equivalence guarantee**, a torch-free
FastAPI inference server in Docker, and an Expo mobile client that overlays the
predicted masks on the X-ray with per-instance confidence.

**⚠️ Educational / portfolio demo — not a medical device. Not for clinical use.**

The model segments three implant classes:

| Class | Label |
|---|---|
| `clavo_intramedular` | Intramedullary nail |
| `placa_atornillada` | Screwed plate |
| `protesis_articular` | Joint prosthesis |

---

## Highlights

- **Validated ONNX export.** The `.pt → ONNX` conversion is not taken on faith:
  a script runs both pipelines over the held-out test set and certifies they
  agree. Mask-to-mask agreement ≥ 0.996, per-class IoU/Dice delta ≤ 0.001.
  See [docs/onnx-parity.md](docs/onnx-parity.md).
- **Torch-free serving.** YOLOv8-seg post-processing (NMS, prototype masks,
  polygonization) is re-implemented in numpy + OpenCV, so the Docker image is
  ~300 MB and fits a 512 MB free-tier instance.
- **Premium mobile UI.** iOS-first design with native Liquid Glass, a clinical
  dark palette, and SVG mask overlays that scale to any screen.
- **Honest engineering.** Normalized coordinates, no-detection handling, EXIF
  correction, image-size limits, model versioning, and a medical disclaimer are
  all first-class, not afterthoughts.

## Results

Test set: 36 held-out radiographs (Roboflow v3), confidence threshold 0.5.

| Class | IoU (ONNX) | Dice (ONNX) | Agreement vs `.pt` |
|---|---|---|---|
| Intramedullary nail | 0.9596 | 0.9771 | 0.9990 |
| Screwed plate | 0.9098 | 0.9368 | 0.9968 |
| Joint prosthesis | 0.9377 | 0.9535 | 0.9974 |

Production engine latency (CPU): ~450 ms mean per image. Full report:
[docs/onnx-parity.md](docs/onnx-parity.md).

## Architecture

```
Expo app  ──POST /v1/predict (multipart)──►  FastAPI + ONNX Runtime  ──►  SQLite/Postgres
    ▲                                              │
    └────────── polygons + confidence ◄────────────┘
```

Details and diagram: [docs/architecture.md](docs/architecture.md). API contract:
[docs/api-contract.md](docs/api-contract.md).

```
ortho-vision-ai/
├── model/      # ONNX export + parity validation (torch, offline)
├── backend/    # FastAPI + ONNX Runtime, Dockerized (torch-free)
├── mobile/     # Expo / React Native app
└── docs/       # architecture, API contract, parity report
```

## Quickstart

### 1. Model → ONNX (once)

```bash
cd model
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python export_onnx.py            # writes weights/best.onnx

# Optional: reproduce the parity report
ROBOFLOW_API_KEY=xxxx .venv/Scripts/python download_dataset.py
.venv/Scripts/python validate_parity.py        # writes docs/onnx-parity.md
```

### 2. Backend

```bash
cd backend
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/uvicorn app.main:app --reload    # http://localhost:8000/docs
.venv/Scripts/python -m pytest                 # 7 end-to-end tests
```

Or with Docker (build from the repo root so the weights are included):

```bash
docker build -f backend/Dockerfile -t orthovision-api .
docker run -p 8000:8000 orthovision-api
```

### 3. Mobile

```bash
cd mobile
npm install
npx expo start        # scan the QR with Expo Go (Android/iOS)
```

On a physical device the app auto-discovers the backend at
`http://<your-dev-machine-ip>:8000` (same Wi-Fi). For a deployed backend, set
`expo.extra.apiUrl` in `mobile/app.json`.

## Deployment

The backend ships to [Render](https://render.com) via `render.yaml` (Docker web
service + managed Postgres). Point Render at this repo and it provisions both
from the blueprint; `DATABASE_URL` is injected automatically.

## Tech stack

**Model:** YOLOv8s-seg (ultralytics 8.0.196), ONNX Runtime.
**Backend:** FastAPI, ONNX Runtime, OpenCV, SQLAlchemy, Pydantic, Docker.
**Mobile:** Expo (SDK 57), Expo Router, react-native-svg, expo-glass-effect, TypeScript.

## Credits

Trained model and original research notebook by *Grupo 3 — Proyecto Final Visión
Computacional con Deep Learning*. Dataset via Roboflow. Productization,
backend, mobile app, and ONNX validation in this repository.
