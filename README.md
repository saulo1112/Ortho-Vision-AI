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

# Optional: reproduce the parity report (requires a Roboflow API key)
cp .env.example .env && $EDITOR .env           # set ROBOFLOW_API_KEY (gitignored)
.venv/Scripts/python download_dataset.py
.venv/Scripts/python validate_parity.py        # writes docs/onnx-parity.md
```

### 2. Backend

```bash
cd backend
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# --host 0.0.0.0 exposes the API on your LAN so a physical phone can reach it
.venv/Scripts/uvicorn app.main:app --reload --host 0.0.0.0   # http://localhost:8000/docs
.venv/Scripts/python -m pytest                               # 7 end-to-end tests
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
npm run android       # dev build on a USB-connected Android device
```

> Note: Expo Go only supports a single SDK version, so an outdated Expo Go
> from the store may refuse this project (SDK 57). The dev build
> (`npm run android`) is the standard workflow here and enables all native
> modules. Requires Android Studio / SDK; always run it from `mobile/`, not
> the repo root.

#### Test on a physical device

The app probes candidate backend URLs at startup (`/health`) and uses the
first one that answers, in this order:

1. `expo.extra.apiUrl` from `mobile/app.json` — set this for a deployed
   backend (e.g. your Render URL).
2. `http://localhost:8000` — works over **USB**: `npm run android` runs
   `adb reverse tcp:8000 tcp:8000`, which tunnels the phone's port 8000 to
   your dev machine. Immune to firewalls and multi-adapter setups.
3. `http://<metro-host-ip>:8000` — same **Wi-Fi** network. Requires the
   backend started with `--host 0.0.0.0` and Windows Firewall allowing
   inbound TCP 8000 (`netsh advfirewall firewall add rule
   name="OrthoVision API" dir=in action=allow protocol=TCP localport=8000`,
   from an elevated shell).

Either way, start the backend first (`npm run backend` from the repo root).

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
