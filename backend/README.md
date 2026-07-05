# OrthoVision AI — Backend

FastAPI + ONNX Runtime inference server. Torch-free by design.

## Run

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/uvicorn app.main:app --reload --host 0.0.0.0
```

Swagger UI at http://localhost:8000/docs. The `--host 0.0.0.0` flag exposes
the API on the LAN so a physical phone can reach it (omit it if you only need
localhost, e.g. when testing over USB with `adb reverse`).

## Test

```bash
.venv/Scripts/python -m pytest      # requires model/weights/best.onnx
```

Tests run the real ONNX model against fixture radiographs (no mocks): known
detections, the no-detections path, oversized/invalid uploads, and the
history flow.

## Docker

Build from the **repo root** so the weights are copied in:

```bash
docker build -f backend/Dockerfile -t orthovision-api .
docker run -p 8000:8000 orthovision-api
```

## Layout

```
app/
├── main.py              # app factory, CORS, lifespan (loads the ONNX session)
├── core/config.py       # env-driven settings
├── api/routes/          # health, predict, inferences
├── inference/           # engine (ORT session) + preprocess + postprocess
├── schemas/             # Pydantic response models
└── db/                  # SQLAlchemy models + session
```

See [../docs/api-contract.md](../docs/api-contract.md) for the full contract.
