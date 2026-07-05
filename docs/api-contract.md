# OrthoVision AI — API Contract

Base URL (local): `http://localhost:8000` · Interactive docs at `/docs` (Swagger UI).

All coordinates returned by the API (bounding boxes and polygon points) are
**normalized to `[0, 1]` relative to the original uploaded image**. Clients
multiply by their rendered width/height — no assumptions about the model's
internal 640×640 working size ever leak to the client.

## Endpoints

### `GET /health`

Liveness + model identity. Also used as the deploy health check.

```json
{ "status": "ok", "model_version": "orthovision-yolov8s-seg-1.0-onnx", "uptime_s": 812 }
```

### `POST /v1/predict`

`multipart/form-data` with a single `image` field (JPEG or PNG, ≤ 10 MB).
Optional query parameter `conf` (`0.05`–`0.95`, default `0.5` — the threshold
validated in the model's ablation study).

EXIF orientation is honored server-side, so phone photos of printed films
arrive correctly rotated. Images larger than 4096 px on the longest side are
downscaled before inference.

**200 — success** (including the no-detections case, which is not an error):

```json
{
  "inference_id": "3f6c1e0a-…",
  "created_at": "2026-07-03T19:24:31.512Z",
  "model_version": "orthovision-yolov8s-seg-1.0-onnx",
  "image": { "width": 1024, "height": 768, "sha256": "ab12…" },
  "timing_ms": { "preprocess": 14, "inference": 402, "postprocess": 21, "total": 437 },
  "detections": [
    {
      "id": 0,
      "class_id": 2,
      "class_name": "protesis_articular",
      "confidence": 0.9185,
      "bbox": { "x1": 0.3391, "y1": 0.0782, "x2": 0.6702, "y2": 0.5310 },
      "polygons": [ [[0.4531, 0.0798], [0.4482, 0.0865], [0.4413, 0.1092]] ]
    }
  ],
  "counts": { "protesis_articular": 1 }
}
```

Notes:

- `detections` carries **one entry per instance** — two plates in one film
  produce two entries with the same `class_name`. `counts` aggregates per class.
- `polygons` is a **list of rings** (a mask fragmented into several blobs
  yields one ring per blob). Each ring is a list of `[x, y]` points forming a
  closed polygon. Rings are simplified (Douglas-Peucker, ε = 0.2% of the
  contour perimeter); blobs smaller than 0.05% of the image are dropped as
  noise. Holes inside a mask are not represented (irrelevant for a translucent
  overlay).
- Classes: `0 = clavo_intramedular`, `1 = placa_atornillada`, `2 = protesis_articular`.

**Errors**

| Status | Meaning |
|---|---|
| 400 | Bytes are not a decodable JPEG/PNG (or empty upload) |
| 413 | Image exceeds 10 MB |
| 422 | `image` field missing, or `conf` out of range |

### `GET /v1/inferences?limit=20&offset=0`

Paginated history, newest first. `limit` ≤ 100.

```json
{
  "items": [
    {
      "inference_id": "3f6c1e0a-…",
      "created_at": "2026-07-03T19:24:31.512Z",
      "model_version": "orthovision-yolov8s-seg-1.0-onnx",
      "counts": { "protesis_articular": 1 },
      "num_detections": 1,
      "max_confidence": 0.9185,
      "thumbnail_b64": "/9j/4AAQSkZJRg…"
    }
  ],
  "total": 12,
  "limit": 20,
  "offset": 0
}
```

`thumbnail_b64` is a small (≤512 px) JPEG preview, base64-encoded. The full
radiograph is never stored — only its SHA-256, dimensions, and this thumbnail.

### `GET /v1/inferences/{inference_id}`

Full stored record: everything `POST /v1/predict` returned, plus
`conf_threshold` and `thumbnail_b64`. `404` if the id does not exist.

## Design decisions

- **Sync (def) route for prediction**: FastAPI executes it in a threadpool, so
  CPU-bound ONNX inference never blocks the event loop.
- **Polygons instead of raster masks**: a typical response is < 30 KB; the
  client draws with SVG at any resolution.
- **No auth for the MVP**: the API is a portfolio demo. The router split and
  the DB schema leave room to add API keys/JWT per route later.
