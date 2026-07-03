"""End-to-end API tests running the real ONNX model (no mocks)."""

import hashlib
import io

from PIL import Image

from .conftest import FIXTURES, requires_model

RADIOGRAPH_1 = FIXTURES / "radiograph_1.jpg"  # known: protesis_articular ~0.92
RADIOGRAPH_2 = FIXTURES / "radiograph_2.jpg"  # known: clavo_intramedular ~0.87


def _post_image(client, data: bytes, filename="xray.jpg"):
    return client.post("/v1/predict", files={"image": (filename, data, "image/jpeg")})


def _gray_jpeg(size=(640, 640)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


@requires_model
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "onnx" in body["model_version"]


@requires_model
def test_predict_known_radiograph(client):
    data = RADIOGRAPH_1.read_bytes()
    res = _post_image(client, data)
    assert res.status_code == 200
    body = res.json()

    assert body["image"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert body["counts"] == {"protesis_articular": 1}

    (det,) = body["detections"]
    assert det["class_name"] == "protesis_articular"
    assert det["confidence"] > 0.85
    assert len(det["polygons"]) >= 1
    assert len(det["polygons"][0]) >= 3
    # every coordinate normalized to the original image
    for ring in det["polygons"]:
        assert all(0.0 <= v <= 1.0 for point in ring for v in point)
    bbox = det["bbox"]
    assert 0.0 <= bbox["x1"] < bbox["x2"] <= 1.0
    assert 0.0 <= bbox["y1"] < bbox["y2"] <= 1.0
    assert body["timing_ms"]["total"] > 0


@requires_model
def test_predict_no_detections_is_200(client):
    res = _post_image(client, _gray_jpeg())
    assert res.status_code == 200
    body = res.json()
    assert body["detections"] == []
    assert body["counts"] == {}


@requires_model
def test_predict_rejects_non_image(client):
    res = _post_image(client, b"definitely not an image", filename="notes.txt")
    assert res.status_code == 400


@requires_model
def test_predict_rejects_oversized_upload(client):
    res = _post_image(client, b"\xff" * (10 * 1024 * 1024 + 1))
    assert res.status_code == 413


@requires_model
def test_predict_missing_field_is_422(client):
    assert client.post("/v1/predict").status_code == 422


@requires_model
def test_history_flow(client):
    first = _post_image(client, RADIOGRAPH_2.read_bytes()).json()

    listing = client.get("/v1/inferences?limit=5").json()
    assert listing["total"] >= 1
    newest = listing["items"][0]
    assert newest["inference_id"] == first["inference_id"]
    assert newest["counts"] == {"clavo_intramedular": 1}
    assert newest["thumbnail_b64"]  # visual history without storing the full film

    detail = client.get(f"/v1/inferences/{first['inference_id']}").json()
    assert detail["detections"] == first["detections"]
    assert detail["conf_threshold"] == 0.5

    assert client.get("/v1/inferences/nonexistent-id").status_code == 404
