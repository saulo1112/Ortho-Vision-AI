"""API response models. All coordinates are normalized to [0, 1] relative to
the ORIGINAL uploaded image, so clients can scale to any render size."""

from datetime import datetime

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ImageMeta(BaseModel):
    width: int
    height: int
    sha256: str


class TimingMs(BaseModel):
    preprocess: int
    inference: int
    postprocess: int
    total: int


class DetectionOut(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    # One entry per blob; each ring is a list of [x, y] normalized points.
    polygons: list[list[list[float]]]


class PredictResponse(BaseModel):
    inference_id: str
    created_at: datetime
    model_version: str
    image: ImageMeta
    timing_ms: TimingMs
    detections: list[DetectionOut]
    counts: dict[str, int]


class InferenceSummary(BaseModel):
    inference_id: str
    created_at: datetime
    model_version: str
    counts: dict[str, int]
    num_detections: int
    max_confidence: float | None = None
    thumbnail_b64: str | None = Field(
        default=None, description="Small JPEG preview, base64-encoded"
    )


class InferenceDetail(PredictResponse):
    conf_threshold: float
    thumbnail_b64: str | None = None


class InferenceListResponse(BaseModel):
    items: list[InferenceSummary]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_s: int
