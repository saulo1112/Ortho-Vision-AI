import hashlib
import uuid
from collections import Counter
from datetime import datetime, timezone

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Inference
from app.db.session import get_db
from app.inference.preprocess import ImageDecodeError, load_image
from app.schemas.predict import (
    BBox,
    DetectionOut,
    ImageMeta,
    PredictResponse,
    TimingMs,
)

router = APIRouter()


def _make_thumbnail(img, side: int) -> bytes:
    h, w = img.shape[:2]
    scale = side / max(h, w)
    small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


# Sync route on purpose: FastAPI runs it in a threadpool, so CPU-bound
# inference does not block the event loop.
@router.post("/predict", response_model=PredictResponse)
def predict(
    request: Request,
    image: UploadFile = File(...),
    conf: float | None = Query(
        None, ge=0.05, le=0.95,
        description="Confidence threshold; defaults to the server-configured 0.5",
    ),
    db: Session = Depends(get_db),
) -> PredictResponse:
    settings = get_settings()

    data = image.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, detail="Image exceeds the 10 MB limit")
    if not data:
        raise HTTPException(400, detail="Empty upload")

    try:
        img = load_image(data)
    except ImageDecodeError:
        raise HTTPException(400, detail="File is not a decodable JPEG/PNG image")

    conf_value = conf if conf is not None else settings.conf_threshold
    result = request.app.state.engine.predict(
        img, conf=conf_value, iou=settings.iou_threshold
    )

    detections = [
        DetectionOut(
            id=i,
            class_id=det.class_id,
            class_name=det.class_name,
            confidence=det.confidence,
            bbox=BBox(x1=det.bbox[0], y1=det.bbox[1], x2=det.bbox[2], y2=det.bbox[3]),
            polygons=det.polygons,
        )
        for i, det in enumerate(result.detections)
    ]
    counts = dict(Counter(d.class_name for d in detections))
    h, w = img.shape[:2]

    response = PredictResponse(
        inference_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        model_version=settings.model_version,
        image=ImageMeta(width=w, height=h, sha256=hashlib.sha256(data).hexdigest()),
        timing_ms=TimingMs(**result.timing_ms),
        detections=detections,
        counts=counts,
    )

    db.add(
        Inference(
            id=response.inference_id,
            created_at=response.created_at,
            image_sha256=response.image.sha256,
            image_width=w,
            image_height=h,
            model_version=settings.model_version,
            conf_threshold=conf_value,
            detections=[d.model_dump() for d in detections],
            counts=counts,
            timing_ms=result.timing_ms,
            thumbnail=_make_thumbnail(img, settings.thumbnail_side),
        )
    )
    db.commit()
    return response
