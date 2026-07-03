import base64

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Inference
from app.db.session import get_db
from app.schemas.predict import (
    InferenceDetail,
    InferenceListResponse,
    InferenceSummary,
)

router = APIRouter()


def _thumb_b64(row: Inference) -> str | None:
    return base64.b64encode(row.thumbnail).decode() if row.thumbnail else None


@router.get("", response_model=InferenceListResponse)
def list_inferences(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> InferenceListResponse:
    total = db.scalar(select(func.count()).select_from(Inference)) or 0
    rows = db.scalars(
        select(Inference)
        .order_by(Inference.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    items = [
        InferenceSummary(
            inference_id=row.id,
            created_at=row.created_at,
            model_version=row.model_version,
            counts=row.counts,
            num_detections=len(row.detections),
            max_confidence=max((d["confidence"] for d in row.detections), default=None),
            thumbnail_b64=_thumb_b64(row),
        )
        for row in rows
    ]
    return InferenceListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{inference_id}", response_model=InferenceDetail)
def get_inference(inference_id: str, db: Session = Depends(get_db)) -> InferenceDetail:
    row = db.get(Inference, inference_id)
    if row is None:
        raise HTTPException(404, detail="Inference not found")
    return InferenceDetail(
        inference_id=row.id,
        created_at=row.created_at,
        model_version=row.model_version,
        image={
            "width": row.image_width,
            "height": row.image_height,
            "sha256": row.image_sha256,
        },
        timing_ms=row.timing_ms,
        detections=row.detections,
        counts=row.counts,
        conf_threshold=row.conf_threshold,
        thumbnail_b64=_thumb_b64(row),
    )
