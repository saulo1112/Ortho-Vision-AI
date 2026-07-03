import time

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.schemas.predict import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=get_settings().model_version,
        uptime_s=round(time.time() - request.app.state.started_at),
    )
