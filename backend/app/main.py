import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, inferences, predict
from app.core.config import get_settings
from app.db.session import init_db
from app.inference.engine import SegmentationEngine

DESCRIPTION = """
Server-side instance segmentation of orthopedic implants in radiographs
(YOLOv8s-seg served via ONNX Runtime).

**Educational demo — not a medical device. Not for clinical use.**
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # One session per process, shared across requests (ORT run() is thread-safe).
    app.state.engine = SegmentationEngine(settings.model_path)
    app.state.started_at = time.time()
    init_db()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="OrthoVision AI API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, prefix="/v1", tags=["inference"])
    app.include_router(inferences.router, prefix="/v1/inferences", tags=["history"])
    return app


app = create_app()
