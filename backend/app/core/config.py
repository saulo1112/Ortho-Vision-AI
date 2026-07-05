"""Application settings, overridable via environment variables or .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces: our fields legitimately start with model_ (model_path,
    # model_version), which pydantic reserves by default.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )

    # Path to the ONNX weights. Dev default assumes uvicorn runs from backend/;
    # the Docker image sets MODEL_PATH=/app/model/best.onnx.
    model_path: str = "../model/weights/best.onnx"
    model_version: str = "orthovision-yolov8s-seg-1.0-onnx"

    database_url: str = "sqlite:///./orthovision.db"

    conf_threshold: float = 0.5
    iou_threshold: float = 0.7
    max_upload_bytes: int = 10 * 1024 * 1024
    # 512px keeps history/detail previews crisp on 3x phone screens while the
    # full radiograph is still never stored.
    thumbnail_side: int = 512

    cors_origins: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
