"""SQLAlchemy models. Schema is created at startup via create_all — no
migrations for the MVP; switch to Alembic when the schema starts evolving."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Inference(Base):
    __tablename__ = "inferences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    image_sha256: Mapped[str] = mapped_column(String(64))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(80))
    conf_threshold: Mapped[float] = mapped_column(Float)
    # Full detection payload (no masks — polygons only), as returned to the client.
    detections: Mapped[list] = mapped_column(JSON)
    counts: Mapped[dict] = mapped_column(JSON)
    timing_ms: Mapped[dict] = mapped_column(JSON)
    # Small JPEG preview so the history screen stays visual without storing
    # the full radiograph (lighter and less sensitive).
    thumbnail: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
