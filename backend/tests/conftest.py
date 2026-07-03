import os
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
FIXTURES = Path(__file__).parent / "fixtures"

MODEL_PATH = REPO_ROOT / "model" / "weights" / "best.onnx"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="best.onnx not present — run model/export_onnx.py first",
)


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    """TestClient against a fresh app with an isolated SQLite database.

    Environment must be set before the app modules read their settings.
    """
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    os.environ["MODEL_PATH"] = str(MODEL_PATH)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"

    from app.core.config import get_settings

    get_settings.cache_clear()

    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
