# Build from the REPO ROOT so the ONNX weights can be copied in:
#   docker build -f backend/Dockerfile -t orthovision-api .
FROM python:3.11-slim

# opencv-python-headless still links against glib on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY model/weights/best.onnx ./model/best.onnx

ENV MODEL_PATH=/app/model/best.onnx \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Single worker on purpose: one shared ONNX session, and the target instance
# (Render free tier) has 512 MB of RAM. Render injects PORT at runtime.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
