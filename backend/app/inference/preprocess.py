"""Image decoding and YOLO-style preprocessing (EXIF fix, downscale, letterbox)."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

# Longest side accepted before server-side downscale. Keeps memory bounded on
# small instances while staying far above the model's 640px working size.
MAX_SIDE = 4096

IMG_SIZE = 640
PAD_COLOR = 114  # Ultralytics letterbox padding value


class ImageDecodeError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


def load_image(data: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR uint8 array.

    Applies EXIF orientation (phone photos of printed films arrive rotated)
    and caps the longest side at MAX_SIDE.
    """
    try:
        with Image.open(io.BytesIO(data)) as pil:
            pil = ImageOps.exif_transpose(pil)
            rgb = pil.convert("RGB")
            img = np.asarray(rgb)[:, :, ::-1].copy()  # canonical in-memory format is BGR
    except Exception as exc:  # PIL raises a zoo of exceptions on corrupt input
        raise ImageDecodeError(f"Could not decode image: {exc}") from exc

    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
        )
    return img


def letterbox(
    img: np.ndarray, new_shape: int = IMG_SIZE
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to new_shape x new_shape.

    Mirrors ultralytics LetterBox with auto=False (fixed square input, the
    configuration the ONNX graph was exported with).

    Returns (padded image, scale ratio, (left, top) padding in pixels).
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_w, new_h = round(w * r), round(h * r)

    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = (new_shape - new_w) / 2, (new_shape - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(PAD_COLOR, PAD_COLOR, PAD_COLOR),
    )
    return img, r, (left, top)


def to_tensor(img: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> RGB float32 NCHW in [0, 1]."""
    tensor = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor[None])
