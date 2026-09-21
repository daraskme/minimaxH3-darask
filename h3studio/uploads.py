from __future__ import annotations

from pathlib import Path

from PIL import Image


MAX_IMAGE_EDGE = 16_384
MAX_IMAGE_PIXELS = 100_000_000


def validate_image(path: Path, expected_format: str) -> tuple[int, int]:
    """Decode and verify an uploaded conditioning image before model loading."""
    with Image.open(path) as image:
        width, height = image.size
        actual_format = (image.format or "").upper()
        if actual_format != expected_format.upper():
            raise ValueError(f"content is {actual_format or 'unknown'}, expected {expected_format}")
        if width <= 0 or height <= 0 or width > MAX_IMAGE_EDGE or height > MAX_IMAGE_EDGE:
            raise ValueError(f"unsupported image dimensions: {width}x{height}")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError(f"image has too many pixels: {width}x{height}")
        image.verify()
    return width, height
