from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any
import json


def inspect_video(path: Path) -> dict[str, Any]:
    """Validate an MP4 and return stable source facts without decoding it all."""
    import av

    with av.open(str(path), mode="r", format="mov", options={"protocol_whitelist": "file,pipe"}) as container:
        if not container.streams.video:
            raise ValueError("MP4 has no video stream")
        stream = container.streams.video[0]
        width, height = int(stream.width or 0), int(stream.height or 0)
        if width <= 0 or height <= 0 or width > 16384 or height > 16384:
            raise ValueError(f"unsupported video dimensions: {width}x{height}")
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        if not (0.1 <= fps <= 240.0):
            raise ValueError(f"unsupported video frame rate: {fps}")
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        if duration <= 0:
            raise ValueError("video duration is unavailable")
        rate_fraction = Fraction(rate).limit_denominator(100_000)
        frame_count = 0
        previous_timestamp: float | None = None
        is_cfr = True
        expected_delta = 1.0 / fps
        for frame in container.decode(stream):
            frame_count += 1
            if frame.pts is None or frame.time_base is None:
                is_cfr = False
                continue
            timestamp = float(frame.pts * frame.time_base)
            if previous_timestamp is not None:
                delta = timestamp - previous_timestamp
                if delta <= 0 or abs(delta - expected_delta) > max(0.0005, expected_delta * 0.02):
                    is_cfr = False
            previous_timestamp = timestamp
        if frame_count < 2:
            is_cfr = False
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "fps_numerator": rate_fraction.numerator,
            "fps_denominator": rate_fraction.denominator,
            "duration_seconds": duration,
            "has_audio": bool(container.streams.audio),
            "frame_count": frame_count,
            "is_cfr": is_cfr,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_embedded_metadata(path: Path, max_bytes: int = 512_000) -> dict[str, Any] | None:
    """Read bounded H3 JSON metadata as inert provenance data."""
    import av

    with av.open(str(path), mode="r", format="mov", options={"protocol_whitelist": "file,pipe"}) as container:
        raw = container.metadata.get("comment")
    if not raw or len(raw.encode("utf-8")) > max_bytes:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    return value
