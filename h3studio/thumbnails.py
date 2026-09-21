from __future__ import annotations

import hashlib
import os
import threading
import uuid
from pathlib import Path

from PIL import Image


_thumbnail_lock = threading.Lock()


def create_thumbnail(video: Path, cache_root: Path, job_id: str) -> Path:
    """Return a cached first-frame JPEG bound to this immutable MP4 revision."""
    import av

    if video.suffix.lower() != ".mp4" or not video.is_file():
        raise ValueError("Thumbnail source must be an existing MP4")
    stat = video.stat()
    revision = hashlib.sha256(
        f"{video.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:20]
    safe_job_id = "".join(character for character in job_id if character.isalnum() or character in "-_")
    if not safe_job_id:
        raise ValueError("Invalid thumbnail job id")
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved_cache = cache_root.resolve()
    target = (resolved_cache / f"{safe_job_id}-{revision}.jpg").resolve()
    if not target.is_relative_to(resolved_cache):
        raise ValueError("Unsafe thumbnail cache path")
    if target.is_file() and target.stat().st_size > 0:
        return target

    with _thumbnail_lock:
        if target.is_file() and target.stat().st_size > 0:
            return target
        with av.open(
            str(video), mode="r", format="mov", options={"protocol_whitelist": "file,pipe"},
        ) as container:
            if not container.streams.video:
                raise ValueError("MP4 has no video stream")
            frame = next(iter(container.decode(container.streams.video[0])), None)
            if frame is None:
                raise ValueError("MP4 has no decodable frame")
            image = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
            image.thumbnail((480, 270), Image.Resampling.LANCZOS)
            temporary = resolved_cache / f".{target.stem}-{uuid.uuid4().hex}.tmp"
            try:
                image.save(temporary, format="JPEG", quality=84, optimize=True, progressive=True)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)

        # A job normally has one immutable output. Remove stale revisions only
        # after the replacement is fully published.
        for stale in resolved_cache.glob(f"{safe_job_id}-*.jpg"):
            if stale != target:
                stale.unlink(missing_ok=True)
    return target
