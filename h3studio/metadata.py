from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _ffmpeg_path(configured: str | None = None) -> str:
    if configured:
        return configured
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def ensure_within(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Unsafe output path: {candidate}")
    return candidate


def write_sidecar(video_path: Path, metadata: dict[str, Any]) -> Path:
    sidecar = video_path.with_suffix(".json")
    temporary = sidecar.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, sidecar)
    return sidecar


def embed_mp4_metadata(video_path: Path, metadata: dict[str, Any], ffmpeg: str | None = None) -> None:
    """Atomically add UTF-8 metadata while stream-copying every video/audio stream."""
    payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    prompt = str(metadata.get("prompt", ""))
    fd, raw = tempfile.mkstemp(prefix=f".{video_path.stem}-", suffix=".mp4", dir=video_path.parent)
    os.close(fd)
    temporary = Path(raw)
    metadata_file: Path | None = None
    try:
        meta_fd, meta_raw = tempfile.mkstemp(prefix=f".{video_path.stem}-", suffix=".ffmeta", dir=video_path.parent)
        os.close(meta_fd)
        metadata_file = Path(meta_raw)

        def escape(value: str) -> str:
            value = value.replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
            return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\\n")

        metadata_file.write_text(
            ";FFMETADATA1\n" + f"comment={escape(payload)}\n" + f"description={escape(prompt)}\n",
            encoding="utf-8", newline="\n",
        )
        command = [
            _ffmpeg_path(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path), "-f", "ffmetadata", "-i", str(metadata_file),
            "-map", "0", "-map_metadata", "1", "-c", "copy",
            "-movflags", "+use_metadata_tags", str(temporary),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
        if completed.returncode != 0 or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg metadata remux failed: {completed.stderr.strip()}")
        os.replace(temporary, video_path)
    finally:
        temporary.unlink(missing_ok=True)
        if metadata_file is not None:
            metadata_file.unlink(missing_ok=True)


def finalize_video(video_path: Path, output_root: Path, metadata: dict[str, Any], ffmpeg: str | None = None) -> Path:
    safe_video = ensure_within(output_root, video_path)
    # Write the recovery record first. If remux fails, the expensive generated
    # MP4 and exact metadata remain available for a later retry.
    sidecar = write_sidecar(safe_video, metadata)
    embed_mp4_metadata(safe_video, metadata, ffmpeg)
    return sidecar
