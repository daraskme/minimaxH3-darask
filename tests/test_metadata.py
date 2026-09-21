import json
import subprocess
from pathlib import Path

import av
import pytest

from h3studio.metadata import _ffmpeg_path, ensure_within, finalize_video
from h3studio.video import inspect_video, read_embedded_metadata


def test_unsafe_output_path_is_rejected(tmp_path: Path):
    root = tmp_path / "outputs"
    root.mkdir()
    with pytest.raises(ValueError):
        ensure_within(root, tmp_path / "outside.mp4")


def test_mp4_metadata_roundtrip_preserves_audio(tmp_path: Path):
    ffmpeg = _ffmpeg_path()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    video = outputs / "sample.mp4"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=0.3", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video)],
        check=True,
    )
    metadata = {"prompt": ("雨と電車の音 🚃\\=;#\n" * 1200), "resolved": {"seed": 9007199254740991}}
    sidecar = finalize_video(video, outputs, metadata, ffmpeg)
    with av.open(str(video)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        embedded = json.loads(container.metadata["comment"])
        assert embedded == metadata
        assert container.metadata["description"] == metadata["prompt"]
    assert json.loads(sidecar.read_text(encoding="utf-8")) == metadata
    probe = inspect_video(video)
    assert probe["is_cfr"] is True
    assert probe["frame_count"] > 1
    assert probe["has_audio"] is True
    assert read_embedded_metadata(video) == metadata
