from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

from h3studio import seedvr2
from h3studio.engine import GenerationCancelled
from h3studio.metadata import _ffmpeg_path
from h3studio.video import inspect_video


def source_settings(**updates):
    source = {
        "width": 64, "height": 64, "frame_count": 5,
        "fps_numerator": 24_000, "fps_denominator": 1_001, "has_audio": True,
    }
    source.update(updates)
    return {"scale": 2, "seed": 7, "source": source}


def test_seedvr2_bounds_keep_exact_rational_rate():
    width, height, frames, rate = seedvr2._validate_settings(source_settings())
    assert (width, height, frames, rate) == (128, 128, 5, Fraction(24_000, 1_001))
    with pytest.raises(RuntimeError, match="only 2x"):
        seedvr2._validate_settings({**source_settings(), "scale": 3})
    with pytest.raises(RuntimeError, match="frame count"):
        seedvr2._validate_settings(source_settings(frame_count=seedvr2.MAX_SOURCE_FRAMES + 1))
    with pytest.raises(RuntimeError, match="staging limit"):
        seedvr2._validate_settings(source_settings(width=8192, height=8192, frame_count=20))


def test_png_sequence_requires_exact_shape_and_contiguous_names(tmp_path: Path):
    for index in range(3):
        Image.new("RGB", (128, 96), (index, 0, 0)).save(tmp_path / f"source_{index:06d}.png")
    frames = seedvr2._inspect_png_sequence(tmp_path, 3, 128, 96)
    assert [path.name for path in frames] == [f"source_{i:06d}.png" for i in range(3)]
    (tmp_path / "source_000001.png").rename(tmp_path / "source_000004.png")
    with pytest.raises(RuntimeError, match="non-contiguous"):
        seedvr2._inspect_png_sequence(tmp_path, 3, 128, 96)


def test_capability_receipt_requires_matching_real_execution(tmp_path: Path):
    runtime = {"torch": "test", "driver": "test"}
    receipt = tmp_path / seedvr2.RECEIPT_NAME
    receipt.write_text(json.dumps({
        "schema": seedvr2.RECEIPT_SCHEMA,
        "repository_revision": seedvr2.AUDITED_REVISION,
        "source_tree_sha256": seedvr2.UPSTREAM_TREE_SHA256,
        "source_sha256": {**seedvr2.UPSTREAM_SHA256, "h3studio_entry.py": seedvr2.LAUNCHER_SHA256},
        "model_sha256": seedvr2.MODEL_SHA256,
        "runtime": runtime,
        "inference": {"actual_neural_execution": True, "frames": 5},
    }), encoding="utf-8")
    assert seedvr2._valid_receipt(receipt, runtime) == (True, "")
    assert seedvr2._valid_receipt(receipt, {"driver": "changed"})[0] is False


def test_prerequisites_reject_changed_pinned_source(tmp_path: Path, monkeypatch):
    repo, models = tmp_path / "repo", tmp_path / "models"
    repo.mkdir()
    models.mkdir()
    upstream = repo / "inference_cli.py"
    launcher = repo / "h3studio_entry.py"
    model = models / "model.safetensors"
    upstream.write_bytes(b"audited upstream")
    launcher.write_bytes(b"audited launcher")
    model.write_bytes(b"audited model")
    (repo / "UPSTREAM_REVISION").write_text(seedvr2.AUDITED_REVISION, encoding="ascii")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    monkeypatch.setattr(seedvr2, "UPSTREAM_SHA256", {"inference_cli.py": digest(upstream)})
    monkeypatch.setattr(seedvr2, "LAUNCHER_SHA256", digest(launcher))
    monkeypatch.setattr(seedvr2, "UPSTREAM_TREE_SHA256", seedvr2._source_tree_sha256(repo))
    monkeypatch.setattr(seedvr2, "MODEL_SHA256", {"model.safetensors": digest(model)})
    monkeypatch.setattr(seedvr2, "configured_paths", lambda root: (repo, models, python))
    signature = ["runtime-one"]
    seen_signatures = []
    monkeypatch.setattr(seedvr2, "_runtime_signature", lambda *args: (True, signature[0]))
    monkeypatch.setattr(
        seedvr2, "_probe_runtime",
        lambda *args: (seen_signatures.append(args[-1]) or True, {"driver": args[-1]}),
    )
    assert seedvr2._prerequisites(tmp_path)[0]["installed"] is True
    signature[0] = "runtime-two"
    assert seedvr2._prerequisites(tmp_path)[0]["installed"] is True
    assert seen_signatures == ["runtime-one", "runtime-two"]
    upstream.write_bytes(b"changed upstream source")
    status = seedvr2._prerequisites(tmp_path)[0]
    assert status["installed"] is False
    assert "ソースのSHA256" in status["reason"]


def test_outer_export_preserves_frames_fractional_fps_and_audio(tmp_path: Path, monkeypatch):
    ffmpeg = _ffmpeg_path(None)
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24000/1001",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-frames:v", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    info = inspect_video(source)
    settings = {"scale": 2, "seed": 7, "source": info}
    fake_repo = tmp_path / "repo"
    fake_models = tmp_path / "models"
    fake_repo.mkdir()
    fake_models.mkdir()
    monkeypatch.setattr(seedvr2, "configured_paths", lambda root: (fake_repo, fake_models, Path("python")))

    def fake_run(command, cwd, environment, cancel, phase, progress, progress_value):
        if "h3studio_entry.py" in " ".join(command):
            render_root = Path(command[command.index("--output") + 1]) / "source"
            render_root.mkdir(parents=True)
            for index in range(5):
                Image.new("RGB", (128, 128), (index * 20, 0, 0)).save(
                    render_root / f"source_{index:06d}.png"
                )
        else:
            subprocess.run(command, cwd=cwd, env=environment, check=True)
        return []

    monkeypatch.setattr(seedvr2, "_run_cancellable", fake_run)
    result = seedvr2.SeedVR2Engine(tmp_path)._run_verified(
        source, output, settings, lambda *_: None, threading.Event(), {"driver": "test"},
    )
    probe = inspect_video(output)
    assert (probe["width"], probe["height"], probe["frame_count"]) == (128, 128, 5)
    assert Fraction(probe["fps_numerator"], probe["fps_denominator"]) == Fraction(24_000, 1_001)
    assert probe["has_audio"] is True
    assert result["timing"] == "exact_source_rational_cfr"


def test_silent_child_is_cancelled_and_reaped(tmp_path: Path):
    cancel = threading.Event()
    timer = threading.Timer(0.25, cancel.set)
    timer.start()
    try:
        with pytest.raises(GenerationCancelled, match="cancelled during test phase"):
            seedvr2._run_cancellable(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                tmp_path, os.environ.copy(), cancel, "test phase", lambda *_: None, 0.1,
            )
    finally:
        timer.cancel()


def test_failed_encode_removes_partial_publication(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"staged input")
    output = tmp_path / "output.mp4"
    fake_repo = tmp_path / "repo"
    fake_models = tmp_path / "models"
    fake_repo.mkdir()
    fake_models.mkdir()
    monkeypatch.setattr(seedvr2, "configured_paths", lambda root: (fake_repo, fake_models, Path("python")))

    def fake_run(command, cwd, environment, cancel, phase, progress, progress_value):
        if "h3studio_entry.py" in " ".join(command):
            render_root = Path(command[command.index("--output") + 1]) / "source"
            render_root.mkdir(parents=True)
            for index in range(5):
                Image.new("RGB", (128, 128)).save(render_root / f"source_{index:06d}.png")
            return []
        Path(command[-1]).write_bytes(b"partial")
        raise RuntimeError("encode failed")

    monkeypatch.setattr(seedvr2, "_run_cancellable", fake_run)
    with pytest.raises(RuntimeError, match="encode failed"):
        seedvr2.SeedVR2Engine(tmp_path)._run_verified(
            source, output, source_settings(), lambda *_: None, threading.Event(), {"driver": "test"},
        )
    assert not output.exists()
    assert not output.with_name(".output.seedvr2-working.mp4").exists()
