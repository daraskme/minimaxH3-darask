from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .engine import GenerationCancelled, StandaloneH3Engine
import json

from .metadata import embed_mp4_metadata, finalize_video
from .postprocess import VideoPostprocessEngine
from .store import JobStore, utcnow
from .video import inspect_video, sha256_file


class JobRunner:
    def __init__(self, config: Settings, store: JobStore):
        self.config = config
        self.store = store
        self.engine = StandaloneH3Engine(
            config.model_root, config.lora_root, config.uploads, config.latent_upscaler_root,
        )
        self.postprocess = VideoPostprocessEngine(config.ffmpeg)
        self._condition = threading.Condition()
        self._stop = False
        self._active_id: str | None = None
        self._active_kind: str | None = None
        self._cancel = threading.Event()
        self._control_revision = 0
        self._pending_control: dict[str, Any] | None = None
        self._active_control: dict[str, Any] | None = None
        self._pending_loras: list[dict[str, Any]] = []
        self._control_status: dict[str, Any] = {
            "state": "idle", "revision": 0, "kind": None, "progress": 0.0,
            "phase": "モデルは未読み込みです", "error": None,
        }
        self._thread = threading.Thread(target=self._loop, name="h3studio-worker", daemon=True)

    def start(self) -> None:
        self.store.recover_interrupted()
        self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._cancel.set()
            self._condition.notify_all()
        self._thread.join(timeout=5)

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def queue_model_load(self, settings: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            self._control_revision += 1
            payload = dict(settings)
            # An explicit load request is authoritative, including loras=[].
            # Previously a stale pending auto-apply could silently overwrite it.
            self._pending_loras = list(payload.get("loras", []))
            self._pending_control = {"revision": self._control_revision, "kind": "load", "settings": payload}
            self._control_status = {
                "state": "queued", "revision": self._control_revision, "kind": "load",
                "progress": 0.0, "phase": "モデル読み込みを待機しています", "error": None,
            }
            self._condition.notify_all()
            return self.control_status()

    def queue_loras(self, loras: list[dict[str, Any]]) -> dict[str, Any]:
        with self._condition:
            self._pending_loras = list(loras)
            self._control_revision += 1
            if self._pending_control is not None:
                if self._pending_control["kind"] == "load":
                    self._pending_control["settings"]["loras"] = list(loras)
                    kind, phase = "load", "モデルと最新のLoRA設定を待機しています"
                else:
                    self._pending_control["loras"] = list(loras)
                    kind, phase = "loras", "最新のLoRA設定を待機しています"
                self._pending_control["revision"] = self._control_revision
                state = "queued"
            elif (
                self._active_control is not None
                or self._active_kind == "generate"
                or (self._active_kind is None and self.engine.runtime_status()["loaded"])
            ):
                self._pending_control = {
                    "revision": self._control_revision, "kind": "loras", "loras": list(loras),
                }
                kind, state, phase = "loras", "queued", "LoRA設定の適用を待機しています"
            else:
                self._pending_control = None
                kind, state, phase = "loras", "pending_model", "モデル読み込み時にLoRAを適用します"
            self._control_status = {
                "state": state, "revision": self._control_revision, "kind": kind,
                "progress": 0.0, "phase": phase, "error": None,
            }
            self._condition.notify_all()
            return self.control_status()

    def control_status(self) -> dict[str, Any]:
        with self._condition:
            return {
                **self._control_status,
                "pending_loras": list(self._pending_loras),
                "runtime": self.engine.runtime_status(),
            }

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self.store.get(job_id)
            if job["status"] in {"completed", "failed", "cancelled"}:
                return job
            if job["status"] == "queued":
                return self.store.update(job_id, status="cancelled", progress=0, phase="cancelled", cancel_requested=1, finished_at=utcnow())
            if job_id != self._active_id:
                raise RuntimeError("Job is no longer owned by the active worker")
            self._cancel.set()
            return self.store.update(job_id, status="cancelling", cancel_requested=1, phase="キャンセルしています")

    def retry_finalize(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = self.store.get(job_id)
            if job["status"] != "failed" or not job.get("output_path") or not job.get("metadata_path"):
                raise RuntimeError("This job has no recoverable generated video and metadata")
            video = (self.config.outputs / job["output_path"]).resolve()
            sidecar = (self.config.outputs / job["metadata_path"]).resolve()
            if not video.is_relative_to(self.config.outputs.resolve()) or not sidecar.is_relative_to(self.config.outputs.resolve()):
                raise RuntimeError("Unsafe recovery path")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            embed_mp4_metadata(video, metadata, self.config.ffmpeg)
            return self.store.update(job_id, status="completed", progress=1.0, phase="完了", error=None, finished_at=utcnow())

    def _next(self) -> dict[str, Any] | None:
        for job in reversed(self.store.list(limit=500)):
            if job["status"] == "queued":
                return job
        return None

    def _loop(self) -> None:
        while True:
            with self._condition:
                if self._stop:
                    return
                control = self._pending_control
                if control is not None:
                    self._pending_control = None
                    self._active_control = control
                    self._cancel.clear()
                    self._control_status = {
                        "state": "running", "revision": control["revision"], "kind": control["kind"],
                        "progress": 0.01,
                        "phase": "モデルを読み込んでいます" if control["kind"] == "load" else "LoRAを適用しています",
                        "error": None,
                    }
                else:
                    control = None
                if control is not None:
                    job = None
                else:
                    job = self._next()
                if control is None and job is None:
                    self._condition.wait(timeout=1.0)
                    continue
                if control is not None:
                    self._active_id = None
                    self._active_kind = None
                else:
                    assert job is not None
                    self._active_id = job["id"]
                    self._active_kind = job.get("kind", "generate")
                    self._cancel.clear()
                    job = self.store.update(
                        job["id"], status="running", phase="モデルを準備しています",
                        progress=0.01, started_at=utcnow(),
                    )
            if control is not None:
                try:
                    self._run_control(control)
                finally:
                    with self._condition:
                        if self._active_control is control:
                            self._active_control = None
                continue
            assert job is not None
            self._run(job)
            with self._condition:
                self._active_id = None
                self._active_kind = None

    def _run_control(self, control: dict[str, Any]) -> None:
        revision = int(control["revision"])

        def progress(value: float, phase: str) -> None:
            if self._cancel.is_set():
                raise GenerationCancelled("cancelled during engine control")
            with self._condition:
                if self._control_status.get("revision") == revision:
                    self._control_status.update(progress=max(0.0, min(0.99, value)), phase=phase)

        try:
            if control["kind"] == "load":
                self.engine.preload(control["settings"], progress, self._cancel)
            else:
                if not self.engine.runtime_status()["loaded"]:
                    with self._condition:
                        if self._control_status.get("revision") == revision:
                            self._control_status.update(
                                state="pending_model", progress=0.0,
                                phase="モデル読み込み時にLoRAを適用します", error=None,
                            )
                    return
                self.engine.configure_loras(control["loras"], progress, self._cancel)
            with self._condition:
                # A newer request may already be queued. Keep its visible state.
                if self._control_status.get("revision") == revision:
                    self._control_status.update(
                        state="loaded", progress=1.0,
                        phase="モデルとLoRAを読み込みました", error=None,
                    )
        except Exception as exc:
            with self._condition:
                if self._control_status.get("revision") == revision:
                    self._control_status.update(
                        state="error", phase="読み込みに失敗しました",
                        error=f"{type(exc).__name__}: {exc}",
                    )

    def _run(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        started = utcnow()
        date_dir = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
        output_dir = self.config.outputs / date_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        kind = job.get("kind", "generate")
        output = output_dir / f"h3-{kind}-{job_id[:8]}.mp4"

        def progress(value: float, phase: str) -> None:
            if self._cancel.is_set():
                raise GenerationCancelled("cancelled")
            self.store.update(job_id, progress=max(0.0, min(0.98, value)), phase=phase)

        generated = False
        sidecar = output.with_suffix(".json")
        try:
            if kind == "generate":
                engine_info = self.engine.generate(job["resolved"], output, progress, self._cancel)
            else:
                # The queue is deliberately shared. Free the large H3 pipeline
                # before loading/running a video model on the same GPU.
                self.engine.release()
                source_info = job["resolved"]["source"]
                source_root = self.config.outputs if source_info["type"] == "output" else self.config.uploads
                source = (source_root / source_info["path"]).resolve()
                if not source.is_relative_to(source_root.resolve()) or not source.is_file():
                    raise FileNotFoundError("Postprocess source video is missing")
                if sha256_file(source) != source_info["sha256"]:
                    raise RuntimeError("Postprocess source changed after the job was queued")
                engine_info = self.postprocess.run(kind, job["resolved"], source, output, progress, self._cancel)
                progress(0.97, "出力を検証しています")
                probe = inspect_video(output)
                expected_width, expected_height = int(job["resolved"]["width"]), int(job["resolved"]["height"])
                expected_fps = float(job["resolved"]["fps"])
                expected_duration = float(job["resolved"]["duration_seconds"])
                if (probe["width"], probe["height"]) != (expected_width, expected_height):
                    raise RuntimeError(f"Output geometry mismatch: {probe['width']}x{probe['height']} != {expected_width}x{expected_height}")
                if abs(float(probe["fps"]) - expected_fps) > 0.001:
                    raise RuntimeError(f"Output FPS mismatch: {probe['fps']} != {expected_fps}")
                if abs(float(probe["duration_seconds"]) - expected_duration) > max(0.05, 1.0 / expected_fps):
                    raise RuntimeError(f"Output duration mismatch: {probe['duration_seconds']} != {expected_duration}")
                if bool(probe["has_audio"]) != bool(source_info["has_audio"]):
                    raise RuntimeError("Output audio-stream preservation check failed")
                if kind == "interpolate":
                    expected_frames = int(source_info["frame_count"]) * int(job["resolved"]["factor"])
                elif job["resolved"]["method"] in {"realesrgan_x2plus", "seedvr2_3b"}:
                    expected_frames = int(source_info["frame_count"])
                else:
                    expected_frames = round(expected_duration * expected_fps)
                if abs(int(probe["frame_count"]) - expected_frames) > (1 if job["resolved"]["method"] == "lanczos" else 0):
                    raise RuntimeError(f"Output frame count mismatch: {probe['frame_count']} != {expected_frames}")
                engine_info["output_validation"] = probe
            generated = True
            if self._cancel.is_set():
                raise GenerationCancelled("cancelled")
            prompt = job["resolved"].get("prompt", "")
            if not prompt and kind != "generate":
                source_provenance = job["resolved"].get("source") or {}
                origin = source_provenance.get("origin") or {}
                embedded = origin.get("embedded_metadata") or origin.get("embedded_h3_metadata") or {}
                prompt = str(
                    embedded.get("prompt")
                    or (origin.get("request") or {}).get("prompt")
                    or ""
                )[:20000]
            metadata = {
                "schema": "h3studio.generation/v1" if kind == "generate" else "h3studio.postprocess/v1",
                "job_id": job_id,
                "kind": kind,
                "prompt": prompt,
                "requested": job["request"],
                "resolved": job["resolved"],
                "provenance": job["resolved"].get("source"),
                "engine": engine_info,
                "started_at": started,
                "finished_at": utcnow(),
            }
            sidecar = finalize_video(output, self.config.outputs, metadata, self.config.ffmpeg)
            with self._condition:
                if self._cancel.is_set():
                    output.unlink(missing_ok=True)
                    sidecar.unlink(missing_ok=True)
                    raise GenerationCancelled("cancelled during metadata finalization")
                self.store.update(
                    job_id, status="completed", progress=1.0, phase="完了",
                    output_path=str(output.relative_to(self.config.outputs).as_posix()),
                    metadata_path=str(sidecar.relative_to(self.config.outputs).as_posix()), finished_at=metadata["finished_at"],
                )
        except GenerationCancelled:
            output.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            self.store.update(job_id, status="cancelled", phase="キャンセル済み", finished_at=utcnow())
        except Exception as exc:
            if not generated:
                output.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
            message = f"{type(exc).__name__}: {exc}"
            details = traceback.format_exc(limit=12)
            recovery = {}
            if generated and output.is_file():
                recovery["output_path"] = str(output.relative_to(self.config.outputs).as_posix())
                if sidecar.is_file():
                    recovery["metadata_path"] = str(sidecar.relative_to(self.config.outputs).as_posix())
            self.store.update(
                job_id, status="failed", phase="後処理に失敗" if generated else "失敗",
                error=f"{message}\n{details}", finished_at=utcnow(), **recovery,
            )
