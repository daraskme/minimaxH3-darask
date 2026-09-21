import threading
import time
from pathlib import Path

from h3studio.config import Settings
from h3studio.runner import JobRunner
from h3studio.store import JobStore


class BlockingEngine:
    def __init__(self):
        self.started = threading.Event()

    def generate(self, settings, output_path, progress, cancel):
        self.started.set()
        while not cancel.wait(0.01):
            progress(0.2, "test")
        from h3studio.engine import GenerationCancelled
        raise GenerationCancelled()


class CompletedEngine:
    def generate(self, settings, output_path, progress, cancel):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"expensive-video")
        return {"name": "test"}


class ControlEngine:
    def __init__(self):
        self.loaded = False
        self.model = None
        self.loras = []
        self.calls = []

    def runtime_status(self):
        return {"loaded": self.loaded, "model": self.model, "loras": self.loras}

    def preload(self, settings, progress, cancel):
        self.calls.append(("load", settings["model"], list(settings["loras"])))
        progress(0.5, "test load")
        self.loaded = True
        self.model = settings["model"]
        self.loras = [item for item in settings["loras"] if item.get("enabled", True)]
        return self.runtime_status()

    def configure_loras(self, loras, progress, cancel):
        self.calls.append(("loras", list(loras)))
        progress(0.5, "test loras")
        self.loras = [item for item in loras if item.get("enabled", True)]
        return self.runtime_status()


class BlockingControlEngine(ControlEngine):
    def __init__(self, block_kind):
        super().__init__()
        self.block_kind = block_kind
        self.started = threading.Event()
        self.release = threading.Event()

    def _block(self, kind):
        if self.block_kind == kind and not self.started.is_set():
            self.started.set()
            assert self.release.wait(2)

    def preload(self, settings, progress, cancel):
        self.calls.append(("load", settings["model"], list(settings["loras"])))
        self._block("load")
        self.loaded = True
        self.model = settings["model"]
        self.loras = [item for item in settings["loras"] if item.get("enabled", True)]
        return self.runtime_status()

    def configure_loras(self, loras, progress, cancel):
        self.calls.append(("loras", list(loras)))
        self._block("loras")
        self.loras = [item for item in loras if item.get("enabled", True)]
        return self.runtime_status()


class BlockingGenerationControlEngine(ControlEngine):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, settings, output_path, progress, cancel):
        self.calls.append(("generate", settings["model"]))
        self.started.set()
        assert self.release.wait(2)
        self.loaded = True
        self.model = settings["model"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")
        return {"name": "test"}


def test_targeted_running_cancel_finishes_cancelled(tmp_path: Path):
    config = Settings(outputs=tmp_path / "outputs", uploads=tmp_path / "uploads", state=tmp_path / "state.sqlite3")
    config.ensure_dirs()
    store = JobStore(config.state)
    runner = JobRunner(config, store)
    fake = BlockingEngine()
    runner.engine = fake
    request = {"prompt": "x", "model": "m"}
    resolved = {**request, "width": 32, "height": 32, "frames": 124}
    store.create("job-1", request, resolved)
    runner.start()
    try:
        assert fake.started.wait(2)
        result = runner.cancel("job-1")
        # A fast cooperative worker may publish the terminal state before the
        # cancellation response is read; both responses acknowledge this job.
        assert result["status"] in {"cancelling", "cancelled"}
        deadline = time.time() + 2
        while time.time() < deadline and store.get("job-1")["status"] != "cancelled":
            time.sleep(0.01)
        assert store.get("job-1")["status"] == "cancelled"
    finally:
        runner.stop()


def test_queued_cancel_never_reaches_engine(tmp_path: Path):
    config = Settings(outputs=tmp_path / "outputs", uploads=tmp_path / "uploads", state=tmp_path / "state.sqlite3")
    config.ensure_dirs()
    store = JobStore(config.state)
    runner = JobRunner(config, store)
    store.create("job-2", {"prompt": "x", "model": "m"}, {"prompt": "x", "model": "m"})
    assert runner.cancel("job-2")["status"] == "cancelled"
    runner.start()
    try:
        time.sleep(0.05)
        assert store.get("job-2")["status"] == "cancelled"
    finally:
        runner.stop()


def test_postprocess_failure_preserves_video_and_sidecar_for_retry(tmp_path: Path, monkeypatch):
    config = Settings(outputs=tmp_path / "outputs", uploads=tmp_path / "uploads", state=tmp_path / "state.sqlite3")
    config.ensure_dirs()
    store = JobStore(config.state)
    runner = JobRunner(config, store)
    runner.engine = CompletedEngine()

    def fail_after_sidecar(video, output_root, metadata, ffmpeg):
        sidecar = video.with_suffix(".json")
        sidecar.write_text(__import__("json").dumps(metadata), encoding="utf-8")
        raise RuntimeError("remux failed")

    monkeypatch.setattr("h3studio.runner.finalize_video", fail_after_sidecar)
    request = {"prompt": "x", "model": "m"}
    resolved = {**request, "width": 32, "height": 32, "frames": 124}
    store.create("job-3", request, resolved)
    runner.start()
    try:
        deadline = time.time() + 2
        while time.time() < deadline and store.get("job-3")["status"] not in {"failed", "completed"}:
            time.sleep(0.01)
        job = store.get("job-3")
        assert job["status"] == "failed"
        assert job["phase"] == "後処理に失敗"
        assert (config.outputs / job["output_path"]).read_bytes() == b"expensive-video"
        assert (config.outputs / job["metadata_path"]).is_file()
    finally:
        runner.stop()


def test_store_persists_job_kind(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job = store.create(
        "post-1", {"method": "lanczos", "scale": 2},
        {"kind": "upscale", "method": "lanczos", "scale": 2}, kind="upscale",
    )
    assert job["kind"] == "upscale"


def test_explicit_model_load_overrides_stale_pending_loras(tmp_path: Path):
    config = Settings(outputs=tmp_path / "outputs", uploads=tmp_path / "uploads", state=tmp_path / "state.sqlite3")
    config.ensure_dirs()
    runner = JobRunner(config, JobStore(config.state))
    fake = ControlEngine(); runner.engine = fake
    first = [{"path": "a.safetensors", "weight": 0.5, "enabled": True}]
    assert runner.queue_loras(first)["state"] == "pending_model"
    assert fake.calls == []
    runner.queue_model_load({"model": "MiniMax-H3", "attention": "sdpa", "memory_profile": "auto", "loras": []})
    runner.start()
    try:
        deadline = time.time() + 2
        while time.time() < deadline and runner.control_status()["state"] not in {"loaded", "error"}:
            time.sleep(0.01)
        assert runner.control_status()["state"] == "loaded"
        assert fake.calls == [("load", "MiniMax-H3", [])]
    finally:
        runner.stop()


def test_latest_lora_edit_survives_an_active_model_load(tmp_path: Path):
    config = Settings(outputs=tmp_path / "outputs", uploads=tmp_path / "uploads", state=tmp_path / "state.sqlite3")
    config.ensure_dirs()
    runner = JobRunner(config, JobStore(config.state))
    fake = BlockingControlEngine("load"); runner.engine = fake
    initial = [{"path": "initial.safetensors", "weight": 1.0, "enabled": True}]
    latest = [{"path": "latest.safetensors", "weight": 0.7, "enabled": True}]
    runner.queue_model_load({"model": "MiniMax-H3", "attention": "sdpa", "memory_profile": "auto", "loras": initial})
    runner.start()
    try:
        assert fake.started.wait(2)
        assert runner.queue_loras([{"path": "middle.safetensors", "weight": 0.5, "enabled": True}])["state"] == "queued"
        assert runner.queue_loras(latest)["state"] == "queued"
        fake.release.set()
        deadline = time.time() + 2
        while time.time() < deadline and len(fake.calls) < 2:
            time.sleep(0.01)
        assert fake.calls == [("load", "MiniMax-H3", initial), ("loras", latest)]
        assert runner.control_status()["state"] == "loaded"
    finally:
        fake.release.set(); runner.stop()


def test_latest_lora_edit_survives_an_active_lora_apply(tmp_path: Path):
    config = Settings(outputs=tmp_path / "outputs", uploads=tmp_path / "uploads", state=tmp_path / "state.sqlite3")
    config.ensure_dirs()
    runner = JobRunner(config, JobStore(config.state))
    fake = BlockingControlEngine("loras"); fake.loaded = True; fake.model = "MiniMax-H3"; runner.engine = fake
    first = [{"path": "first.safetensors", "weight": 1.0, "enabled": True}]
    latest = [{"path": "latest.safetensors", "weight": 0.25, "enabled": True}]
    runner.queue_loras(first)
    runner.start()
    try:
        assert fake.started.wait(2)
        runner.queue_loras([{"path": "middle.safetensors", "weight": 0.5, "enabled": True}])
        runner.queue_loras(latest)
        fake.release.set()
        deadline = time.time() + 2
        while time.time() < deadline and len(fake.calls) < 2:
            time.sleep(0.01)
        assert fake.calls == [("loras", first), ("loras", latest)]
        assert runner.control_status()["state"] == "loaded"
    finally:
        fake.release.set(); runner.stop()


def test_latest_lora_edit_during_generation_applies_after_job(tmp_path: Path, monkeypatch):
    config = Settings(outputs=tmp_path / "outputs", uploads=tmp_path / "uploads", state=tmp_path / "state.sqlite3")
    config.ensure_dirs()
    store = JobStore(config.state)
    runner = JobRunner(config, store)
    fake = BlockingGenerationControlEngine(); runner.engine = fake

    def finalize(video, output_root, metadata, ffmpeg):
        sidecar = video.with_suffix(".json")
        sidecar.write_text(__import__("json").dumps(metadata), encoding="utf-8")
        return sidecar

    monkeypatch.setattr("h3studio.runner.finalize_video", finalize)
    store.create(
        "job-control", {"prompt": "x", "model": "MiniMax-H3"},
        {"prompt": "x", "model": "MiniMax-H3", "width": 32, "height": 32, "frames": 124},
    )
    latest = [{"path": "latest.safetensors", "weight": 0.6, "enabled": True}]
    runner.start()
    try:
        assert fake.started.wait(2)
        assert runner.queue_loras([{"path": "middle.safetensors", "weight": 0.4, "enabled": True}])["state"] == "queued"
        assert runner.queue_loras(latest)["state"] == "queued"
        fake.release.set()
        deadline = time.time() + 2
        while time.time() < deadline and len(fake.calls) < 2:
            time.sleep(0.01)
        assert fake.calls == [("generate", "MiniMax-H3"), ("loras", latest)]
        assert store.get("job-control")["status"] == "completed"
    finally:
        fake.release.set(); runner.stop()
