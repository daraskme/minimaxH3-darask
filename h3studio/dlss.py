from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
import msvcrt
from dataclasses import asdict, dataclass
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Iterator


ROOT = Path(__file__).resolve().parents[1]
DLSS_BIN = ROOT / "tools" / "dlss_exporter" / "bin"
SR_WORKER = DLSS_BIN / "H3DLSSExporter.exe"
SR_RUNTIME = DLSS_BIN / "nvngx_dlss.dll"
BUILD_MANIFEST = DLSS_BIN / "build-manifest.json"
_PACKAGED_NR = ROOT / ".cache" / "runtimes" / "dlss5-v0.24.0" / "DLSSVideoPlayer-v0.24.0-win64" / "neural-runtime"
_DEFAULT_NR = DLSS_BIN / "neural-runtime"
NR_DIRECTORY = Path(os.environ.get(
    "H3STUDIO_DLSS5_NR_DIRECTORY",
    _DEFAULT_NR if _DEFAULT_NR.is_dir() else _PACKAGED_NR,
))
NR_WORKER = NR_DIRECTORY / "NeuralWorker.exe"
NR_VALIDATION = NR_DIRECTORY / "h3studio-validation.json"
NR_LOCK = ROOT / "tools" / "dlss_exporter" / "runtime-lock.v0.24.0.json"

# NVIDIA/DLSS v310.9.1, Windows x86_64 release runtime.  The hash is also
# checked by scripts/build_dlss_exporter.ps1 before it is staged.
OFFICIAL_DLSS_SR_SHA256 = "3975567b8943c53acce397f2b72380092f84f162d00b0d2c7d08a1025c563983"
OFFICIAL_DLSS_COMMIT = "374959484e79a640feaba44c93ac8cfb0a03f5b5"
PLAYER_COMMIT = "ffb01f5d015ee1a0544a6a46452cfe4b334553a3"
MIN_DRIVER = (610, 47)

# This runtime is intentionally not downloaded or staged by this project.  It
# is the unsigned, community-modified runtime pinned by the audited upstream
# source.  Keeping the lock here lets diagnostics reject a different binary.
NR_RUNTIME_SHA256 = "6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927"
NR_RUNTIME_SIZE = 165_830_144

WIRE_MAGIC = 0x3152574E
WIRE_VERSION = 6
WIRE_MAX_PAYLOAD = 64 * 1024
WIRE_HEADER = struct.Struct("<IHHI")
WIRE_PROGRESS = struct.Struct("<IQQQqqII")
WIRE_RESULT = struct.Struct("<10B6xQqQQQQIIqdddddQQIIII")
WIRE_PREFLIGHT = struct.Struct("<B3xI")


class DLSSProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class NeuralResult:
    ok: bool
    cancelled: bool
    encoder: int
    feature18_armed_before_capture: bool
    upscaling_off: bool
    inline_interception_contract: bool
    feature18_created: bool
    feature18_evaluated: bool
    later_failure: bool
    failure: int
    frame_count: int
    duration_100ns: int
    native_evaluations: int
    verified_neural_frames: int
    highest_observed_evaluation: int
    job_id: int
    history_resets: int
    frame_retries: int
    first_timestamp_100ns: int
    neural_gpu_ms_p50: float
    neural_gpu_ms_p95: float
    neural_gpu_ms_max: float
    guide_ms_mean: float
    capture_ms_mean: float
    peak_local_vram_mib: int
    timing_samples: int
    accepted_strong_cuts: int
    accepted_weak_cuts: int
    suppressed_cuts: int
    detail: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=16)
def _sha256_cached(path_text: str, size: int, modified_ns: int) -> str:
    path = Path(path_text)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _parse_version(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    try:
        return tuple(int(part) for part in text.strip().split(".") if part != "")
    except ValueError:
        return None


def driver_version() -> str | None:
    cached = getattr(driver_version, "_cache", None)
    now = time.monotonic()
    if cached is not None and now - cached[0] < 10.0:
        return cached[1]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=4, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    version = next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)
    driver_version._cache = (now, version)
    return version


def _verified_build() -> tuple[bool, str, dict[str, object]]:
    if not BUILD_MANIFEST.is_file() or not SR_WORKER.is_file() or not SR_RUNTIME.is_file():
        return False, "公式SDK版DLSS SRエクスポーターが未ビルドです。", {}
    try:
        manifest = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unknown manifest schema")
        if manifest.get("player_commit") != PLAYER_COMMIT or manifest.get("dlss_commit") != OFFICIAL_DLSS_COMMIT:
            raise ValueError("source revision mismatch")
        if _sha256(SR_RUNTIME) != OFFICIAL_DLSS_SR_SHA256:
            raise ValueError("official runtime hash mismatch")
        if _sha256(SR_WORKER) != str(manifest.get("worker_sha256", "")).lower():
            raise ValueError("worker hash mismatch")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"DLSS SRビルド検証に失敗しました: {exc}", {}
    return True, "", manifest


def _nr_runtime_status() -> tuple[bool, str]:
    runtime = NR_DIRECTORY / "nvngx_dlssnr.dll"
    if not runtime.is_file():
        return False, "DLSS 5 NRの未署名・改変ランタイムは配置されていません。"
    stat = runtime.stat()
    if stat.st_size != NR_RUNTIME_SIZE or _sha256(runtime) != NR_RUNTIME_SHA256:
        return False, "DLSS 5 NRランタイムが監査済みハッシュと一致しません。"
    # The validator records the user's explicit opt-in beside this exact,
    # hash-locked runtime.  That durable marker lets a normal launcher expose
    # the validated capability without weakening process-wide DLL policy.
    authorized = os.environ.get("H3STUDIO_ALLOW_UNSIGNED_DLSS_NR") == "1"
    if not authorized and NR_VALIDATION.is_file():
        try:
            marker = json.loads(NR_VALIDATION.read_text(encoding="utf-8"))
            authorized = marker.get("user_authorized_unsigned_runtime") is True
        except (OSError, json.JSONDecodeError):
            authorized = False
    if not authorized:
        return False, "未署名DLSS 5 NRランタイムの実行許可がありません。"
    return True, ""


def verify_nr_locked_files() -> dict[str, object]:
    if not NR_LOCK.is_file():
        raise RuntimeError("DLSS 5 NR runtime lock is missing")
    lock = json.loads(NR_LOCK.read_text(encoding="utf-8"))
    entries = lock["entries"]
    if lock.get("schemaVersion") != 1 or not isinstance(entries, list) or not entries:
        raise RuntimeError("invalid runtime lock")
    for entry in entries:
        target = (NR_DIRECTORY / str(entry["destination"])).resolve()
        if not target.is_relative_to(NR_DIRECTORY.resolve()) or not target.is_file():
            raise RuntimeError(f"missing {entry['destination']}")
        if target.stat().st_size != int(entry["size"]) or _sha256(target).upper() != str(entry["sha256"]).upper():
            raise RuntimeError(f"runtime lock mismatch for {entry['destination']}")
    if not NR_WORKER.is_file():
        raise RuntimeError("missing NeuralWorker.exe")
    return lock


def _verify_nr_runtime_set() -> tuple[bool, str, dict[str, object]]:
    try:
        verify_nr_locked_files()
        validation = json.loads(NR_VALIDATION.read_text(encoding="utf-8")) if NR_VALIDATION.is_file() else {}
        if validation.get("schema_version") != 1 or validation.get("protocol_version") != WIRE_VERSION:
            raise ValueError("DLSS 5 NR preflight validation is missing")
        if validation.get("player_commit") != PLAYER_COMMIT:
            raise ValueError("DLSS 5 NR worker revision mismatch")
        if validation.get("worker_sha256") != _sha256(NR_WORKER):
            raise ValueError("DLSS 5 NR worker hash mismatch")
        if validation.get("runtime_lock_sha256") != _sha256(NR_LOCK):
            raise ValueError("DLSS 5 NR lock revision mismatch")
        if validation.get("driver") != driver_version() or validation.get("preflight_ok") is not True:
            raise ValueError("DLSS 5 NR preflight was not validated on this driver")
        if validation.get("user_authorized_unsigned_runtime") is not True:
            raise ValueError("DLSS 5 NR unsigned runtime authorization is missing")
        export = validation.get("export_validation")
        if not isinstance(export, dict) or export.get("ok") is not True or export.get("driver") != driver_version():
            raise ValueError("DLSS 5 NR real-file export has not been validated on this driver")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        return False, str(exc), {}
    return True, "", validation


def ffmpeg_helper_directory(ffmpeg: str | Path) -> Path:
    """Expose the selected ffmpeg under the fixed name expected by the native worker."""
    source = Path(ffmpeg).resolve(strict=True)
    directory = DLSS_BIN / "helpers"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "ffmpeg.exe"
    marker = directory / "ffmpeg-source.json"
    stat = source.stat()
    identity = {"source": str(source), "size": stat.st_size, "modified_ns": stat.st_mtime_ns}
    try:
        existing = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
    except (OSError, json.JSONDecodeError):
        existing = None
    if existing == identity and target.is_file() and target.stat().st_size == stat.st_size:
        return directory
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    marker.write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


def capabilities() -> list[dict[str, object]]:
    version = driver_version()
    parsed_driver = _parse_version(version)
    driver_ok = parsed_driver is not None and parsed_driver >= MIN_DRIVER
    build_ok, build_reason, manifest = _verified_build()
    smoke = manifest.get("smoke_validation") if isinstance(manifest.get("smoke_validation"), dict) else {}
    smoke_ok = bool(manifest.get("smoke_validated") is True and smoke.get("driver") == version)
    sr_ready = bool(driver_ok and build_ok and smoke_ok)
    if not driver_ok:
        sr_reason = f"NVIDIAドライバー610.47以上が必要です（現在: {version or '検出不能'}）。"
    elif not build_ok:
        sr_reason = build_reason
    elif not smoke_ok:
        sr_reason = "このドライバー/GPUで実ファイル書き出し検証が完了していません。"
    else:
        sr_reason = ""

    nr_runtime_ok, nr_reason = _nr_runtime_status()
    nr_set_ok, nr_set_reason, nr_validation = _verify_nr_runtime_set()
    nr_ready = bool(driver_ok and nr_runtime_ok and nr_set_ok)
    if not driver_ok:
        nr_reason = f"NVIDIAドライバー610.47以上が必要です（現在: {version or '検出不能'}）。"
    elif not NR_WORKER.is_file():
        nr_reason = "固定リビジョンのNeuralWorkerが未ビルドです。"
    elif not nr_set_ok:
        nr_reason = f"DLSS 5 NRの固定runtime/preflight検証が未完了です: {nr_set_reason}"

    return [
        {
            "id": "dlss_super_resolution",
            "label": "NVIDIA DLSS Super Resolution",
            "available": sr_ready,
            "kind": "ai",
            "factors": [2] if sr_ready else [],
            "trust": "official_signed_runtime_local_source_build",
            "installed": build_ok,
            "validated": smoke_ok,
            "driver": version,
            "engine_revision": manifest.get("player_commit") if manifest else PLAYER_COMMIT,
            "description": "NVIDIA公式DLSS SDKのSR/DLAA機能で2倍に書き出します。DLSS 5 Neural Renderingではありません。",
            **({} if sr_ready else {"reason": sr_reason}),
        },
        {
            "id": "dlss5_neural_rendering",
            "label": "DLSS 5 Neural Rendering（実験）",
            "available": nr_ready,
            "kind": "experimental_ai",
            "factors": [1] if nr_ready else [],
            "trust": "unsigned_community_modified_runtime",
            "installed": bool(NR_WORKER.is_file() and nr_runtime_ok),
            "validated": nr_set_ok,
            "driver": version,
            "engine_revision": PLAYER_COMMIT,
            "description": "ソース解像度のNeural Rendering書き出しです。未署名のコミュニティ改変runtimeを隔離workerで使用します。公式DLSS SRとは別機能です。",
            **({"validation": nr_validation} if nr_set_ok else {}),
            **({} if nr_ready else {"reason": nr_reason}),
        },
    ]


def iter_wire_messages(stream: BinaryIO) -> Iterator[tuple[int, bytes]]:
    """Decode exact v6 NeuralWorker frames from its inherited metadata pipe."""
    while True:
        header = _read_exact(stream, WIRE_HEADER.size, allow_clean_eof=True)
        if not header:
            return
        magic, version, kind, payload_bytes = WIRE_HEADER.unpack(header)
        if magic != WIRE_MAGIC or version != WIRE_VERSION:
            raise DLSSProtocolError("neural worker protocol identity mismatch")
        if kind not in range(1, 8):
            raise DLSSProtocolError("unknown neural worker message kind")
        if payload_bytes > WIRE_MAX_PAYLOAD:
            raise DLSSProtocolError("neural worker payload exceeds the v6 limit")
        yield kind, _read_exact(stream, payload_bytes)


def _read_exact(stream: BinaryIO, size: int, allow_clean_eof: bool = False) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        part = stream.read(size - len(payload))
        if not part:
            if allow_clean_eof and not payload:
                return b""
            raise DLSSProtocolError("truncated neural worker stream")
        payload.extend(part)
    return bytes(payload)


def decode_result(payload: bytes) -> NeuralResult:
    if len(payload) < WIRE_RESULT.size:
        raise DLSSProtocolError("truncated v6 result")
    values = WIRE_RESULT.unpack_from(payload)
    if any(value not in (0, 1) for value in (*values[0:2], *values[3:9])):
        raise DLSSProtocolError("invalid boolean in v6 result")
    if payload[10:16] != b"\0" * 6:
        raise DLSSProtocolError("nonzero reserved bytes in v6 result")
    if values[2] not in (0, 1) or not (0 <= values[9] <= 11):
        raise DLSSProtocolError("unknown enum in v6 result")
    if values[18] < 0 or not all(math.isfinite(value) and 0 <= value <= 1e12 for value in values[19:24]):
        raise DLSSProtocolError("invalid timing metric in v6 result")
    detail_bytes = values[-1]
    if detail_bytes > 4 * 1024 or detail_bytes % 2 or len(payload) != WIRE_RESULT.size + detail_bytes:
        raise DLSSProtocolError("invalid v6 result detail length")
    detail = payload[WIRE_RESULT.size:].decode("utf-16-le", errors="strict")
    if "\0" in detail:
        raise DLSSProtocolError("NUL in v6 result detail")
    ok, cancelled, failure = bool(values[0]), bool(values[1]), values[9]
    evidence_valid = all(bool(values[index]) for index in (4, 5, 6, 7)) and not bool(values[8]) and values[14] > 0
    if ok and (
        cancelled or not values[10] or values[11] <= 0 or not values[12] or values[13] < values[10]
        or not bool(values[3]) or not evidence_valid or failure != 0
    ):
        raise DLSSProtocolError("inconsistent successful v6 result")
    if cancelled and (ok or failure != 8):
        raise DLSSProtocolError("inconsistent cancelled v6 result")
    if not ok and not cancelled and failure == 0:
        raise DLSSProtocolError("failed v6 result has no failure classification")
    return NeuralResult(
        ok=ok, cancelled=cancelled, encoder=values[2],
        feature18_armed_before_capture=bool(values[3]), upscaling_off=bool(values[4]),
        inline_interception_contract=bool(values[5]), feature18_created=bool(values[6]),
        feature18_evaluated=bool(values[7]), later_failure=bool(values[8]), failure=values[9],
        frame_count=values[10], duration_100ns=values[11], native_evaluations=values[12],
        verified_neural_frames=values[13], highest_observed_evaluation=values[14], job_id=values[15],
        history_resets=values[16], frame_retries=values[17], first_timestamp_100ns=values[18],
        neural_gpu_ms_p50=values[19], neural_gpu_ms_p95=values[20], neural_gpu_ms_max=values[21],
        guide_ms_mean=values[22], capture_ms_mean=values[23], peak_local_vram_mib=values[24],
        timing_samples=values[25], accepted_strong_cuts=values[26], accepted_weak_cuts=values[27],
        suppressed_cuts=values[28], detail=detail,
    )


def decode_preflight(payload: bytes) -> dict[str, object]:
    if len(payload) < WIRE_PREFLIGHT.size:
        raise DLSSProtocolError("truncated v6 preflight")
    ok, json_bytes = WIRE_PREFLIGHT.unpack_from(payload)
    if ok not in (0, 1) or payload[1:4] != b"\0\0\0":
        raise DLSSProtocolError("invalid v6 preflight flags")
    if not json_bytes or json_bytes > WIRE_MAX_PAYLOAD - WIRE_PREFLIGHT.size or len(payload) != WIRE_PREFLIGHT.size + json_bytes:
        raise DLSSProtocolError("invalid v6 preflight JSON length")
    raw = payload[WIRE_PREFLIGHT.size:]
    if b"\0" in raw:
        raise DLSSProtocolError("NUL in v6 preflight JSON")
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, dict):
        raise DLSSProtocolError("v6 preflight payload is not an object")
    value["ok"] = bool(ok)
    return value


def _spawn_with_metadata(command: list[str]) -> tuple[subprocess.Popen[str], BinaryIO]:
    read_fd, write_fd = os.pipe()
    write_handle = msvcrt.get_osfhandle(write_fd)
    os.set_handle_inheritable(write_handle, True)
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    startup.lpAttributeList = {"handle_list": [write_handle]}
    # The exact v6 parser permits --configuration-restarted only as the final
    # token.  Keep the inherited handle pair before it on the one allowed
    # repair restart.
    restarted = bool(command and command[-1] == "--configuration-restarted")
    base = command[:-1] if restarted else command
    resolved = [*base, "--metadata-handle", str(write_handle)]
    if restarted:
        resolved.append("--configuration-restarted")
    try:
        process = subprocess.Popen(
            resolved, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", shell=False, close_fds=True,
            startupinfo=startup, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            cwd=str(NR_DIRECTORY),
        )
    except Exception:
        os.close(read_fd)
        os.close(write_fd)
        raise
    os.close(write_fd)
    return process, os.fdopen(read_fd, "rb", buffering=0)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Stop only the worker and descendants it created, then reap handles."""
    try:
        import psutil

        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            parent.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs([*children, parent], timeout=5)
        for item in alive:
            try:
                item.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5)
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _run_neural_worker(
    arguments: list[str], cancel=None, on_message=None,
    *, timeout_seconds: float = 3600.0, idle_timeout_seconds: float = 300.0,
) -> tuple[int, list[tuple[int, bytes]], list[str]]:
    """Run one exact-tag worker and collect a bounded set of v6 pipe frames."""
    restarted = False
    while True:
        command = [str(NR_WORKER), *arguments]
        if restarted:
            command.append("--configuration-restarted")
        process, metadata = _spawn_with_metadata(command)
        frames: deque[tuple[int, bytes]] = deque(maxlen=256)
        errors: list[BaseException] = []
        diagnostics: deque[str] = deque(maxlen=200)
        activity = [time.monotonic()]

        def read_metadata() -> None:
            try:
                for message in iter_wire_messages(metadata):
                    activity[0] = time.monotonic()
                    frames.append(message)
                    if on_message:
                        on_message(*message)
            except BaseException as exc:
                errors.append(exc)
            finally:
                metadata.close()

        def read_diagnostics() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                activity[0] = time.monotonic()
                diagnostics.append(line.rstrip())

        readers = [threading.Thread(target=read_metadata), threading.Thread(target=read_diagnostics)]
        for reader in readers:
            reader.start()
        started = time.monotonic()
        try:
            while process.poll() is None:
                if errors:
                    _terminate_process_tree(process)
                    break
                if cancel is not None and cancel.is_set():
                    _terminate_process_tree(process)
                    from .engine import GenerationCancelled
                    raise GenerationCancelled("cancelled during DLSS 5 Neural Rendering")
                now = time.monotonic()
                if now - started > timeout_seconds:
                    _terminate_process_tree(process)
                    raise TimeoutError("DLSS 5 Neural Rendering exceeded its job deadline")
                if now - activity[0] > idle_timeout_seconds:
                    _terminate_process_tree(process)
                    raise TimeoutError("DLSS 5 Neural Rendering stopped reporting progress")
                time.sleep(0.05)
            code = process.wait()
        finally:
            if process.poll() is None:
                _terminate_process_tree(process)
            for reader in readers:
                reader.join(timeout=5)
        if any(reader.is_alive() for reader in readers):
            raise DLSSProtocolError("neural worker pipe did not close")
        if errors:
            raise DLSSProtocolError(str(errors[0]))
        if code == 75 and not restarted:
            if frames:
                raise DLSSProtocolError("configuration restart carried unexpected metadata")
            restarted = True
            continue
        return code, list(frames), list(diagnostics)


def run_neural_preflight() -> dict[str, object]:
    if not NR_WORKER.is_file():
        raise RuntimeError(f"NeuralWorker is missing: {NR_WORKER}")
    code, frames, diagnostics = _run_neural_worker(
        ["--neural-preflight"], timeout_seconds=120, idle_timeout_seconds=90,
    )
    preflights = [decode_preflight(payload) for kind, payload in frames if kind == 3]
    if code != 0 or len(preflights) != 1 or any(kind != 3 for kind, _ in frames):
        raise RuntimeError(f"DLSS 5 preflight protocol failed ({code}): {'; '.join(diagnostics[-20:])}")
    result = preflights[0]
    if result.get("ok") is not True:
        raise RuntimeError(f"DLSS 5 preflight refused the runtime: {result}")
    return result


def _is_quantized_cfr_delta(delta: float, fps: float, quantum: float) -> bool:
    tolerance = max(0.0005, 0.02 / fps, abs(quantum) * 1.01)
    return delta > 0 and abs(delta - 1.0 / fps) <= tolerance


def _inspect_neural_output(path: Path) -> dict[str, object]:
    import av

    with av.open(str(path), mode="r", format="matroska", options={"protocol_whitelist": "file,pipe"}) as container:
        if len(container.streams.video) != 1:
            raise RuntimeError("DLSS 5 worker output must contain exactly one video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if not rate:
            raise RuntimeError("DLSS 5 worker output has no frame rate")
        fps = float(rate)
        frames = 0
        previous: float | None = None
        cfr = True
        for frame in container.decode(stream):
            frames += 1
            if frame.pts is None or frame.time_base is None:
                cfr = False
                continue
            timestamp = float(frame.pts * frame.time_base)
            if previous is not None:
                delta = timestamp - previous
                # The worker's Matroska time base is commonly 1 ms.  At 60 or
                # 59.94 fps a valid CFR grid therefore alternates 16/17 ms.
                # Admit one timestamp tick of quantization here; the public MP4
                # is restamped onto the exact rational source grid afterwards.
                quantum = abs(float(frame.time_base))
                if not _is_quantized_cfr_delta(delta, fps, quantum):
                    cfr = False
            previous = timestamp
        return {
            "width": int(stream.width), "height": int(stream.height), "fps": fps,
            "frames": frames, "is_cfr": cfr and frames >= 2,
        }


def run_neural_rendering(
    source: Path, staging: Path, source_info: dict[str, object], progress, cancel,
    *, require_export_validation: bool = True,
) -> dict[str, object]:
    if require_export_validation:
        method = next(item for item in capabilities() if item["id"] == "dlss5_neural_rendering")
        if not method["available"]:
            raise RuntimeError(str(method.get("reason", "DLSS 5 Neural Rendering is unavailable")))
    else:
        runtime_ok, reason = _nr_runtime_status()
        if not runtime_ok:
            raise RuntimeError(reason)
        verify_nr_locked_files()
    width, height = int(source_info["width"]), int(source_info["height"])
    fps = float(source_info["fps"])
    expected_frames = int(source_info["frame_count"])
    duration_100ns = int(round(float(source_info["duration_seconds"]) * 10_000_000))
    job_material = f"{source_info.get('sha256', source)}:{width}:{height}:{fps}:{expected_frames}".encode()
    job_id = int.from_bytes(hashlib.sha256(job_material).digest()[:8], "little") or 1
    staging.unlink(missing_ok=True)

    latest_progress = 0

    def handle_message(kind: int, payload: bytes) -> None:
        nonlocal latest_progress
        if kind != 1:
            return
        if len(payload) != WIRE_PROGRESS.size:
            raise DLSSProtocolError("invalid v6 progress size")
        phase, completed, total, _bytes, _elapsed, _remaining, recovering, _retries = WIRE_PROGRESS.unpack(payload)
        if phase > 9 or recovering > 11 or completed < latest_progress or (total and completed > total):
            raise DLSSProtocolError("invalid v6 progress values")
        latest_progress = completed
        if total:
            progress(0.05 + 0.80 * min(1.0, completed / total), f"DLSS 5 Neural Rendering · {completed}/{total}f")

    arguments = [
        "--neural-worker", "--source", str(source), "--staging", str(staging),
        "--width", str(width), "--height", str(height), "--fps", format(fps, ".12g"),
        "--duration-100ns", str(duration_100ns), "--job-id", str(job_id),
        "--range-start-100ns", "0", "--range-end-100ns", "0", "--preroll-frames", "60",
        "--frame-retry-limit", "3", "--guides", "mv=1,depth=1",
        "--gpu-color-conversion", "1", "--nvenc-preset", "5",
    ]
    render_deadline = max(600.0, min(3600.0, float(source_info["duration_seconds"]) * 120.0))
    code, frames, diagnostics = _run_neural_worker(
        arguments, cancel=cancel, on_message=handle_message,
        timeout_seconds=render_deadline, idle_timeout_seconds=300,
    )
    results = [decode_result(payload) for kind, payload in frames if kind == 2]
    if code != 0 or len(results) != 1:
        raise RuntimeError(f"DLSS 5 worker protocol failed ({code}): {'; '.join(diagnostics[-20:])}")
    if any(kind not in (1, 2, 5, 7) for kind, _ in frames):
        raise DLSSProtocolError("unexpected message in DLSS 5 render stream")
    result = results[0]
    if not result.ok:
        raise RuntimeError(f"DLSS 5 render failed ({result.failure}): {result.detail}")
    if result.job_id != job_id or result.frame_count != expected_frames:
        raise DLSSProtocolError("DLSS 5 terminal receipt does not match the requested job")
    if abs(result.duration_100ns - duration_100ns) > round(10_000_000 / fps):
        raise DLSSProtocolError("DLSS 5 terminal duration does not match the source")
    if not staging.is_file() or staging.stat().st_size == 0:
        raise RuntimeError("DLSS 5 worker produced no finalized video")
    inspected = _inspect_neural_output(staging)
    if (
        inspected["width"] != width or inspected["height"] != height
        or inspected["frames"] != expected_frames or not inspected["is_cfr"]
        or abs(float(inspected["fps"]) - fps) > 0.005
    ):
        raise RuntimeError(f"DLSS 5 output validation failed: {inspected}")
    return {**result.as_dict(), "output": inspected}


def run_super_resolution(
    source: Path,
    silent_output: Path,
    receipt_path: Path,
    source_info: dict[str, object],
    ffmpeg: str,
    progress,
    cancel,
) -> dict[str, object]:
    """Run the separately built official DLSS SR exporter.

    This function does not implement or load the unsigned DLSS 5 NR runtime.
    Availability is gated by a build manifest and an exact-driver smoke receipt.
    """
    method = next(item for item in capabilities() if item["id"] == "dlss_super_resolution")
    if not method["available"]:
        raise RuntimeError(str(method.get("reason", "DLSS Super Resolution is unavailable")))
    width = int(source_info["width"]) * 2
    height = int(source_info["height"]) * 2
    expected_frames = int(source_info["frame_count"])
    fps = float(source_info["fps"])
    silent_output.unlink(missing_ok=True)
    receipt_path.unlink(missing_ok=True)
    command = [
        str(SR_WORKER), "--h3-dlss-sr", "--source", str(source), "--output", str(silent_output),
        "--receipt", str(receipt_path), "--ffmpeg-dir", str(ffmpeg_helper_directory(ffmpeg)),
        "--target-width", str(width), "--target-height", str(height),
        "--expected-fps", format(fps, ".12g"), "--expected-frames", str(expected_frames),
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    lines: queue.Queue[str | None] = queue.Queue()
    diagnostics: list[str] = []

    def drain() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    stream_done = False
    try:
        while process.poll() is None or not stream_done:
            if cancel.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                from .engine import GenerationCancelled
                raise GenerationCancelled("cancelled during NVIDIA DLSS Super Resolution")
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                stream_done = True
                continue
            stripped = line.strip()
            if stripped.startswith("H3DLSS_PROGRESS "):
                parts = stripped.split()
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit() and int(parts[2]) > 0:
                    progress(0.05 + 0.82 * min(1.0, int(parts[1]) / int(parts[2])),
                             f"NVIDIA DLSS SR · {parts[1]}f")
            elif len(diagnostics) < 100:
                diagnostics.append(stripped)
        code = process.wait()
        if code != 0:
            raise RuntimeError("DLSS SR exporter failed: " + "\n".join(diagnostics[-20:]))
        if not receipt_path.is_file() or not silent_output.is_file() or silent_output.stat().st_size == 0:
            raise RuntimeError("DLSS SR exporter returned without a complete receipt and output")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("contract") != "h3studio-dlss-sr-v1" or receipt.get("ok") is not True
            or receipt.get("width") != width or receipt.get("height") != height
            or receipt.get("frames") != expected_frames
            or receipt.get("dlss_evaluations") != expected_frames
        ):
            raise RuntimeError(f"DLSS SR receipt failed validation: {receipt}")
        return receipt
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=2)
