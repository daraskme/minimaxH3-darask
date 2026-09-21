from __future__ import annotations

"""Bounded native-Windows adapter for the standalone SeedVR2-3B runtime."""

import json
import hashlib
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .engine import GenerationCancelled
from .metadata import _ffmpeg_path
from .video import inspect_video, sha256_file


ProgressCallback = Callable[[float, str], None]
AUDITED_REPOSITORY = "numz/ComfyUI-SeedVR2_VideoUpscaler"
AUDITED_REVISION = "4490bd1f482e026674543386bb2a4d176da245b9"
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
VAE_MODEL = "ema_vae_fp16.safetensors"
MODEL_SHA256 = {
    DIT_MODEL: "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304",
    VAE_MODEL: "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1",
}
UPSTREAM_SHA256 = {
    "inference_cli.py": "906ef894c7fe9fba7bded45d3be4192e7aa991dd7e5f93509cef9c378235e0a1",
    "src/core/infer.py": "ebdf30dbe77d3a7ec332f3c31dbd98864f096419796d704dfcc3ae4409d8434e",
    "src/core/model_loader.py": "f598cfdaef3a2350a1d1f52e858b126e8babcb5b6e80601a61866b400ca27989",
    "src/models/dit_3b/nadit.py": "882154cb00141001ec23e0f594b4075c12432c6345a4521b342587b8211a5e84",
    "configs_3b/main.yaml": "1b251c0679ef1abf6eef6224bcf44c0c0847dde63b535e771a3d8387333a0c36",
    "pos_emb.pt": "fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799",
    "neg_emb.pt": "6a43e5800ef2354f1c156d27535834da055cbec8248298b8923492bba2076581",
}
LAUNCHER_SHA256 = "1787f03f358f9eb848636e2d4a48472fe86d2c19c3da0baba67222db971d39a3"
UPSTREAM_TREE_SHA256 = "5e27e19bc22c4c8e58222f8ddf108f11382cbb223d75071ecfbd3d3d35749d50"
RUNTIME_DISTRIBUTIONS = (
    "torch", "torchvision", "diffusers", "peft", "safetensors", "einops", "tqdm", "psutil",
    "opencv-python", "omegaconf", "gguf", "rotary-embedding-torch", "matplotlib",
)
RECEIPT_NAME = "h3studio-seedvr2-selfcheck.json"
RECEIPT_SCHEMA = "h3studio.seedvr2-selfcheck/v1"
MAX_SOURCE_FRAMES = 2_400
MAX_OUTPUT_PIXEL_FRAMES = 2_000_000_000


def configured_paths(root: Path) -> tuple[Path, Path, Path]:
    repo = Path(os.environ.get("H3STUDIO_SEEDVR2_REPO", root / "tools" / "seedvr2-standalone"))
    checkpoints = Path(
        os.environ.get("H3STUDIO_SEEDVR2_CHECKPOINTS", root / "models" / "seedvr2" / "SeedVR2-3B")
    )
    default_python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = Path(os.environ.get("H3STUDIO_SEEDVR2_PYTHON", default_python))
    return repo, checkpoints, python


@lru_cache(maxsize=32)
def _verified_hash(path_text: str, size: int, modified_ns: int, expected: str) -> bool:
    path = Path(path_text)
    return (
        path.is_file()
        and path.stat().st_size == size
        and path.stat().st_mtime_ns == modified_ns
        and sha256_file(path) == expected
    )


def _hash_matches(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    return _verified_hash(str(path), stat.st_size, stat.st_mtime_ns, expected)


def _source_tree_sha256(repo: Path) -> str:
    """Hash every vendored upstream file; ignore only H3 additions and Python caches."""
    digest = hashlib.sha256()
    excluded = {"h3studio_entry.py", "UPSTREAM_REVISION"}
    files = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repo).as_posix()
        if relative in excluded or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append((relative, path))
    for relative, path in sorted(files):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _runtime_signature(python_text: str) -> tuple[bool, str]:
    distributions = json.dumps(RUNTIME_DISTRIBUTIONS)
    code = r"""
import importlib.metadata as md, json, subprocess, sys
names = json.loads(sys.argv[1])
driver = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
    text=True, encoding='utf-8', errors='replace', timeout=10,
).splitlines()[0].strip()
print(json.dumps({
    'python': sys.version.split()[0], 'driver': driver,
    'packages': {name: md.version(name) for name in names},
}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [python_text, "-X", "utf8", "-c", code, distributions],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"隔離SeedVR2環境の版情報を検査できません: {exc}"
    if completed.returncode:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        return False, f"隔離SeedVR2依存関係が未準備です: {(lines[-1] if lines else 'signature probe failed')}"
    try:
        signature = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "隔離SeedVR2環境の版情報を検証できません"
    return True, json.dumps(signature, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=8)
def _probe_runtime(
    python_text: str, repo_text: str, python_modified_ns: int, signature_text: str,
) -> tuple[bool, Any]:
    del python_modified_ns
    code = r"""
import json, sys, torch, torchvision, diffusers, peft, safetensors, einops, tqdm, psutil
sys.path.insert(0, sys.argv[1])
import cv2, gguf, matplotlib, omegaconf, rotary_embedding_torch
from src.models.dit_3b.nadit import NaDiT
result = json.loads(sys.argv[2])
result.update({
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
print(json.dumps(result, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [python_text, "-X", "utf8", "-c", code, repo_text, signature_text],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"隔離SeedVR2環境を検査できません: {exc}"
    if completed.returncode:
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        return False, f"隔離SeedVR2依存関係が未準備です: {(lines[-1] if lines else 'probe failed')}"
    try:
        data = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False, "隔離SeedVR2環境の応答を検証できません"
    if not data.get("cuda_available"):
        return False, "隔離SeedVR2環境からCUDA GPUを利用できません"
    return True, data


def _prerequisites(root: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    repo, checkpoints, python = configured_paths(root)
    base: dict[str, Any] = {
        "id": "seedvr2_3b", "label": "SeedVR2 3B FP16", "kind": "ai-restoration", "factors": [2],
        "repository": AUDITED_REPOSITORY, "revision": AUDITED_REVISION,
        "checkpoint": "numz/SeedVR2_comfyUI", "attention": "sdpa", "precision": "fp16",
        "limits": {"source_frames": MAX_SOURCE_FRAMES, "output_pixel_frames": MAX_OUTPUT_PIXEL_FRAMES},
    }
    if not (repo / "h3studio_entry.py").is_file():
        return {**base, "installed": False, "reason": "SeedVR2 Windowsランチャーが未配置です"}, repo, checkpoints, python
    if not _hash_matches(repo / "h3studio_entry.py", LAUNCHER_SHA256):
        return {**base, "installed": False, "reason": "SeedVR2 WindowsランチャーのSHA256が一致しません"}, repo, checkpoints, python
    for relative, expected in UPSTREAM_SHA256.items():
        path = repo / relative
        if not path.is_file():
            return {**base, "installed": False, "reason": f"監査済みSeedVR2ソースが未配置です（{relative}）"}, repo, checkpoints, python
        if not _hash_matches(path, expected):
            return {**base, "installed": False, "reason": f"SeedVR2ソースのSHA256が一致しません（{relative}）"}, repo, checkpoints, python
    if _source_tree_sha256(repo) != UPSTREAM_TREE_SHA256:
        return {**base, "installed": False, "reason": "SeedVR2監査済みソースツリー全体のSHA256が一致しません"}, repo, checkpoints, python
    revision_file = repo / "UPSTREAM_REVISION"
    if not revision_file.is_file() or revision_file.read_text(encoding="ascii").strip() != AUDITED_REVISION:
        return {**base, "installed": False, "reason": "SeedVR2ソースのリビジョン記録が一致しません"}, repo, checkpoints, python
    for name, expected in MODEL_SHA256.items():
        path = checkpoints / name
        if not path.is_file():
            return {**base, "installed": False, "reason": f"SeedVR2モデルが未配置です（{name}）"}, repo, checkpoints, python
        if not _hash_matches(path, expected):
            return {**base, "installed": False, "reason": f"SeedVR2モデルのSHA256が一致しません（{name}）"}, repo, checkpoints, python
    if not python.is_file():
        return {**base, "installed": False, "reason": "SeedVR2を実行するPython環境が未配置です"}, repo, checkpoints, python
    signature_ready, signature = _runtime_signature(str(python))
    if not signature_ready:
        return {**base, "installed": False, "reason": signature}, repo, checkpoints, python
    ready, runtime = _probe_runtime(str(python), str(repo), python.stat().st_mtime_ns, signature)
    if not ready:
        return {**base, "installed": False, "reason": runtime}, repo, checkpoints, python
    return {**base, "installed": True, "runtime": runtime}, repo, checkpoints, python


def _valid_receipt(receipt_path: Path, runtime: dict[str, Any]) -> tuple[bool, str]:
    if not receipt_path.is_file():
        return False, "実モデル推論セルフチェックがまだ完了していません"
    if receipt_path.stat().st_size > 64 * 1024:
        return False, "SeedVR2セルフチェック記録が不正です"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "SeedVR2セルフチェック記録を読み込めません"
    expected = {
        "schema": RECEIPT_SCHEMA, "repository_revision": AUDITED_REVISION,
        "source_tree_sha256": UPSTREAM_TREE_SHA256,
        "source_sha256": {**UPSTREAM_SHA256, "h3studio_entry.py": LAUNCHER_SHA256},
        "model_sha256": MODEL_SHA256, "runtime": runtime,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            return False, f"SeedVR2セルフチェック記録が現在の{key}と一致しません"
    inference = receipt.get("inference") or {}
    if not inference.get("actual_neural_execution") or inference.get("frames", 0) < 2:
        return False, "SeedVR2セルフチェック記録に実推論結果がありません"
    return True, ""


def capability(root: Path) -> dict[str, Any]:
    status, _, checkpoints, _ = _prerequisites(root)
    if not status.get("installed"):
        return {**status, "available": False}
    valid, reason = _valid_receipt(checkpoints / RECEIPT_NAME, status["runtime"])
    if not valid:
        return {**status, "available": False, "tested": False, "reason": reason}
    return {**status, "available": True, "tested": True}


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, shell=False)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        process.kill()
        process.wait(timeout=5)


def _run_cancellable(
    command: list[str], cwd: Path, environment: dict[str, str], cancel, phase: str,
    progress: ProgressCallback, progress_value: float,
) -> list[str]:
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", shell=False,
        creationflags=flags, start_new_session=os.name != "nt",
    )
    messages: queue.Queue[str | None] = queue.Queue()
    tail: list[str] = []

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line.rstrip())
        messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        done = False
        while process.poll() is None or not done:
            if cancel.is_set():
                _terminate_tree(process)
                raise GenerationCancelled(f"cancelled during {phase}")
            try:
                line = messages.get(timeout=0.15)
            except queue.Empty:
                progress(progress_value, phase)
                continue
            if line is None:
                done = True
            else:
                tail.append(line)
                tail = tail[-160:]
        if process.returncode:
            raise RuntimeError(f"{phase} failed: " + "\n".join(tail[-20:]))
        return tail
    finally:
        if process.poll() is None:
            _terminate_tree(process)
        reader.join(timeout=2)


def _validate_settings(settings: dict[str, Any]) -> tuple[int, int, int, Fraction]:
    if int(settings.get("scale", 2)) != 2:
        raise RuntimeError("SeedVR2 3B supports only 2x output")
    source = settings["source"]
    width, height = int(source["width"]) * 2, int(source["height"]) * 2
    frames = int(source["frame_count"])
    rate = Fraction(int(source["fps_numerator"]), int(source["fps_denominator"]))
    if frames < 2 or frames > MAX_SOURCE_FRAMES:
        raise RuntimeError(f"SeedVR2 source frame count must be 2..{MAX_SOURCE_FRAMES}")
    if width * height * frames > MAX_OUTPUT_PIXEL_FRAMES:
        raise RuntimeError("SeedVR2 job exceeds the bounded decoded-frame staging limit")
    return width, height, frames, rate


def validate_request_bounds(settings: dict[str, Any]) -> None:
    """Fail a queued request before it can reserve the worker for an unbounded run."""
    _validate_settings(settings)


def _inspect_png_sequence(directory: Path, expected_frames: int, width: int, height: int) -> list[Path]:
    from PIL import Image

    frames = sorted(directory.glob("*.png"))
    if len(frames) != expected_frames:
        raise RuntimeError(f"SeedVR2 returned {len(frames)} frames; exact {expected_frames} was required")
    for index, path in enumerate(frames):
        if path.name.rsplit("_", 1)[-1] != f"{index:06d}.png":
            raise RuntimeError(f"SeedVR2 returned a non-contiguous PNG sequence at frame {index}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (width, height):
                raise RuntimeError(
                    f"SeedVR2 frame {index} is {image.size[0]}x{image.size[1]}; exact {width}x{height} was required"
                )
    return frames


class SeedVR2Engine:
    def __init__(self, root: Path, ffmpeg: str | None = None):
        self.root = root
        self.ffmpeg = ffmpeg

    def run(
        self, source: Path, output: Path, settings: dict[str, Any],
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        status = capability(self.root)
        if not status["available"]:
            raise RuntimeError(status["reason"])
        return self._run_verified(source, output, settings, progress, cancel, status["runtime"])

    def selfcheck(
        self, source: Path, output: Path, settings: dict[str, Any],
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        status, _, checkpoints, _ = _prerequisites(self.root)
        if not status.get("installed"):
            raise RuntimeError(status["reason"])
        result = self._run_verified(source, output, settings, progress, cancel, status["runtime"])
        receipt = {
            "schema": RECEIPT_SCHEMA, "repository": AUDITED_REPOSITORY,
            "repository_revision": AUDITED_REVISION,
            "source_tree_sha256": UPSTREAM_TREE_SHA256,
            "source_sha256": {**UPSTREAM_SHA256, "h3studio_entry.py": LAUNCHER_SHA256},
            "model_sha256": MODEL_SHA256,
            "runtime": status["runtime"],
            "inference": {
                "actual_neural_execution": True, "frames": result["source_frames"],
                "width": result["width"], "height": result["height"],
                "attention": result["attention"], "precision": result["precision"],
            },
        }
        temporary = checkpoints / f".{RECEIPT_NAME}.tmp"
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, checkpoints / RECEIPT_NAME)
        return {**result, "selfcheck_receipt": str(checkpoints / RECEIPT_NAME)}

    def _run_verified(
        self, source: Path, output: Path, settings: dict[str, Any], progress: ProgressCallback,
        cancel, runtime: dict[str, Any],
    ) -> dict[str, Any]:
        target_width, target_height, expected_frames, rate = _validate_settings(settings)
        repo, checkpoints, python = configured_paths(self.root)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.seedvr2-working.mp4")
        temporary.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="h3studio-seedvr2-", dir=output.parent) as directory:
            work = Path(directory)
            staged = work / "source.mp4"
            shutil.copyfile(source, staged)
            render_root = work / "restored"
            command = [
                str(python), "-X", "utf8", str(repo / "h3studio_entry.py"), str(staged),
                "--output", str(render_root), "--output_format", "png",
                "--model_dir", str(checkpoints), "--dit_model", DIT_MODEL,
                "--resolution", str(min(target_width, target_height)),
                "--max_resolution", str(max(target_width, target_height)),
                "--batch_size", "5", "--uniform_batch_size", "--chunk_size", "85",
                "--temporal_overlap", "1", "--seed", str(int(settings.get("seed", 42))),
                "--cuda_device", "0", "--attention_mode", "sdpa",
                "--tensor_offload_device", "cpu", "--vae_encode_tiled", "--vae_decode_tiled",
            ]
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(repo), environment.get("PYTHONPATH", "")) if part
            )
            environment["PYTHONUTF8"] = "1"
            progress(0.03, "隔離SeedVR2 FP16ランタイムを開始しています")
            _run_cancellable(
                command, repo, environment, cancel, "SeedVR2で1-step復元しています", progress, 0.12,
            )
            frame_directory = render_root / staged.stem
            _inspect_png_sequence(frame_directory, expected_frames, target_width, target_height)
            progress(0.88, "元の有理FPSと音声でMP4を構築しています")
            pattern = frame_directory / f"{staged.stem}_%06d.png"
            fps_text = f"{rate.numerator}/{rate.denominator}"
            command = [
                _ffmpeg_path(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", fps_text, "-start_number", "0", "-i", str(pattern), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a?", "-frames:v", str(expected_frames),
                "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
                "-fps_mode", "cfr", "-c:a", "copy", "-map_metadata", "1", "-movflags", "+faststart",
                str(temporary),
            ]
            try:
                _run_cancellable(
                    command, output.parent, os.environ.copy(), cancel,
                    "SeedVR2のMP4と音声を構築しています", progress, 0.93,
                )
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

        if cancel.is_set():
            temporary.unlink(missing_ok=True)
            raise GenerationCancelled("cancelled before SeedVR2 publication")
        try:
            restored = inspect_video(temporary)
            if (int(restored["width"]), int(restored["height"])) != (target_width, target_height):
                raise RuntimeError("SeedVR2 publication geometry validation failed")
            if int(restored["frame_count"]) != expected_frames:
                raise RuntimeError("SeedVR2 publication frame-count validation failed")
            restored_rate = Fraction(int(restored["fps_numerator"]), int(restored["fps_denominator"]))
            if restored_rate != rate:
                raise RuntimeError(f"SeedVR2 publication FPS mismatch: {restored_rate} != {rate}")
            if bool(restored["has_audio"]) != bool(settings["source"]["has_audio"]):
                raise RuntimeError("SeedVR2 publication audio-stream preservation check failed")
            os.replace(temporary, output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        progress(1.0, "SeedVR2復元が完了しました")
        return {
            "engine": "numz SeedVR2 standalone CLI", "method": "seedvr2_3b",
            "kind": "ai-restoration", "repository": AUDITED_REPOSITORY,
            "repository_revision": AUDITED_REVISION, "model": DIT_MODEL,
            "model_sha256": MODEL_SHA256[DIT_MODEL], "vae_model": VAE_MODEL,
            "vae_sha256": MODEL_SHA256[VAE_MODEL], "scale": 2,
            "width": target_width, "height": target_height, "source_frames": expected_frames,
            "output_frames": expected_frames, "source_fps": f"{rate.numerator}/{rate.denominator}",
            "output_fps": f"{rate.numerator}/{rate.denominator}",
            "timing": "exact_source_rational_cfr", "audio": "stream-copy",
            "precision": "fp16-weights/bfloat16-compute", "attention": "pytorch-sdpa",
            "runtime": runtime,
        }
