from __future__ import annotations

import os
import json
import importlib.util
import queue
import subprocess
import threading
import time
from collections import deque
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .engine import GenerationCancelled
from .metadata import _ffmpeg_path


ProgressCallback = Callable[[float, str], None]
ROOT = Path(__file__).resolve().parents[1]
REALESRGAN_MODEL = ROOT / "models" / "upscalers" / "RealESRGAN_x2plus.pth"
RIFE_MODEL = ROOT / "models" / "interpolators" / "rife4.26" / "flownet.pkl"
REALESRGAN_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
RIFE_SHA256 = "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"


@lru_cache(maxsize=8)
def _verified_weight(path_text: str, size: int, modified_ns: int, expected: str) -> bool:
    from .video import sha256_file

    path = Path(path_text)
    return path.is_file() and path.stat().st_size == size and path.stat().st_mtime_ns == modified_ns and sha256_file(path) == expected


def _weight_ready(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    stat = path.stat()
    return _verified_weight(str(path), stat.st_size, stat.st_mtime_ns, expected)


def capabilities() -> dict[str, Any]:
    from .dlss import capabilities as dlss_capabilities
    from .seedvr2 import capability as seedvr2_capability

    realesrgan_ready = _weight_ready(REALESRGAN_MODEL, REALESRGAN_SHA256) and importlib.util.find_spec("spandrel") is not None
    rife_ready = _weight_ready(RIFE_MODEL, RIFE_SHA256)
    return {
        "upscale": {
            "default_method": "realesrgan_x2plus" if realesrgan_ready else "lanczos",
            "methods": [
                {
                    "id": "lanczos", "label": "Lanczos", "available": True,
                    "kind": "conventional", "factors": [2, 3, 4],
                    "description": "高品質な従来補間。AI超解像ではありません。",
                },
                {
                    "id": "realesrgan_x2plus", "label": "Real-ESRGAN x2plus", "available": realesrgan_ready,
                    "kind": "ai", "factors": [2],
                    **({} if realesrgan_ready else {"reason": "モデルまたはSpandrelが未導入です。"}),
                },
                seedvr2_capability(ROOT),
                *dlss_capabilities(),
            ],
        },
        "interpolate": {
            "default_method": "rife_v4_26" if rife_ready else None,
            "methods": [
                {
                    "id": "rife_v4_26", "label": "RIFE 4.26", "available": rife_ready,
                    "kind": "ai", "factors": [2, 3, 4],
                    **({} if rife_ready else {"reason": "RIFE 4.26モデルが未導入です。"}),
                },
            ],
        },
    }


class VideoPostprocessEngine:
    def __init__(self, ffmpeg: str | None = None):
        self.ffmpeg = ffmpeg

    def run(
        self, kind: str, settings: dict[str, Any], source: Path, output: Path,
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        if kind == "upscale" and settings.get("method") == "lanczos":
            return self._lanczos(source, output, int(settings["scale"]), settings["source"], progress, cancel)
        if kind == "upscale" and settings.get("method") == "realesrgan_x2plus":
            if int(settings["scale"]) != 2:
                raise RuntimeError("Real-ESRGAN x2plus supports only 2x")
            return self._realesrgan(source, output, settings, progress, cancel)
        if kind == "upscale" and settings.get("method") == "seedvr2_3b":
            from .seedvr2 import SeedVR2Engine

            return SeedVR2Engine(ROOT, self.ffmpeg).run(source, output, settings, progress, cancel)
        if kind == "upscale" and settings.get("method") == "dlss_super_resolution":
            if int(settings["scale"]) != 2:
                raise RuntimeError("NVIDIA DLSS Super Resolution supports only 2x in this release")
            return self._dlss_super_resolution(source, output, settings, progress, cancel)
        if kind == "upscale" and settings.get("method") == "dlss5_neural_rendering":
            if int(settings["scale"]) != 1:
                raise RuntimeError("DLSS 5 Neural Rendering writes at source resolution (1x)")
            return self._dlss_neural_rendering(source, output, settings, progress, cancel)
        if kind == "interpolate" and settings.get("method") == "rife_v4_26":
            return self._rife(source, output, settings, progress, cancel)
        raise RuntimeError(f"Postprocess method is not available: {kind}/{settings.get('method')}")

    def _dlss_super_resolution(
        self, source: Path, output: Path, settings: dict[str, Any],
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        from .dlss import (
            BUILD_MANIFEST, OFFICIAL_DLSS_COMMIT, OFFICIAL_DLSS_SR_SHA256, PLAYER_COMMIT,
            SR_WORKER, driver_version, run_super_resolution,
        )
        from .video import sha256_file

        silent = output.with_name(f".{output.stem}.dlss-silent.mp4")
        receipt_path = output.with_name(f".{output.stem}.dlss-receipt.json")
        temporary = output.with_name(f".{output.stem}.working.mp4")
        for path in (silent, receipt_path, temporary):
            path.unlink(missing_ok=True)
        source_info = settings["source"]
        ffmpeg = _ffmpeg_path(self.ffmpeg)
        try:
            receipt = run_super_resolution(
                source, silent, receipt_path, source_info, ffmpeg, progress, cancel,
            )
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(silent), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                "-c", "copy", "-map_metadata", "1", "-movflags", "+faststart",
                "-progress", "pipe:1", "-nostats", str(temporary),
            ]
            self._ffmpeg_progress(
                command, float(source_info["duration_seconds"]), progress, cancel,
                "音声を保持しています", start=0.88, span=0.10,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("DLSS SR audio remux produced no video")
            os.replace(temporary, output)
            manifest = json.loads(BUILD_MANIFEST.read_text(encoding="utf-8-sig"))
            return {
                "engine": "nvidia-official-dlss-sdk", "method": "dlss_super_resolution",
                "kind": "ai", "scale": 2, "feature": "super_resolution",
                "not_neural_rendering": True, "driver": driver_version(),
                "player_source_commit": PLAYER_COMMIT, "nvidia_dlss_commit": OFFICIAL_DLSS_COMMIT,
                "worker_sha256": sha256_file(SR_WORKER),
                "runtime_sha256": OFFICIAL_DLSS_SR_SHA256,
                "smoke_validation": manifest.get("smoke_validation"),
                "receipt": receipt, "output_codec": "h264", "audio": "stream-copy",
                "timing": "source_cfr_preserved",
            }
        finally:
            silent.unlink(missing_ok=True)
            receipt_path.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)

    def _dlss_neural_rendering(
        self, source: Path, output: Path, settings: dict[str, Any],
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        from .dlss import (
            NR_LOCK, NR_VALIDATION, NR_WORKER, PLAYER_COMMIT, driver_version, run_neural_rendering,
        )
        from .video import sha256_file

        staging = output.with_name(f".{output.stem}.dlss5-nr.mkv")
        temporary = output.with_name(f".{output.stem}.working.mp4")
        staging.unlink(missing_ok=True); temporary.unlink(missing_ok=True)
        source_info = settings["source"]
        rate = f"{int(source_info['fps_numerator'])}/{int(source_info['fps_denominator'])}"
        try:
            receipt = run_neural_rendering(source, staging, source_info, progress, cancel)
            # The exact-tag worker writes Matroska with a millisecond time base.
            # Decode its genuine NR frames once and stamp the source's exact CFR
            # grid while encoding the public H.264 MP4. Audio remains stream-copy.
            command = [
                _ffmpeg_path(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(staging), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                "-vf", f"fps={rate}", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-map_metadata", "1",
                "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(temporary),
            ]
            self._ffmpeg_progress(
                command, float(source_info["duration_seconds"]), progress, cancel,
                "DLSS 5 NRを正確なタイムラインへ保存しています", start=0.86, span=0.12,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("DLSS 5 Neural Rendering finalization produced no MP4")
            os.replace(temporary, output)
            validation = json.loads(NR_VALIDATION.read_text(encoding="utf-8"))
            return {
                "engine": "community-dlss5-neural-rendering", "method": "dlss5_neural_rendering",
                "kind": "experimental_ai", "scale": 1, "feature": "neural_rendering",
                "not_super_resolution": True, "driver": driver_version(),
                "player_source_commit": PLAYER_COMMIT, "worker_sha256": sha256_file(NR_WORKER),
                "runtime_lock_sha256": sha256_file(NR_LOCK), "trust": "unsigned_community_modified_runtime",
                "validation": validation, "receipt": receipt,
                "output_codec": "h264", "audio": "stream-copy",
                "timing": "source_cfr_restamped_after_verified_nr_frames",
            }
        finally:
            staging.unlink(missing_ok=True); temporary.unlink(missing_ok=True)

    def _ffmpeg_progress(
        self, command: list[str], duration: float, progress: ProgressCallback, cancel,
        phase: str, start: float = 0.05, span: float = 0.85,
    ) -> None:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=False)
        messages: queue.Queue[str | None] = queue.Queue()
        errors: deque[str] = deque(maxlen=200)

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                messages.put(line)
            messages.put(None)

        def read_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                errors.append(line)

        readers = [threading.Thread(target=read_stdout), threading.Thread(target=read_stderr)]
        for reader in readers:
            reader.start()
        stdout_done = False
        try:
            while not (stdout_done and process.poll() is not None):
                if cancel.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=5)
                    raise GenerationCancelled(f"cancelled during {phase}")
                try:
                    line = messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    stdout_done = True
                    continue
                key, _, value = line.strip().partition("=")
                if key in {"out_time_us", "out_time_ms"} and duration > 0:
                    seconds = float(value or 0) / 1_000_000
                    progress(start + span * min(1.0, seconds / duration), phase)
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"ffmpeg failed: {''.join(errors).strip()}")
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)
            for reader in readers:
                reader.join(timeout=2)

    def _lanczos(
        self, source: Path, output: Path, scale: int, source_info: dict[str, Any],
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        duration = float(source_info["duration_seconds"])
        rate = f"{int(source_info['fps_numerator'])}/{int(source_info['fps_denominator'])}"
        temporary = output.with_name(f".{output.stem}.working.mp4")
        temporary.unlink(missing_ok=True)
        command = [
            _ffmpeg_path(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
            "-vf", f"fps={rate},scale=iw*{scale}:ih*{scale}:flags=lanczos",
            "-c:v", "libx264", "-preset", "slow", "-crf", "14", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-map_metadata", "0", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(temporary),
        ]
        try:
            self._ffmpeg_progress(command, duration, progress, cancel, "Lanczosで拡大しています")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("ffmpeg Lanczos upscale produced no video")
            os.replace(temporary, output)
            return {
                "engine": "ffmpeg", "method": "lanczos", "kind": "conventional", "scale": scale,
                "timing": "normalized_cfr_at_average_rate", "fps": rate, "audio": "stream-copy",
            }
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _upscale_tensor(model, tensor, scale: int = 2):
        import torch

        _, _, height, width = tensor.shape
        if height * width <= 2_500_000:
            return model(tensor).clamp_(0, 1)
        core, overlap = 768, 32
        result = torch.empty((1, 3, height * scale, width * scale), dtype=torch.uint8, device="cpu")
        for top in range(0, height, core):
            for left in range(0, width, core):
                bottom, right = min(height, top + core), min(width, left + core)
                in_top, in_left = max(0, top - overlap), max(0, left - overlap)
                in_bottom, in_right = min(height, bottom + overlap), min(width, right + overlap)
                tile = model(tensor[:, :, in_top:in_bottom, in_left:in_right]).clamp_(0, 1)
                crop_top, crop_left = (top - in_top) * scale, (left - in_left) * scale
                crop_bottom = crop_top + (bottom - top) * scale
                crop_right = crop_left + (right - left) * scale
                result[:, :, top * scale:bottom * scale, left * scale:right * scale] = (
                    tile[:, :, crop_top:crop_bottom, crop_left:crop_right].mul(255).round().to(torch.uint8).cpu()
                )
        return result.float().div_(255)

    def _realesrgan(
        self, source: Path, output: Path, settings: dict[str, Any],
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        import av
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
        from .video import sha256_file

        if not torch.cuda.is_available():
            raise RuntimeError("Real-ESRGAN requires a CUDA GPU")
        if not REALESRGAN_MODEL.is_file():
            raise RuntimeError(f"Real-ESRGAN model is missing: {REALESRGAN_MODEL}")
        if not _weight_ready(REALESRGAN_MODEL, REALESRGAN_SHA256):
            raise RuntimeError("Real-ESRGAN model failed SHA256 verification")
        descriptor = ModelLoader().load_from_file(REALESRGAN_MODEL)
        if not isinstance(descriptor, ImageModelDescriptor):
            raise RuntimeError("The selected upscaler is not an image-to-image model")
        if int(descriptor.scale) != 2:
            raise RuntimeError(f"Unexpected Real-ESRGAN scale: {descriptor.scale}")
        descriptor = descriptor.cuda().half().eval()
        silent = output.with_name(f".{output.stem}.silent.mp4")
        temporary = output.with_name(f".{output.stem}.working.mp4")
        silent.unlink(missing_ok=True); temporary.unlink(missing_ok=True)
        source_info = settings["source"]
        rate = Fraction(int(source_info["fps_numerator"]), int(source_info["fps_denominator"]))
        expected_frames = max(1, round(float(source_info["duration_seconds"]) * float(rate)))
        frame_count = 0
        try:
            with av.open(str(source), mode="r", format="mov", options={"protocol_whitelist": "file,pipe"}) as input_container, av.open(str(silent), mode="w") as output_container:
                input_stream = input_container.streams.video[0]
                output_stream = output_container.add_stream("libx264", rate=rate)
                output_stream.width = int(source_info["width"]) * 2
                output_stream.height = int(source_info["height"]) * 2
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"crf": "14", "preset": "slow"}
                with torch.inference_mode():
                    for index, frame in enumerate(input_container.decode(input_stream)):
                        if cancel.is_set():
                            raise GenerationCancelled("cancelled during Real-ESRGAN upscale")
                        array = frame.to_ndarray(format="rgb24")
                        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda().half().div_(255)
                        enhanced = self._upscale_tensor(descriptor, tensor, 2)
                        output_array = enhanced.squeeze(0).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()
                        output_frame = av.VideoFrame.from_ndarray(output_array, format="rgb24")
                        output_frame.pts = index
                        output_frame.time_base = Fraction(1, 1) / rate
                        for packet in output_stream.encode(output_frame):
                            output_container.mux(packet)
                        frame_count = index + 1
                        progress(0.05 + 0.82 * min(1.0, frame_count / expected_frames), f"Real-ESRGAN x2 · {frame_count}f")
                for packet in output_stream.encode():
                    output_container.mux(packet)
            if frame_count == 0:
                raise RuntimeError("Source video contained no decodable frames")
            command = [
                _ffmpeg_path(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(silent), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                "-c", "copy", "-map_metadata", "1", "-movflags", "+faststart",
                "-progress", "pipe:1", "-nostats", str(temporary),
            ]
            self._ffmpeg_progress(
                command, float(source_info["duration_seconds"]), progress, cancel,
                "音声を保持しています", start=0.88, span=0.10,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("Real-ESRGAN audio remux produced no video")
            os.replace(temporary, output)
            return {
                "engine": "spandrel", "version": "0.4.2", "method": "realesrgan_x2plus",
                "kind": "ai", "scale": 2, "precision": "float16", "frames": frame_count,
                "model": REALESRGAN_MODEL.name, "model_sha256": sha256_file(REALESRGAN_MODEL),
                "timing": "normalized_cfr_at_average_rate", "fps": f"{rate.numerator}/{rate.denominator}",
                "audio": "stream-copy",
            }
        finally:
            silent.unlink(missing_ok=True); temporary.unlink(missing_ok=True)
            try:
                descriptor.cpu()
            finally:
                del descriptor
                torch.cuda.empty_cache()

    @staticmethod
    def _load_rife_model():
        import torch
        from .vendor.rife.IFNet_HDv3 import IFNet

        if not RIFE_MODEL.is_file():
            raise RuntimeError(f"RIFE model is missing: {RIFE_MODEL}")
        if not _weight_ready(RIFE_MODEL, RIFE_SHA256):
            raise RuntimeError("RIFE model failed SHA256 verification")
        model = IFNet()
        checkpoint = torch.load(RIFE_MODEL, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError("RIFE checkpoint does not contain a state dictionary")
        cleaned = {}
        for key, value in checkpoint.items():
            name = key[7:] if key.startswith("module.") else key
            if name.startswith(("teacher.", "caltime.")):
                continue
            cleaned[name] = value
        # The official inference graph comments out exactly teacher/caltime.
        # Strict loading after that allowlist prevents silent partial models.
        model.load_state_dict(cleaned, strict=True)
        return model.cuda().eval()

    @staticmethod
    def _frame_tensor(frame):
        import torch

        array = frame.to_ndarray(format="rgb24")
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).cuda().float().div_(255)

    @staticmethod
    def _pad_rife(tensor):
        import torch.nn.functional as functional

        height, width = tensor.shape[-2:]
        pad_height = (128 - height % 128) % 128
        pad_width = (128 - width % 128) % 128
        return functional.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")

    def _rife(
        self, source: Path, output: Path, settings: dict[str, Any],
        progress: ProgressCallback, cancel,
    ) -> dict[str, Any]:
        import av
        import torch
        import torch.nn.functional as functional
        from .video import sha256_file

        if not torch.cuda.is_available():
            raise RuntimeError("RIFE requires a CUDA GPU")
        factor = int(settings["factor"])
        if factor not in {2, 3, 4}:
            raise RuntimeError("RIFE factor must be 2, 3 or 4")
        model = self._load_rife_model()
        source_info = settings["source"]
        source_rate = Fraction(int(source_info["fps_numerator"]), int(source_info["fps_denominator"]))
        output_rate = source_rate * factor
        expected_source_frames = max(1, int(source_info.get("frame_count") or round(float(source_info["duration_seconds"]) * float(source_rate))))
        silent = output.with_name(f".{output.stem}.silent.mp4")
        temporary = output.with_name(f".{output.stem}.working.mp4")
        silent.unlink(missing_ok=True); temporary.unlink(missing_ok=True)
        written = 0
        source_frames = 0
        scene_cuts = 0

        def to_video_frame(tensor, width: int, height: int):
            cropped = tensor[:, :, :height, :width]
            array = cropped.squeeze(0).permute(1, 2, 0).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            return av.VideoFrame.from_ndarray(array, format="rgb24")

        try:
            with av.open(str(source), mode="r", format="mov", options={"protocol_whitelist": "file,pipe"}) as input_container, av.open(str(silent), mode="w") as output_container:
                input_stream = input_container.streams.video[0]
                width, height = int(source_info["width"]), int(source_info["height"])
                output_stream = output_container.add_stream("libx264", rate=output_rate)
                output_stream.width = width; output_stream.height = height
                output_stream.pix_fmt = "yuv420p"
                output_stream.options = {"crf": "14", "preset": "slow"}

                def encode_tensor(tensor) -> None:
                    nonlocal written
                    frame = to_video_frame(tensor, width, height)
                    frame.pts = written; frame.time_base = Fraction(1, 1) / output_rate
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)
                    written += 1
                    progress(0.05 + 0.82 * min(1.0, written / (expected_source_frames * factor)), f"RIFE {factor}x · {written}f")

                decoded = iter(input_container.decode(input_stream))
                first = next(decoded, None)
                if first is None:
                    raise RuntimeError("Source video contained no decodable frames")
                previous = self._pad_rife(self._frame_tensor(first))
                source_frames = 1
                with torch.inference_mode():
                    for frame in decoded:
                        if cancel.is_set():
                            raise GenerationCancelled("cancelled during RIFE interpolation")
                        current = self._pad_rife(self._frame_tensor(frame))
                        encode_tensor(previous)
                        preview_previous = functional.interpolate(previous, size=(64, 64), mode="area")
                        preview_current = functional.interpolate(current, size=(64, 64), mode="area")
                        scene_cut = float((preview_previous - preview_current).abs().mean().item()) > 0.32
                        if scene_cut:
                            scene_cuts += 1
                        for step in range(1, factor):
                            if cancel.is_set():
                                raise GenerationCancelled("cancelled during RIFE interpolation")
                            if scene_cut:
                                interpolated = previous
                            else:
                                timestep = step / factor
                                merged = model(
                                    torch.cat((previous, current), dim=1), timestep=timestep,
                                    scale_list=[16, 8, 4, 2, 1], training=False,
                                )[2][-1]
                                interpolated = merged.clamp(0, 1)
                            encode_tensor(interpolated)
                        previous = current; source_frames += 1
                    encode_tensor(previous)
                    for _ in range(factor - 1):
                        encode_tensor(previous)
                for packet in output_stream.encode():
                    output_container.mux(packet)
            command = [
                _ffmpeg_path(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(silent), "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                "-c", "copy", "-map_metadata", "1", "-movflags", "+faststart",
                "-progress", "pipe:1", "-nostats", str(temporary),
            ]
            self._ffmpeg_progress(
                command, float(source_info["duration_seconds"]), progress, cancel,
                "音声を保持しています", start=0.88, span=0.10,
            )
            if written != source_frames * factor:
                raise RuntimeError(f"RIFE frame invariant failed: {written} != {source_frames}*{factor}")
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("RIFE audio remux produced no video")
            os.replace(temporary, output)
            return {
                "engine": "official-rife-graph", "method": "rife_v4_26", "kind": "ai",
                "factor": factor, "precision": "float32", "source_frames": source_frames,
                "output_frames": written, "scene_cuts": scene_cuts, "scene_cut_policy": "hold_previous",
                "scene_cut_threshold_mean_absolute_rgb": 0.32,
                "model": RIFE_MODEL.name, "model_sha256": sha256_file(RIFE_MODEL),
                "timing": "cfr_integer_multiplier_with_final_tail_hold",
                "source_fps": f"{source_rate.numerator}/{source_rate.denominator}",
                "output_fps": f"{output_rate.numerator}/{output_rate.denominator}", "audio": "stream-copy",
            }
        finally:
            silent.unlink(missing_ok=True); temporary.unlink(missing_ok=True)
            try:
                model.cpu()
            finally:
                del model
                from .vendor.rife.warplayer import backwarp_tenGrid
                backwarp_tenGrid.clear()
                torch.cuda.empty_cache()
