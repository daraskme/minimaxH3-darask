from __future__ import annotations

"""Standalone MiniMax H3 learned latent upscaler.

The network layout and normalization contract follow LBH-123-AI's Apache-2.0
``Minimax_h3_latent_Upscaler`` release. This module deliberately contains no
ComfyUI imports. Diffusers exposes H3's normalized 24-channel latents, which
still need the release model's additional channel-wise input transform. The
inverse transform returns the network output to Diffusers' latent domain.
"""

import gc
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


LATENT_CHANNELS = 24
VAE_SPATIAL_FACTOR = 16
LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)
SUPPORTED_SUFFIXES = {".safetensors"}
RELEASE_REVISION = "3f941d5"
RELEASE_ARCHITECTURE = {
    "in_channels": 24, "channels": 512, "in_blocks": 12, "out_blocks": 12,
    "temporal_every": 2, "temporal_kernel": 5, "attention": False,
}
RELEASE_SHA256 = {
    "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6",  # bf16
    "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2",  # fp16
}


def _normalization_tensors(reference):
    """Return the pinned release statistics in B,C,T,H,W broadcast layout."""
    import torch

    mean = torch.tensor(LATENTS_MEAN, device=reference.device, dtype=reference.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=reference.device, dtype=reference.dtype).view(1, -1, 1, 1, 1)
    return mean, std


def normalize_for_upscaler(latents):
    """Map Diffusers H3 latents into the learned upscaler's training domain."""
    if latents.ndim != 5 or latents.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"expected Bx24xTxHxW H3 latents, got {tuple(latents.shape)}")
    mean, std = _normalization_tensors(latents)
    return (latents - mean) / std


def denormalize_from_upscaler(latents):
    """Map the learned upscaler output back into Diffusers' H3 latent domain."""
    if latents.ndim != 5 or latents.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"expected Bx24xTxHxW H3 latents, got {tuple(latents.shape)}")
    mean, std = _normalization_tensors(latents)
    return latents * std + mean


@lru_cache(maxsize=16)
def _sha256_cached(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def safe_weight_path(root: Path, relative: str) -> Path:
    base = root.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise FileNotFoundError(f"Unknown H3 latent upscaler: {relative}")
    return path


@lru_cache(maxsize=16)
def inspect_checkpoint(path_text: str, size: int, modified_ns: int) -> dict[str, Any]:
    """Validate the released architecture from safetensors headers only."""
    del size, modified_ns
    from safetensors import safe_open

    path = Path(path_text)
    with safe_open(path, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        prefix = "upscaler." if any(key.startswith("upscaler.") for key in keys) else ""

        def shape(name: str) -> tuple[int, ...]:
            key = prefix + name
            if key not in keys:
                raise ValueError(f"missing tensor: {key}")
            return tuple(weights.get_slice(key).get_shape())

        conv_in = shape("conv_in.weight")
        conv_out = shape("conv_out.weight")
        if len(conv_in) != 5 or conv_in[1] != LATENT_CHANNELS:
            raise ValueError(f"expected a 24-channel Conv3d input, got {conv_in}")
        if len(conv_out) != 5 or conv_out[0] != LATENT_CHANNELS or conv_out[1] != conv_in[0]:
            raise ValueError(f"expected a 24-channel Conv3d output, got {conv_out}")
        channels = conv_in[0]
        in_ids = {
            int(match.group(1)) for key in keys
            if (match := re.match(rf"{re.escape(prefix)}in_blocks\.(\d+)\.in_layers\.", key))
        }
        out_ids = {
            int(match.group(1)) for key in keys
            if (match := re.match(rf"{re.escape(prefix)}out_blocks\.(\d+)\.in_layers\.", key))
        }
        temporal = [key for key in keys if key.startswith(prefix) and key.endswith("dwconv.weight")]
        if not in_ids or not out_ids:
            raise ValueError("checkpoint does not contain the expected residual blocks")
        temporal_kernel = shape(temporal[0][len(prefix):])[2] if temporal else 0
        return {
            "prefix": prefix,
            "in_channels": LATENT_CHANNELS,
            "channels": channels,
            "in_blocks": len(in_ids),
            "out_blocks": len(out_ids),
            "temporal_every": 2 if temporal else 0,
            "temporal_kernel": temporal_kernel,
            "attention": any("attn" in key for key in keys),
            "format": "safetensors",
        }


def checkpoint_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return inspect_checkpoint(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def validate_release(path: Path) -> dict[str, Any]:
    architecture = checkpoint_info(path)
    mismatch = {
        key: (architecture.get(key), expected)
        for key, expected in RELEASE_ARCHITECTURE.items()
        if architecture.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"unsupported latent-upscaler architecture: {mismatch}")
    digest = _sha256(path)
    if digest not in RELEASE_SHA256:
        raise ValueError(f"checkpoint SHA256 is not a pinned 3D Conv v1 release: {digest}")
    return {**architecture, "sha256": digest, "release_revision": RELEASE_REVISION}


def validated_weight_path(root: Path, relative: str) -> Path:
    path = safe_weight_path(root, relative)
    validate_release(path)
    return path


def scan_latent_upscalers(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.safetensors"), key=lambda item: str(item).lower()):
        relative = path.relative_to(root).as_posix()
        try:
            architecture = validate_release(path)
            ready, reason = True, None
        except Exception as exc:
            architecture, ready, reason = None, False, str(exc)
        result.append({
            "name": relative, "size": path.stat().st_size, "ready": ready,
            "reason": reason, "architecture": architecture,
        })
    return result


def resolve_low_resolution(width: int, height: int, scale: float = 2.0) -> tuple[int, int]:
    """Choose a legal H3 first-pass canvas nearest to target/scale."""
    if not 1.0 < scale <= 4.0:
        raise ValueError("latent upscale must be greater than 1x and at most 4x")
    # Round half up so a target such as 960x544 resolves to 480x288 instead
    # of Python's banker's-rounding surprise at 272/32 == 8.5.
    low_width = max(256, int(width / scale / 32 + 0.5) * 32)
    low_height = max(256, int(height / scale / 32 + 0.5) * 32)
    return low_width, low_height


def build_model(architecture: dict[str, Any]):
    """Build the released pure-3D network lazily to keep web startup light."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    def normalization(channels: int):
        return nn.GroupNorm(32, channels)

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, embedding_channels: int, dropout: float = 0.1):
            super().__init__()
            self.in_layers = nn.Sequential(
                normalization(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1)
            )
            self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embedding_channels, 2 * channels))
            self.out_norm = normalization(channels)
            self.out_layers = nn.Sequential(
                nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(channels, channels, 3, padding=1)
            )

        def forward(self, tensor, embedding):
            hidden = self.in_layers(tensor)
            scale, shift = torch.chunk(self.emb_layers(embedding).type_as(hidden), 2, dim=1)
            while scale.ndim < hidden.ndim:
                scale, shift = scale[..., None], shift[..., None]
            hidden = self.out_norm(hidden) * (1 + scale) + shift
            return tensor + self.out_layers(hidden)

    class TemporalBlock(nn.Module):
        def __init__(self, channels: int, kernel: int):
            super().__init__()
            self.norm = normalization(channels)
            self.dwconv = nn.Conv3d(
                channels, channels, kernel_size=(kernel, 1, 1), padding=(kernel // 2, 0, 0), groups=channels
            )
            self.pwconv = nn.Conv3d(channels, channels, 1)

        def forward(self, tensor):
            return tensor + self.pwconv(self.dwconv(functional.silu(self.norm(tensor))))

    class LatentResizer3D(nn.Module):
        def __init__(self):
            super().__init__()
            channels = int(architecture["channels"])
            self.conv_in = nn.Conv3d(LATENT_CHANNELS, channels, 3, padding=1)
            self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
            self.in_blocks = self._blocks(
                channels, int(architecture["in_blocks"]), int(architecture["temporal_every"]),
                int(architecture["temporal_kernel"]), ResidualBlock, TemporalBlock,
            )
            self.out_blocks = self._blocks(
                channels, int(architecture["out_blocks"]), int(architecture["temporal_every"]),
                int(architecture["temporal_kernel"]), ResidualBlock, TemporalBlock,
            )
            self.norm_out = normalization(channels)
            self.conv_out = nn.Conv3d(channels, LATENT_CHANNELS, 3, padding=1)

        @staticmethod
        def _blocks(channels, count, temporal_every, temporal_kernel, residual, temporal):
            blocks = nn.ModuleList()
            for index in range(count):
                blocks.append(residual(channels, 64))
                if temporal_every > 0 and index % temporal_every == 0:
                    blocks.append(temporal(channels, temporal_kernel))
            return blocks

        def segment(self, tensor, effective_scale: float, target: tuple[int, int, int]):
            embedding = self.embed(torch.tensor([[effective_scale - 1]], device=tensor.device, dtype=tensor.dtype))
            hidden = self.conv_in(tensor)
            for block in self.in_blocks:
                hidden = block(hidden, embedding) if isinstance(block, ResidualBlock) else block(hidden)
            hidden = functional.interpolate(hidden, size=target, mode="trilinear", align_corners=False)
            for block in self.out_blocks:
                hidden = block(hidden, embedding) if isinstance(block, ResidualBlock) else block(hidden)
            return self.conv_out(functional.silu(self.norm_out(hidden)))

        def forward(self, tensor, target_height: int, target_width: int, chunk_frames: int = 32):
            if tensor.ndim != 5 or tensor.shape[1] != LATENT_CHANNELS:
                raise ValueError(f"expected Bx24xTxHxW H3 latents, got {tuple(tensor.shape)}")
            batch, channels, frames, height, width = tensor.shape
            target = (frames, target_height, target_width)
            effective_scale = ((target_height / height) + (target_width / width)) / 2
            if not 1.0 <= effective_scale <= 4.0:
                raise ValueError(f"unsupported latent scale: {effective_scale:.3f}")
            temporal_kernel = int(architecture["temporal_kernel"])
            if frames <= chunk_frames or temporal_kernel <= 0:
                return self.segment(tensor, effective_scale, target)
            overlap = temporal_kernel
            padded = functional.pad(tensor, (0, 0, 0, 0, overlap, overlap), mode="replicate")
            output = torch.zeros(
                batch, channels, frames, target_height, target_width, device=tensor.device, dtype=tensor.dtype
            )
            weights = torch.zeros(1, 1, frames, 1, 1, device=tensor.device, dtype=tensor.dtype)
            for start in range(0, frames, chunk_frames):
                core_end = min(frames, start + chunk_frames)
                output_start, output_end = max(0, start - overlap), min(frames, core_end + overlap)
                source_start = max(0, output_start - overlap)
                source_end = min(frames + 2 * overlap, output_end + overlap)
                segment = padded[:, :, source_start:source_end].contiguous()
                resized = self.segment(segment, effective_scale, (segment.shape[2], target_height, target_width))
                valid_start = (output_start + overlap) - source_start
                valid = resized[:, :, valid_start:valid_start + output_end - output_start]
                blend = torch.ones(output_end - output_start, device=tensor.device, dtype=tensor.dtype)
                if start > output_start:
                    length = start - output_start
                    blend[:length] = torch.arange(1, length + 1, device=tensor.device, dtype=tensor.dtype) / (length + 1)
                if output_end > core_end:
                    length = output_end - core_end
                    blend[-length:] = torch.arange(length, 0, -1, device=tensor.device, dtype=tensor.dtype) / (length + 1)
                blend = blend.view(1, 1, -1, 1, 1)
                output[:, :, output_start:output_end] += valid * blend
                weights[:, :, output_start:output_end] += blend
            return output / weights.clamp_min(1e-8)

    return LatentResizer3D()


class H3LatentUpscaler:
    def __init__(self, root: Path):
        self.root = root

    def upscale(
        self, latents, relative: str, target_height: int, target_width: int,
        progress: Callable[[float, str], None] | None = None, cancel=None,
    ):
        import torch
        from safetensors.torch import load_file

        path = validated_weight_path(self.root, relative)
        architecture = checkpoint_info(path)
        if architecture["attention"]:
            raise RuntimeError("attention-bearing latent upscaler checkpoints are not supported")
        if cancel is not None and cancel.is_set():
            raise RuntimeError("cancelled before latent upscale")
        if progress:
            progress(0.0, "24ch潜在アップスケーラーを読み込んでいます")
        state = load_file(str(path), device="cpu")
        prefix = architecture["prefix"]
        if prefix:
            state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        model = build_model(architecture)
        model.load_state_dict(state, strict=True)
        del state
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = model.to(device="cuda", dtype=dtype).eval().requires_grad_(False)
        try:
            if cancel is not None and cancel.is_set():
                raise RuntimeError("cancelled before latent upscale")
            if progress:
                progress(0.25, "潜在空間を拡大しています")
            with torch.inference_mode():
                model_input = normalize_for_upscaler(latents.to(device="cuda", dtype=dtype))
                output = denormalize_from_upscaler(model(
                    model_input,
                    target_height=target_height, target_width=target_width,
                )).float().cpu()
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("latent upscaler produced non-finite values")
            if progress:
                progress(1.0, "潜在空間の拡大が完了しました")
            return output, {
                "model": relative, "model_sha256": _sha256(path), "architecture": architecture,
                "release_revision": RELEASE_REVISION,
                "input_shape": list(latents.shape), "output_shape": list(output.shape),
                "precision": str(dtype).removeprefix("torch."), "temporal_resize": False,
                "normalization": "release_24ch_mean_std_before_and_inverse_after",
            }
        finally:
            model.to("cpu")
            del model
            gc.collect()
            torch.cuda.empty_cache()
