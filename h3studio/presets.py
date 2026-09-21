from __future__ import annotations

from pathlib import PurePosixPath


# Only artifacts whose Diffusers key conversion and full-model tensor shapes
# were inspected are admitted to a low-step preset. This prevents a Ref2VA or
# unrelated distilled adapter from silently receiving the wrong schedule.
ACCELERATION_LORAS: dict[str, dict[str, object]] = {
    "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors": {
        "preset": "turbo8", "workflow": "t2va_fl2va", "evaluations": 8,
        "video_flow_shift": 12.0, "audio_flow_shift": 3.0,
    },
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors": {
        "preset": "turbo4", "workflow": "t2va_fl2va", "evaluations": 4,
        "video_flow_shift": 6.0, "audio_flow_shift": 3.0,
    },
}

ACCELERATION_MARKERS = ("turbo", "lightx2v", "pdd", "fasth3", "taomate")


def acceleration_recipe(path: str) -> dict[str, object] | None:
    return ACCELERATION_LORAS.get(PurePosixPath(path.replace("\\", "/")).name.lower())


def looks_like_acceleration(path: str) -> bool:
    name = PurePosixPath(path.replace("\\", "/")).name.lower()
    return any(marker in name for marker in ACCELERATION_MARKERS)
