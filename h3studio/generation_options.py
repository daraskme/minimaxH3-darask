from __future__ import annotations


NATIVE_FPS = 24
MIN_FRAMES = 124
MAX_FRAMES = 345
FRAME_ALIGNMENT = 17
FRAME_REMAINDER = 5

SIZE_PRESETS = (
    {"id": "landscape_fast", "label": "横・高速", "width": 960, "height": 544, "tier": "fast"},
    {"id": "portrait_fast", "label": "縦・高速", "width": 544, "height": 960, "tier": "fast"},
    {"id": "square_fast", "label": "正方形・高速", "width": 704, "height": 704, "tier": "fast"},
    {"id": "landscape_detail", "label": "横・精細", "width": 1344, "height": 768, "tier": "detail"},
    {"id": "portrait_detail", "label": "縦・精細", "width": 768, "height": 1344, "tier": "detail"},
    {"id": "square_detail", "label": "正方形・精細", "width": 1024, "height": 1024, "tier": "detail"},
)
FRAME_PRESETS = (
    {"frames": 124, "label": "短い", "seconds": 124 / NATIVE_FPS},
    {"frames": 243, "label": "標準", "seconds": 243 / NATIVE_FPS},
    {"frames": 345, "label": "最大", "seconds": 345 / NATIVE_FPS},
)


def align_frames(value: int) -> int:
    frames = max(MIN_FRAMES, int(value))
    while frames % FRAME_ALIGNMENT != FRAME_REMAINDER:
        frames += 1
    return min(frames, MAX_FRAMES)


def public_options() -> dict[str, object]:
    return {
        "native_fps": NATIVE_FPS,
        "frame_rule": f"{FRAME_ALIGNMENT}k+{FRAME_REMAINDER}",
        "frame_alignment": FRAME_ALIGNMENT,
        "frame_remainder": FRAME_REMAINDER,
        "min_frames": MIN_FRAMES,
        "max_frames": MAX_FRAMES,
        "sizes": list(SIZE_PRESETS),
        "durations": list(FRAME_PRESETS),
    }
