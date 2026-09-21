from h3studio.generation_options import (
    FRAME_ALIGNMENT,
    FRAME_PRESETS,
    FRAME_REMAINDER,
    MAX_FRAMES,
    NATIVE_FPS,
    SIZE_PRESETS,
    align_frames,
    public_options,
)


def test_recommended_canvases_are_h3_aligned():
    assert {item["id"] for item in SIZE_PRESETS} == {
        "landscape_fast", "portrait_fast", "square_fast",
        "landscape_detail", "portrait_detail", "square_detail",
    }
    assert all(item["width"] % 32 == 0 and item["height"] % 32 == 0 for item in SIZE_PRESETS)
    assert SIZE_PRESETS[0]["width"] == 960 and SIZE_PRESETS[0]["height"] == 544


def test_frame_presets_follow_native_grid_and_report_seconds():
    assert [item["frames"] for item in FRAME_PRESETS] == [124, 243, 345]
    assert all((item["frames"] - FRAME_REMAINDER) % FRAME_ALIGNMENT == 0 for item in FRAME_PRESETS)
    assert all(item["seconds"] == item["frames"] / NATIVE_FPS for item in FRAME_PRESETS)
    assert align_frames(125) == 141
    assert align_frames(999) == MAX_FRAMES
    options = public_options()
    assert options["native_fps"] == 24
    assert options["frame_alignment"] == 17
    assert options["frame_remainder"] == 5
