import pytest
from pydantic import ValidationError

from h3studio.schemas import GenerationRequest, InterpolateRequest, UpscaleRequest


BASE = {"prompt": "雨の東京。遠くで電車の音。", "model": "MiniMax-H3"}


def test_geometry_and_seed_are_resolved_once():
    request = GenerationRequest(**BASE, width=1337, height=770, frames=125, seed=42)
    resolved = request.resolved()
    assert (resolved["width"], resolved["height"]) == (1344, 768)
    assert resolved["frames"] == 141
    assert resolved["duration_seconds"] == 141 / 24
    assert resolved["seed"] == 42
    assert resolved["sigma_grid_points"] == 21


def test_ordered_loras_preserve_disabled_entries_and_weights():
    request = GenerationRequest(
        **BASE,
        loras=[
            {"path": "minimaxH3/style-a.safetensors", "weight": 0.7, "enabled": True},
            {"path": "minimaxH3/style-b.safetensors", "weight": -0.25, "enabled": False},
        ],
    )
    assert [(x["path"], x["weight"], x["enabled"]) for x in request.resolved()["loras"]] == [
        ("minimaxH3/style-a.safetensors", 0.7, True),
        ("minimaxH3/style-b.safetensors", -0.25, False),
    ]


@pytest.mark.parametrize(
    ("preset", "steps", "path"),
    [
        ("turbo8", 8, "minimaxH3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
        ("turbo4", 4, "minimaxH3/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
    ],
)
def test_turbo_pair_is_explicit(preset, steps, path):
    request = GenerationRequest(**BASE, preset=preset, steps=steps, loras=[{"path": path}])
    assert request.preset == preset


def test_turbo_mismatch_is_rejected():
    with pytest.raises(ValidationError, match="verified matching"):
        GenerationRequest(
            **BASE, preset="turbo8", steps=8,
            loras=[{"path": "minimaxH3/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"}],
        )


def test_last_frame_only_is_supported():
    request = GenerationRequest(**BASE, last_frame="abc.png")
    assert request.last_frame == "abc.png"


def test_latent_two_pass_resolves_low_canvas_and_keeps_target():
    request = GenerationRequest(
        **BASE, width=960, height=544,
        latent_refine={
            "enabled": True, "model": "3d_conv_v1_fp16.safetensors",
            "scale": 2, "strength": 0.35, "steps": 6,
        },
    )
    resolved = request.resolved()
    assert (resolved["width"], resolved["height"]) == (960, 544)
    assert (resolved["latent_refine"]["low_width"], resolved["latent_refine"]["low_height"]) == (480, 288)
    assert resolved["latent_refine"]["target_width"] == 960
    assert resolved["latent_refine"]["target_height"] == 544


def test_latent_two_pass_defaults_match_validated_recipe():
    request = GenerationRequest(
        **BASE,
        latent_refine={"enabled": True, "model": "3d_conv_v1_fp16.safetensors"},
    )
    assert request.latent_refine.strength == 0.18
    assert request.latent_refine.steps == 4


def test_enabled_latent_two_pass_requires_safe_model():
    with pytest.raises(ValidationError, match="requires a model"):
        GenerationRequest(**BASE, latent_refine={"enabled": True})
    with pytest.raises(ValidationError, match="safe relative path"):
        GenerationRequest(**BASE, latent_refine={"enabled": True, "model": "../model.safetensors"})


def test_postprocess_source_is_exactly_one_and_factors_are_bounded():
    upscale = UpscaleRequest(source={"job_id": "job-1"}, method="realesrgan_x2plus", scale=2)
    assert upscale.source.job_id == "job-1"
    interpolation = InterpolateRequest(source={"upload_id": "video-1.mp4"}, factor=4)
    assert interpolation.factor == 4
    with pytest.raises(ValidationError, match="exactly one"):
        UpscaleRequest(source={"job_id": "job-1", "upload_id": "video-1.mp4"})
    with pytest.raises(ValidationError):
        InterpolateRequest(source={"upload_id": "video-1.mp4"}, factor=8)
