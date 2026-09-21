from __future__ import annotations

import secrets
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .presets import acceleration_recipe, looks_like_acceleration
from .generation_options import MAX_FRAMES, MIN_FRAMES, NATIVE_FPS, align_frames


class LoraSpec(BaseModel):
    path: str
    weight: float = Field(default=1.0, ge=-4.0, le=4.0)
    enabled: bool = True

    @field_validator("path")
    @classmethod
    def nonempty_path(cls, value: str) -> str:
        value = value.strip().replace("\\", "/")
        if not value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("LoRA path must be a safe relative path")
        return value


class LatentRefineSpec(BaseModel):
    enabled: bool = False
    model: str | None = None
    scale: float = Field(default=2.0, ge=1.1, le=4.0)
    strength: float = Field(default=0.18, gt=0.0, le=0.6)
    steps: int = Field(default=4, ge=1, le=20)

    @field_validator("model")
    @classmethod
    def safe_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().replace("\\", "/")
        if not value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("latent upscaler model must be a safe relative path")
        return value

    @model_validator(mode="after")
    def enabled_has_model(self) -> "LatentRefineSpec":
        if self.enabled and not self.model:
            raise ValueError("enabled latent refinement requires a model")
        return self


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    model: str
    first_frame: str | None = None
    last_frame: str | None = None
    width: int = Field(default=960, ge=256, le=2048)
    height: int = Field(default=544, ge=256, le=2048)
    frames: int = Field(default=MIN_FRAMES, ge=MIN_FRAMES, le=MAX_FRAMES)
    # Browser JSON numbers are exact through 2**53-1. Keep the persisted seed
    # exact across replay and the JavaScript UI.
    seed: int | None = Field(default=None, ge=0, le=2**53 - 1)
    steps: int = Field(default=20, ge=1, le=80)
    preset: Literal["balanced", "quality", "turbo8", "turbo4"] = "balanced"
    attention: Literal["sage", "sdpa"] = "sage"
    precision: Literal["int8"] = "int8"
    memory_profile: Literal["auto", "low_memory"] = "auto"
    loras: list[LoraSpec] = Field(default_factory=list, max_length=16)
    latent_refine: LatentRefineSpec = Field(default_factory=LatentRefineSpec)

    @field_validator("model")
    @classmethod
    def model_required(cls, value: str) -> str:
        value = value.strip().replace("\\", "/")
        if not value or value.startswith("/") or ".." in value.split("/"):
            raise ValueError("model must be a safe relative path")
        return value

    @model_validator(mode="after")
    def combinations(self) -> "GenerationRequest":
        candidates = [x for x in self.loras if x.enabled and looks_like_acceleration(x.path)]
        if len(candidates) > 1:
            raise ValueError("Select exactly one acceleration LoRA; Turbo/PDD/Fast adapters cannot be stacked")
        recipe = acceleration_recipe(candidates[0].path) if candidates else None
        if candidates and recipe is None:
            raise ValueError(f"This acceleration LoRA has no verified standalone recipe: {candidates[0].path}")
        if self.preset.startswith("turbo"):
            if recipe is None or recipe["preset"] != self.preset:
                raise ValueError(f"{self.preset} requires its verified matching acceleration LoRA")
        elif candidates:
            raise ValueError("An acceleration LoRA requires its matching Turbo preset")
        expected = 4 if self.preset == "turbo4" else 8 if self.preset == "turbo8" else None
        if expected and self.steps != expected:
            raise ValueError(f"{self.preset} requires exactly {expected} steps")
        if self.latent_refine.enabled:
            from .latent_upscale import resolve_low_resolution

            target_width = max(32, round(self.width / 32) * 32)
            target_height = max(32, round(self.height / 32) * 32)
            low_width, low_height = resolve_low_resolution(target_width, target_height, self.latent_refine.scale)
            if low_width >= target_width and low_height >= target_height:
                raise ValueError("latent refinement requires a genuinely smaller first-pass canvas")
        return self

    def resolved(self) -> dict[str, object]:
        from .latent_upscale import resolve_low_resolution

        width = max(32, round(self.width / 32) * 32)
        height = max(32, round(self.height / 32) * 32)
        frames = align_frames(self.frames)
        seed = self.seed if self.seed is not None else secrets.randbelow(2**53)
        active_acceleration = next(
            (acceleration_recipe(item.path) for item in self.loras if item.enabled and acceleration_recipe(item.path)),
            None,
        )
        latent_refine = self.latent_refine.model_dump()
        if self.latent_refine.enabled:
            low_width, low_height = resolve_low_resolution(width, height, self.latent_refine.scale)
            latent_refine.update({
                "low_width": low_width, "low_height": low_height,
                "target_width": width, "target_height": height,
            })
        return {
            **self.model_dump(),
            "requested": {"width": self.width, "height": self.height, "frames": self.frames},
            "width": width,
            "height": height,
            "frames": frames,
            "fps": NATIVE_FPS,
            "duration_seconds": frames / NATIVE_FPS,
            "sigma_grid_points": self.steps + 1,
            "seed": seed,
            "loras": [item.model_dump() for item in self.loras],
            "acceleration": active_acceleration,
            "latent_refine": latent_refine,
        }


class VideoSource(BaseModel):
    job_id: str | None = Field(default=None, min_length=1, max_length=128)
    upload_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def exactly_one_source(self) -> "VideoSource":
        if (self.job_id is None) == (self.upload_id is None):
            raise ValueError("Select exactly one source job or uploaded video")
        for value in (self.job_id, self.upload_id):
            if value and ("/" in value or "\\" in value or ".." in value):
                raise ValueError("Video source id must not contain path components")
        return self


class UpscaleRequest(BaseModel):
    source: VideoSource
    method: Literal[
        "lanczos", "realesrgan_x2plus", "seedvr2_3b",
        "dlss_super_resolution", "dlss5_neural_rendering",
    ] = "lanczos"
    scale: Literal[1, 2, 3, 4] = 2


class InterpolateRequest(BaseModel):
    source: VideoSource
    method: Literal["rife_v4_26"] = "rife_v4_26"
    factor: Literal[2, 3, 4] = 2


class EngineLoadRequest(BaseModel):
    model: str
    attention: Literal["sage", "sdpa"] = "sage"
    memory_profile: Literal["auto", "low_memory"] = "auto"
    loras: list[LoraSpec] = Field(default_factory=list, max_length=16)

    @field_validator("model")
    @classmethod
    def safe_repository(cls, value: str) -> str:
        return GenerationRequest.model_required(value)


class EngineLoraRequest(BaseModel):
    loras: list[LoraSpec] = Field(default_factory=list, max_length=16)
