from __future__ import annotations

"""MiniMax H3 low-sigma second-pass blocks for Diffusers 0.40.

The public ModularPipeline API can expose H3's denoised latents by omitting the
decode block.  The refinement graph below reuses the official H3 layout and
denoiser blocks with an explicit low-sigma schedule, while leaving the clean
audio latents untouched.  This is generation-time latent refinement, not a
pixel-space post-processing upscaler.
"""

from copy import deepcopy
from typing import Any


def add_refine_noise(latents, strength: float, generator):
    import torch

    strength = float(strength)
    if not 0.0 < strength <= 0.6:
        raise ValueError("refine strength must be in (0, 0.6]")
    noise = torch.randn(latents.shape, generator=generator, device="cpu", dtype=torch.float32)
    clean = latents.to(device="cpu", dtype=torch.float32)
    return clean * (1.0 - strength) + noise * strength


def first_pass_blocks(blocks):
    result = deepcopy(blocks)
    if "decode" not in result.sub_blocks:
        raise RuntimeError("Diffusers H3 block graph has no decode stage")
    result.sub_blocks.pop("decode")
    return result


def refine_blocks(blocks):
    """Replace the stock full-sigma denoiser with a low-sigma video-only pass."""
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3FL2VAPrepareLatentsStep,
        MiniMaxH3NoKeyframeAnchorsStep,
        MiniMaxH3PrepareConditionLatentsStep,
        MiniMaxH3PrepareLatentsStep,
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
    )
    from diffusers.modular_pipelines.minimax_h3.denoise import (
        MiniMaxH3DenoiseLoopWrapper,
        MiniMaxH3LoopDenoiser,
        MiniMaxH3LoopSchedulerStep,
    )
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3AutoDenoiseStep
    from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks
    from diffusers.modular_pipelines.modular_pipeline_utils import InputParam

    class RefineTimesteps(MiniMaxH3SetTimestepsStep):
        @property
        def inputs(self):
            return super().inputs + [
                InputParam(
                    name="refine_strength", type_hint=float, required=True,
                    description="Highest video sigma of the second pass.",
                )
            ]

        @torch.no_grad()
        def __call__(self, components, state):
            block_state = self.get_block_state(state)
            device = components._execution_device
            points = int(block_state.num_inference_steps)
            if points < 2:
                raise ValueError("latent refinement needs at least one denoiser evaluation")
            strength = float(block_state.refine_strength)
            if not 0.0 < strength <= 0.6:
                raise ValueError("refine strength must be in (0, 0.6]")
            sigmas = torch.linspace(strength, 0.0, points, dtype=torch.float32)
            components.scheduler.set_timesteps(sigmas=sigmas, device=device)
            block_state.timesteps = components.scheduler.timesteps
            # The second pass conditions on the already generated soundtrack.
            # Give audio rows clean t=1 and do not step them in RefineUpdate.
            block_state.audio_timesteps = torch.ones_like(block_state.timesteps)
            block_state.row_timestep_plan = [
                tuple(
                    tensor.to(device) for tensor in self.build_row_timesteps(
                        block_state.video_indices,
                        block_state.audio_indices,
                        block_state.num_condition_video_rows,
                        block_state.num_condition_audio_rows,
                        block_state.text_indices.numel(),
                        float(timestep), 1.0,
                        max(float(timestep), components.keyframe_noise_aug), 1.0,
                    )
                )
                for timestep in block_state.timesteps
            ]
            self.set_block_state(state, block_state)
            return components, state

    class RefineUpdate(MiniMaxH3LoopSchedulerStep):
        @torch.no_grad()
        def __call__(self, components, block_state, i, t):
            start = block_state.num_condition_video_rows
            block_state.latents[start:] = components.scheduler.step(
                block_state.noise_pred[0, start:].float(), t, block_state.latents[start:], return_dict=False
            )[0]
            return components, block_state

    class RefineLoop(MiniMaxH3DenoiseLoopWrapper):
        block_classes = [MiniMaxH3LoopDenoiser, RefineUpdate]
        block_names = ["denoiser", "update"]

    class RefineT2VA(SequentialPipelineBlocks):
        model_name = "minimax-h3"
        block_classes = [
            MiniMaxH3NoKeyframeAnchorsStep,
            MiniMaxH3PrepareLayoutStep,
            MiniMaxH3PrepareLatentsStep,
            RefineTimesteps,
            RefineLoop,
            MiniMaxH3AfterDenoiseStep,
        ]
        block_names = ["no_keyframe_anchors", "prepare_layout", "prepare_latents", "set_timesteps", "denoise", "after_denoise"]

    class RefineFL2VA(SequentialPipelineBlocks):
        model_name = "minimax-h3"
        block_classes = [
            MiniMaxH3PrepareLayoutStep,
            MiniMaxH3PrepareConditionLatentsStep,
            MiniMaxH3PrepareLatentsStep,
            MiniMaxH3FL2VAPrepareLatentsStep,
            RefineTimesteps,
            RefineLoop,
            MiniMaxH3AfterDenoiseStep,
        ]
        block_names = [
            "prepare_layout", "prepare_condition_latents", "prepare_latents", "prepare_latents_fl2va",
            "set_timesteps", "denoise", "after_denoise",
        ]

    class RefineAuto(MiniMaxH3AutoDenoiseStep):
        block_classes = [RefineFL2VA, RefineT2VA]
        block_names = ["fl2va", "t2va"]
        block_trigger_inputs = ["image", "last_image"]
        default_block_name = "t2va"

        def select_block(self, **kwargs):
            if kwargs.get("references") is not None:
                raise ValueError("latent two-pass does not support ref2va references")
            if kwargs.get("image") is not None or kwargs.get("last_image") is not None:
                return "fl2va"
            return None

    result = deepcopy(blocks)
    if "denoise" not in result.sub_blocks or "decode" not in result.sub_blocks:
        raise RuntimeError("Diffusers H3 block graph is incompatible with latent refinement")
    result.sub_blocks["denoise"] = RefineAuto()
    return result


def validate_refine_result(first: dict[str, Any], second: dict[str, Any], target_height: int, target_width: int) -> None:
    import torch

    latents = first.get("latents")
    audio = first.get("audio_latents")
    if latents is None or audio is None:
        raise RuntimeError("first H3 pass did not return video and audio latents")
    videos = second.get("videos")
    rendered_audio = second.get("audio")
    if not videos or rendered_audio is None:
        raise RuntimeError("refine pass did not decode video and audio")
    refined_audio = second.get("audio_latents")
    if refined_audio is None or not torch.equal(audio.detach().cpu(), refined_audio.detach().cpu()):
        raise RuntimeError("latent refinement changed the clean audio latents")
    frame = videos[0][0]
    if getattr(frame, "size", None) != (target_width, target_height):
        raise RuntimeError(f"refine decode returned {getattr(frame, 'size', None)}, expected {(target_width, target_height)}")
