# Astra architecture and adversarial plan review

> SUPERSEDED: The user explicitly required an independent generation engine after this review. The ComfyUI adapter decision and implementation contract below are rejected and retained only as historical review context. The current decision is in `ASTRA_STANDALONE_REVIEW.md`. The VC source-audit findings remain applicable.

Review date: 2026-09-21. Reviewer: Astra. This is a source-level plan review; GPU generation has not yet been verified by this reviewer.

## Decision

Approve a local FastAPI application with a focused Japanese-friendly browser UI and an adapter to the existing **C:\ComfyUI (0.34.0)**. Reuse its Python environment, installed kernels and **D:\comfyui-models**. Do not introduce another Diffusers GPU stack: native ComfyUI H3 nodes and model files already exist, and their actual interfaces were inspected locally.

Use a dedicated backend instance/port and project-owned launch configuration. Bind both services to loopback. Keep existing user workflows and model files intact. Save completed videos and their JSON records under this repository's `outputs` directory. No model downloads are necessary for the first working version.

## Concrete implementation contract

1. Startup diagnoses ComfyUI reachability, version, GPU memory, `/object_info`, available model filenames and required nodes. A disconnected backend is an actionable state, never fake generation. Local model selection is validated against the backend inventory. Do not infer arbitrary architectures are interchangeable merely because a file ends in `.safetensors`.
2. Compile an API-format graph from fixed, tested H3 node contracts. Do not submit frontend workflow JSON as `/prompt` input. Required chain: `UNETLoader` → ordered `LoraLoaderModelOnly` nodes → optional tested model patch → `BasicGuider`; `CLIPLoader(type=minimax)` and video `VAELoader` feed `MiniMaxH3ImageToVideo`; `RandomNoise`, `KSamplerSelect`, `BasicScheduler` and the H3 latent feed `SamplerCustomAdvanced`; split the resulting nested latent using `VAEDecode` and `VAEDecodeAudio`, then `CreateVideo(fps=24)` and `SaveVideo`. Ref2VA uses the distinct native reference node and correct model variant.
3. **Inspect runtime SaveVideo schema.** Local 0.34.0 uses nested DynamicCombo, e.g. `format={"format":"mp4","codec":{"codec":"h264"}}`; older examples with plain string codec may fail. Native SaveVideo already embeds prompt metadata, but the application must verify its own final MP4 metadata and preserve audio.
4. H3 dimensions must be multiples of 32; normal native canvas is 1344×768. Frame counts round upward to `17k+5`; 124 frames at 24 fps is approximately 5.167 seconds. Display and save resolved dimensions/frame count/seconds alongside requested values. Avoid exposing an arbitrary FPS control that merely changes playback speed.
5. Persist job settings before submitting, associate them with the returned prompt UUID, and reconcile queue/history after reconnect. One application job at a time is appropriate for this 96 GB GPU. Preserve backend error details and differentiate queued, running, cancellation requested, cancelled, failed, and completed.
6. Local backend provides atomic `POST /api/jobs/{job_id}/cancel`. Prefer it to `/interrupt`; the latter can interrupt a different job if global, and its older targeted implementation has a check/use race. Never stop unrelated Python processes. A cancellation acknowledgement is not proof a job was cancelled; reconcile final state.
7. Resolve returned output paths beneath the configured backend output root; reject traversal. Use generated unique names in project `outputs`. Save an authoritative JSON sidecar containing prompt, resolved seed, full ordered LoRA list/weights, model filenames, sampler/schedule/steps, attention actually requested/applied, requested/resolved geometry, backend version, timestamps and workflow. Embed UTF-8 JSON in MP4 comment/description metadata using ffmpeg stream copy; preserve audio and check round trip with ffprobe. Video containers use metadata tags rather than JPEG EXIF. Never interpolate prompts or paths into a shell string.
8. Multiple LoRAs are ordered, independently weighted and enabled/disabled. Validate compatible H3 adapters; surface incompatible adapters rather than silently applying zero matching keys. A fused Turbo checkpoint must not automatically receive an extra Turbo/Acc LoRA. Model selection and acceleration presets must resolve together; fast steps are only valid for their paired distilled model/adapter.

## Hardware and initial acceleration profile

Existing assets include FL2VA/Ref2VA pruned INT8 convrot models (~21 GB), non-pruned FL2VA (~34 GB), FP8-scaled FL2VA (~21 GB), a fused Turbo checkpoint, H3 Qwen INT8/NVFP4-AWQ encoders, video/audio VAEs, and multiple 4/8-step Turbo adapters. Existing setup documentation reports Torch 2.14/cu130, SageAttention 2.2 sm_120 and Comfy Kitchen kernels; **runtime validation must confirm those claims**.

Prefer automatic memory management with headroom over forcing all components to remain resident. 96 GB VRAM still needs activation/attention/decode workspace, and 128 GB RAM must retain OS margin. Use a known installed INT8 model and matching 8-step adapter as the first quality/speed preset; a verified 4-step adapter is a draft preset. Dense SageAttention can be selected only if its actual runtime kernel is available. Tiled VAE is a memory option, not a guaranteed speedup. Avoid stacking caches, sparse attention and quantized attention in the default profile without isolated A/B evidence. Do not promise paper speedups on this workload.

## VC-Attention adversarial review: block trusted enablement

The local `C:\ComfyUI\custom_nodes\ComfyUI-VC-Attention` exists, but is **not the official published paper kernel**. Its README itself says the official CuTe kernels were unpublished as of September 19. The paper uses workstation 4-bit V-Smooth without ExpCast-FP8; the latter is its datacenter 8-bit path.

Concrete local discrepancies:

- `vc_attention/forward.py:sage_host_attention` subtracts a single global value mean and adds it back after SageAttention. The paper's per-block mean subtraction and probability-weighted block-mean restoration are absent from this 4-bit host path. The function docstring acknowledges this. A UI must not claim this faithfully applies paper VC-Attention.
- `VCState.should_group` selects grouping using normalized sigma rather than actual denoising-step index. With shifted/nonlinear schedules this does not implement the stated first quarter of steps.
- The alternate tiled path pads Q/K/V before calling `vc_fwd`, with no original token count passed to that function. Audit masking before any enablement: padded key tokens must never enter the softmax normalization. The exposed modes also should not be assumed to implement native FP8/FP4 matmuls merely from their names.
- Broad exception fallback may complete a job using ordinary attention while the UI reports VC. Node presence or successful generation is insufficient evidence of the claimed kernel dispatch.

Recommendation: report VC as detected but blocked/unverified, with a short explanation. Do not auto-enable it. Repairing and validating this separate custom kernel would require math tests (including non-multiple token counts), dtype/layout/scale checks, dispatch evidence and same-seed GPU quality/performance A/B tests. Existing verified speed presets satisfy acceleration without overstating this path.

## P0/P1 acceptance checks

- P0: compile and submit a real small H3 generation using installed weights; retrieve an MP4 with video and native audio. A mock test alone cannot establish GPU success.
- P0: prove final output video metadata recovers a Japanese multiline prompt and exact seed/ordered LoRA settings; verify audio survives and paths cannot escape output roots.
- P0: verify backend failure, OOM/error output, disconnect/reconnect and cancellation cannot be represented as completed jobs. Cancel only the owned prompt UUID.
- P0: LoRA chain ordering, independent weights, disabled adapters, fused-Turbo duplication prevention and stale/unavailable model selection.
- P1: UI browser check at desktop and narrower widths, real model inventory, launch diagnostics, queued generation/progress, playable completed result, settings restoration, outputs-folder access.
- P1: generation with two compatible installed LoRAs, then a different supported model variant. Record changes and backend validation result; do not claim quality/performance from mere loading.
- P1: benchmark baseline versus the chosen accelerated preset with equal resolution, frames, seed and prompt, recording warm/cold timing and actual resolved settings. Clearly separate smoke generation from a production-size benchmark.

## Primary sources inspected

- [ComfyUI native H3 documentation](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [Official H3 native workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json)
- [Official H3 node source](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
- [Official ComfyUI API example](https://github.com/Comfy-Org/ComfyUI/blob/master/script_examples/websockets_api_example.py)
- [Official ComfyUI server source](https://github.com/Comfy-Org/ComfyUI/blob/master/server.py)
- [Official packaged models](https://huggingface.co/Comfy-Org/MiniMax-H3)
- [Official ComfyUI H3 engineering announcement](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui)
- [VC-Attention paper](https://arxiv.org/html/2609.15810v1)

Local source inspection takes precedence over mutable web examples for this installed backend. This document intentionally does not treat unverified third-party README benchmark claims as measured results.
