# Astra standalone-engine review

Date: 2026-09-21. Replaces the ComfyUI adapter plan at the user's explicit direction. The application must run without a ComfyUI process, imports, installation or vendored ComfyUI engine. Existing model files may be reused only when their format is explicitly supported or independently converted and verified.

## Decision and feasibility

Use the official Diffusers **MiniMaxH3ModularPipeline**, behind the app's own single-GPU worker. This is a real native model implementation in Diffusers; there is no H3 `DiffusionPipeline` implementation. The automatic generic Hugging Face example showing `.images[0]` is unsuitable for this model.

The complete standalone checkpoint is not currently cached. Local `models--MiniMaxAI--MiniMax-H3` snapshot `42ed227ee7df40d41602854ae760620d6eb651fe` contains original FL2VA tokenizer/processor and text-encoder configuration only. `D:\comfyui-models\minimax-h3-src` holds original video-VAE source, not a complete runnable checkpoint. Existing ComfyUI INT8-convrot/NVFP4 files must not be passed to stock `from_pretrained` as if they were native Diffusers quantized weights.

A first production-capable route is the official Diffusers layout with **one** transformer partition and both large components quantized to TorchAO INT8 through supported loaders. This requires the missing original/native weights or a verified standalone conversion. A GUI that only detects missing weights has not passed the end-to-end generation acceptance gate.

## Exact upstream integration contracts

- Entry points: `from diffusers import ModularPipeline, ComponentsManager, MiniMaxH3Transformer3DModel, TorchAoConfig`; Qwen conditioner is `transformers.Qwen3VLForConditionalGeneration`.
- `ModularPipeline.from_pretrained(model_root, workflow="t2va")` prunes to text-video; `fl2va` adds first/last image; `ref2va` chooses the other transformer. Native repository layout has `transformer/`, `transformer_ref/`, `text_encoder/`, `vae/`, `audio_vae/`, `tokenizer/`, `processor/`, `scheduler/`, `audio_scheduler/`, and `modular_model_index.json`.
- Never call an unpruned pipeline's `load_components()` with no workflow on 128 GB RAM: it requests both approximately 61.7 GB transformer partitions. FL2VA/T2VA share one partition; Ref2VA requires a model swap.
- Generator call returns a mapping when `output=["videos", "audio", "sampling_rate"]`. Pass `prompt`, `num_frames`, `height`, `width`, `num_inference_steps`, and a seeded `torch.Generator`; for keyframes use `image` / `last_image` with in-memory images.
- Export is `encode_video(results["videos"][0], fps=24, output_path=..., audio=results["audio"][0], audio_sample_rate=results["sampling_rate"])`, from `diffusers.utils.export_utils`. Retain application MP4 metadata round-trip checks and authoritative JSON sidecars.
- There is no `negative_prompt` or `guidance_scale`: both partitions have guidance distilled into them. Video and audio use separate MiniMaxH3Scheduler instances (released shifts 12 and 3).
- **`num_inference_steps` counts sigma-grid points including terminal zero, so actual model evaluations are one fewer.** A UI labeling eight denoising evaluations must resolve the correct nine-point grid or a verified exact Turbo sigma schedule; store both counts. Do not copy ComfyUI step counts unchanged.
- Frame counts round upward to `17n+5`; 24 fps is fixed. Upstream validates the resulting duration in its trained window. Derive advertised maximum from actual validator; a requested 360 frames may snap beyond 15 seconds. Smoke checks cannot simply use five frames when upstream requires a roughly five-second minimum. Use a smaller spatial canvas while keeping valid duration.

## Supported INT8 memory recipe

Upstream explicitly documents `torchao.quantization.Int8WeightOnlyConfig(version=2)`, not an invented FP4/convrot import. Transformer loading uses Diffusers `TorchAoConfig`, excludes `proj_in`, `audio_proj_in`, `context_embedder`, `time_embedder`, `time_proj`, `token_refiner`, `norm_out`, `proj_out`, `audio_proj_out`. **Installed-version correction:** the web recipe says `low_cpu_mem_usage=False`, but installed Diffusers 0.40 `modeling_utils.py` rejects that setting when quantization is configured. Use the installed loader's supported `True` default and test it; do not copy the stale documentation flag. Conditioner loading uses **Transformers** `TorchAoConfig`, excluding `model.visual`, `model.language_model.embed_tokens`, `model.language_model.norm`, `lm_head`.

Inject loaded components using `pipe.update_components(transformer=..., text_encoder=...)` before loading remaining components. Call `requires_grad_(False)` on both large models. The documented low-VRAM route applies block-level group offload with one transformer block per group and leaf-level offload on `pipe.text_encoder.model`, with `use_stream=True`; version-2 quantized tensors are required for pinning. VAE/audio-VAE can reside on the 96 GB GPU. The official doc estimates around 75 GB host RAM for INT8. This is a memory recipe, not proof its current Windows wheels or PEFT combination work; exercise actual imports/load/inference.

For this 96 GB GPU, avoiding continuous layer streaming may be faster than the conservative consumer-GPU recipe. Measure resident INT8 components with headroom, or stage/offload the conditioner once. Do not combine two independent offload managers blindly. Full BF16 components together exceed the 128 GB host budget with overhead, even though each stage individually fits GPU memory.

## LoRA contracts and incompatibilities

Official `MiniMaxH3LoraLoaderMixin` handles Diffusers-format, original/Comfy-style names and raw DiffSynth fused-QKV de-interleaving. Use uniquely named `load_lora_weights(..., adapter_name=...)` adapters then `set_adapters(names, adapter_weights=...)`, and unload/reset them between incompatible models/jobs. Route reference adapters explicitly with `load_into_transformer_ref=True`; both partitions have identical module names, so shape acceptance alone does not establish semantic compatibility.

Upstream explicitly warns that **LoRAs trained on pruned checkpoints fail against the full transformer**. Many currently installed Comfy adapters may therefore require a compatible pruned standalone transformer or different upstream LoRA weights. Do not silently filter mismatched tensor keys. Metadata alpha is significant and must be honored. FP32 adapter factors may inflate memory: use the loader's supported dtype path and verify mixed quantized-weight/PEFT behavior before blanket `.to()` calls. Test two adapters together, then remove one and confirm no residual contribution.

## Worker, progress and cancellation

No `callback_on_step_end` exists in the inspected H3 denoising block. `MiniMaxH3DenoiseLoopWrapper` iterates timesteps and calls `loop_step`. Implement an owned subclass/replacement loop block with progress and cancellation checks, or a carefully scoped model forward hook verified against the actual active transformer. Do not send a conventional DiffusionPipeline callback argument and assume it works. Cancellation checks are also needed between component loading, prompt encoding, denoising and export; a dedicated worker process can be terminated as a final bounded cancellation mechanism without affecting unrelated Python processes. Cancelled workers must be reinitialized for later jobs.

Persist the request before GPU work. Record requested/resolved seed, dimensions, frame count, grid points, actual denoising evaluations, model revision/format, actual attention backend, ordered LoRA identities and weights. Model or acceleration changes invalidate cached pipeline state appropriately. Use one active GPU generation and durable queue/history; actual errors/OOM/cancellation cannot be reported as success.

## Acceleration and VC gate

Begin with the native PyTorch SDPA attention path plus a matched known Turbo adapter and correct step grid. Optional Sage/Flash backends require a genuinely available Windows sm_120 kernel and measured dispatch. The official doc's `_flash_3_hub` example is for Hopper and cannot be assumed correct on RTX PRO 6000 Blackwell.

VC remains blocked as a verified default: the installed independent-looking Comfy custom node uses a global-mean Sage host approximation rather than the paper's per-block mean restoration, and its grouping schedule does not count steps. Reimplementing its wrapper outside Comfy would not correct these defects. Public official kernel availability and numerical/performance validation are required before claiming paper VC support.

### 2026-09-22 real GPU adversarial follow-up

Root imported only the existing local `kernel.py` and `vsmooth.py` as a bounded read-only numerical probe, without importing or starting ComfyUI. With BF16 zero queries/keys, values of one and a final value of 100, uniform attention has an analytic mean. The 128-token aligned case matched exactly; the 129-token case padded to 256 returned 50.5 instead of 1.765625, with maximum absolute error 48.734375. The aligned 256-token case had maximum error 0.0078125. The results are recorded in [existing_vc_padding_probe.json](../data/qa/existing_vc_padding_probe.json). This establishes a real padding/masking correctness failure in the available candidate; finite output alone is not a correctness test.

Root's follow-up of the paper and [Nunchux technical article](https://www.nunchux.ai/blog/attention-is-the-video-bottleneck) did not establish an available validated official four-bit kernel suitable for this standalone installation. Keep VC disabled with that concrete diagnostic. A BF16 reference implementation or a wrapper around the faulty candidate cannot be presented as the requested verified fast VC engine. Repairing one padding defect alone would not close the separate algorithm-fidelity and real speed/quality gates.

## Acceptance gates before claiming completion

1. Start application/worker in its own environment with ComfyUI absent from import paths and no ComfyUI server. Static scan for Comfy imports, process launches, workflow APIs or hidden source loading.
2. Verify supported Diffusers/TorchAO/Transformers/PEFT imports and model-loading APIs against installed versions; pin tested versions or immutable source revisions.
3. Complete a real valid-duration small-canvas native H3 generation and preserve video/audio plus Japanese multiline metadata. Do not substitute a mock for this evidence.
4. Validate multiple compatible LoRAs, a supported model change, cancellation and restart, incompatible weight diagnostics, path containment, and disconnected browser recovery.
5. Benchmark a source-supported acceleration preset at equal seed/geometry and record cold/warm timing; call the speedup unmeasured until actual results exist.

## Primary source references

- [Diffusers H3 pipeline documentation and exact INT8 recipe](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/minimax_h3.md)
- [Native H3 modular pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/modular_pipelines/minimax_h3/modular_pipeline.py)
- [Native H3 denoising loop](https://github.com/huggingface/diffusers/blob/main/src/diffusers/modular_pipelines/minimax_h3/denoise.py)
- [H3 LoRA loader including pruning and partition caveats](https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_pipeline.py)
- [Official model files](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main)
- [VC-Attention paper](https://arxiv.org/html/2609.15810v1)
