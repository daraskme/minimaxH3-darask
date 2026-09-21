# Astra latent two-pass and SeedVR2 review

Reviewed 2026-09-22. These are separate additions to the already tested completed-video Real-ESRGAN/RIFE path. No full H3 two-pass or SeedVR2 inference has yet been verified in this review. Astra reviewed source and sent actionable findings to Sol; production edits remain with Sol.

## Latent two-pass contract

The [LBH-123-AI model repository](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) supplies a learned H3 latent upscaler and identifies an Apache-2.0 license. Pin the exact released 3D v1 source/config/checkpoint revision and hash; moving `main` and a 24-channel header alone are not enough to establish a supported architecture.

Installed Diffusers 0.40 source confirms that `MiniMaxH3AfterDenoiseStep` exposes normalized B×24×T×H×W video latents and stereo audio latents before decode. The decode stage applies the VAE mean/std, so duplicating that normalization before feeding a model trained on normalized H3 latents would be wrong. Explicit MiniMaxH3Scheduler sigmas are used verbatim and converted to time `1 - sigma`; terminal sigma zero is not evaluated. Thus a noised clean video latent, explicit low-sigma schedule, unchanged clean audio rows at time one, and one final decode is a technically coherent implementation contract. It is still an experimental recipe until full-model video quality and conditioning tests pass.

Check that target dimensions are legal, the first pass is genuinely lower resolution, time is unchanged, first/last keyframes are encoded at the second canvas, and original audio latents remain unchanged. Record both canvases, actual schedule, random seeds/noise draw ordering, checkpoint hash and model evaluations in output metadata. Failed or cancelled upscaler/refinement must restore the stock pipeline graph before another job. Unsupported attention-bearing or malformed latent checkpoints must be rejected during discovery/selection rather than advertised as ready and failing later at strict load.

## SeedVR2 adapter findings

The proposed adapter uses the official [ByteDance-Seed/SeedVR 3B inference entry](https://github.com/ByteDance-Seed/SeedVR/blob/main/projects/inference_seedvr2_3b.py) in an isolated runtime. Its reference dependencies include Linux/NCCL, FlashAttention and Apex; the present Windows capability correctly stays unavailable pending a real tested port. An isolated Linux adapter is not evidence that this user's native Windows machine can already run it.

Initial actionable findings sent to Sol:

1. The upstream entry does not consume `SEEDVR_CHECKPOINT_DIR`. It reads prompt embeddings relative to its working directory, so the proposed environment variable cannot relocate the advertised model bundle. Use a verified config/wrapper or staged layout and test actual resolution.
2. A blocking `stdout.readline()` prevents cancellation while inference is silent. Drain logs on a separate bounded reader and poll cancellation independently. Terminating the `torchrun` parent alone can leave its GPU child alive; own and reap the full process group.
3. The upstream geometry transform is area-based with divisibility cropping. Requested width/height do not automatically prove exact output geometry. Validate decoded geometry, frame count and rational FPS/duration, including 24000/1001, before publication.
4. Check cancellation around remux as well as inference. Preserve authoritative metadata lineage and source audio, bound staging/logs, and use the existing publication gate. Imports and checkpoint file presence must not be represented as a successful model/export test.

These findings concern the unfinished implementation. Final corrected-source review and a real runtime smoke test remain open. No fake restoration, pixel resize fallback labeled SeedVR2, or ComfyUI dependency is approved.

## Native Windows feasibility correction: maintained standalone CLI

Deeper investigation found an existing viable implementation route, so the older official Linux dependency list must not be treated as a permanent Windows blocker. [numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) explicitly provides an independent `inference_cli.py`. The source-only checkout at `.cache/research/seedvr2-standalone` is pinned to `4490bd1f482e026674543386bb2a4d176da245b9`, Apache-2.0. Its CLI imports model/core modules without importing Comfy interfaces. Optional Comfy path/interrupt detection is guarded; the runtime works without Comfy installed. A standalone launch can additionally exclude those optional integrations.

Concrete inspected source evidence: 3B/7B attention already includes variable-length PyTorch SDPA, selected by default; native CustomRMSNorm/LayerNorm replace Apex; `init_torch()` containing NCCL is not called on the CLI route; absent sequence-parallel groups resolve to size one. There is therefore no need to invent an unverified attention or normalization replacement just to attempt this GPU. CLI `--help` was actually attempted on Windows with no model loading and stopped at missing `cv2`. A module-location probe found only these additional requirements absent from the validated H3 environment: OpenCV, OmegaConf, GGUF, rotary_embedding_torch and matplotlib. Supply them in an isolated environment and continue with a small real model test.

Use explicit `--model_dir`, `--dit_model seedvr2_ema_3b_fp16.safetensors`, `--attention_mode sdpa`, `--resolution SHORT_EDGE`, `--max_resolution LONG_EDGE`, and `--batch_size 5` initially, without compile or block swapping. The inspected registry pins the 3B FP16 SHA-256 to `2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304`, and shared `ema_vae_fp16.safetensors` to `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`, from `numz/SeedVR2_comfyUI`. Explicit FP16 avoids the CLI's current FP8 default for first correctness. Its config and safe-loaded prompt embeddings are included with the pinned source.

Prefer `--output_format png` and assemble the resulting frames using H3 Studio's exact rational FPS/audio/metadata exporter. Verify frame count and geometry before publication. Fix the CLI's colon-delimited PYTHONPATH for Windows or use an equivalent explicit standalone launcher. Its loader uses `strict=False` and its download-validation cache trusts filename/size/mtime; the outer adapter should require the pinned checkpoint hashes, inspect missing/unexpected keys and fail on unresolved meta tensors. Preserve process isolation/cancellation and bound staging. Native model execution is now a concrete next step, not yet a passed test.
