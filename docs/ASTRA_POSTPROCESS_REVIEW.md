# Astra standalone post-processing plan review

Reviewed 2026-09-22. Scope: completed-video upscaling and frame interpolation, independently runnable on Windows with 128 GB RAM / RTX PRO 6000 Blackwell 96 GB. This is a source and architecture review, not a claim that either model has completed the runtime acceptance tests.

## Decision

Implement a separate post-processing job using a verified Real-ESRGAN x2 checkpoint (root-agent research) and the official Practical-RIFE 4.26 model below. Make each stage independently selectable. Keep generation settings separate from post-processing settings and record their lineage. A completed low-resolution clip remains available if a later stage fails.

SeedVR2 is a real, independently published restoration model, but its current official reference implementation is not a ready-to-run native Windows backend. Do not expose an enabled SeedVR2 option merely because weights exist. Latent upscaling is a different operation within a generation pipeline; it must not be presented as the same completed-video stage.

## RIFE 4.26: verified artifact and minimal interface

The maintainer publishes 4.26 and states that the linked models share the repository's MIT license. The README presently recommends 4.25 generally and describes 4.24+ as suitable for generated-video post-processing; therefore 4.26 is an available user-selected version, not a demonstrated universal quality winner. [Maintainer README](https://github.com/hzwer/Practical-RIFE)

Downloaded directly from the maintainer's linked [RIFEv4.26_0921.zip](https://drive.google.com/file/d/1gViYvvQrtETBgU1w8axZSsr7YUuw31uy/view):

- Research copy: `.cache/research/RIFEv4.26_0921.zip`
- Archive length: 22,867,954 bytes.
- SHA-256: `c2452dd2b244947d4be580156bbead60d6b72af5736860f7d6b3f99648c9c4cc`.
- Weight: `train_log/flownet.pkl`, 24,636,301 bytes.
- All three included Python source files were inspected before any execution. macOS metadata and cached bytecode are unnecessary.

The archive's `RIFE_HDv3.py` wrapper still declares `version = 4.25`; retain 4.26 identity through archive provenance and hash, rather than trusting that stale field. It also initializes training losses/optimizer and uses non-strict checkpoint loading. The inference network itself is in `IFNet_HDv3.py` and only needs Torch plus the repository's small warp helper. A direct inference adapter can omit those training dependencies.

Contract established from the inspected archive: instantiate `IFNet`; load the checkpoint on CPU with `weights_only=True`; remove a leading `module.` prefix from state-dict keys; discard only the training-only `teacher.` and `caltime.` keys; load the remainder strictly. Pass concatenated RGB BCHW inputs in [0,1], timestep 0.5, and `scale_list=[16/scale, 8/scale, 4/scale, 2/scale, 1/scale]`. Take the last merged result from the returned tuple. The default four-entry scale list in `IFNet.forward` is insufficient for its five blocks. Disable ensemble. The [warp helper](https://github.com/hzwer/Practical-RIFE/blob/main/model/warplayer.py) uses native Torch grid sampling. Retain the [MIT copyright/license](https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE) and document local adaptations.

The root agent's first strict load correctly rejected training-only extras. The archive source explicitly comments out `teacher` and `caltime` under its unused-training block. Astra independently loaded only the 24 MB checkpoint on CPU with `weights_only=True`: 198 total keys, exactly 40 training-only keys (30 teacher / 10 caltime), leaving 158 inference keys. An explicit allowlist followed by strict loading preserves meaningful missing/unexpected-key validation; unrestricted `strict=False` is not an acceptable substitute.

Do not launch the upstream video CLI unchanged. It contains shell-string ffmpeg calls, a shared `temp` directory it removes, swallowed decoder errors, and 500-frame queues. Its explicit `--fps` path skips audio transfer; the normal frame sequence is generally 2N−1 rather than duration-preserving 2N. Its preprocessing pads to a multiple of `max(128, 128/scale)`. These are concrete integration concerns found in [upstream inference_video.py](https://github.com/hzwer/Practical-RIFE/blob/main/inference_video.py).

## SeedVR2: exact source and limits

[IceClear/SeedVR2](https://github.com/IceClear/SeedVR2) links to the actual [ByteDance-Seed/SeedVR implementation](https://github.com/ByteDance-Seed/SeedVR). The reference environment uses Python 3.10, old Torch/Diffusers/Transformers packages, FlashAttention 2.5.9 and Apex with provided Linux wheels. The documented resource point is 100 frames at 720×1280 on one H100 80 GB, with larger resolutions described using four GPUs. This does not establish single-GPU 1080p throughput on the user's machine. The authors also acknowledge possible oversharpening of lightly degraded generated inputs.

The [pinned requirements](https://github.com/ByteDance-Seed/SeedVR/blob/main/requirements.txt) specify Torch 2.3.0, Diffusers 0.29.1, Transformers 4.38.2 and Torchvision 0.18.0; installing these into H3 Studio's current environment would conflict with the validated generation runtime. The [distributed initializer](https://github.com/ByteDance-Seed/SeedVR/blob/main/common/distributed/basic.py) unconditionally initializes NCCL even at world size one. The [attention source](https://github.com/ByteDance-Seed/SeedVR/blob/main/models/dit_v2/attention.py) imports FlashAttention varlen directly. A Windows port requires deliberately replacing these paths and testing equivalence; an ordinary subprocess does not resolve them.

The exact initial candidate is [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B/tree/main), Apache-2.0, with `seedvr2_ema_3b.pth` (13.6 GB), `ema_vae.pth` (1 GB), `pos_emb.pt`, and `neg_emb.pt`. The repository also contains Linux Apex wheels; these are not Windows assets. Do not assume community FP8 weights use the same loader format.

The [3B reference inference](https://github.com/ByteDance-Seed/SeedVR/blob/main/projects/inference_seedvr2_3b.py) exposes `configure_runner(sp_size)` and `generation_loop(...)`, or a torchrun CLI with input/output folders, seed, resolution and sequence-parallel size. It uses one sampling step and CFG 1.0 by default, VAE/DiT stage offload, cached text embeddings, 4k+1 temporal padding followed by trimming, and a resize based on target area rather than exact target dimensions. The script reads the whole video and writes output without carrying its source audio. An adapter must preserve exact geometry/timing/audio itself. Do not treat `--out_fps` as interpolation.

## Application contracts and adversarial acceptance gates

These are engineering requirements proposed by Astra, independent of upstream implementations:

1. **Separate immutable lineage.** Post-process only an existing completed job/output or a validated local upload token. Resolve paths server-side beneath trusted roots. Store parent job id, source hash, source metadata, operation order, checkpoint hashes, dimensions, exact rational FPS, frame counts, seed where relevant, and versions. Never overwrite source clips.
2. **Shared GPU ownership.** Use the same single-worker queue for generation and post-processing. Release H3 GPU components before loading the upscaler/interpolator. Keep loaded post-processing models bounded to the active stage. Cancellation checks belong between frames/tiles, with temporary outputs separated from published artifacts.
3. **Exact 2× timing.** For N constant-rate source frames, write each original and its midpoint, then hold the final original for the last half interval. This yields 2N frames at exactly twice the rational input FPS and preserves duration. Preserve AAC audio when compatible. Test 24→48 and 24000/1001→48000/1001 separately; reject or explicitly normalize VFR inputs. Do not implement slow motion accidentally by changing only FPS.
4. **Bounded streaming.** Decode and encode incrementally, with small queues. Do not buffer an entire 4K clip. Upscale at source FPS first and interpolate afterward. Spatial tile padding must prevent seams; batch size one and a conservative tile size are a sensible starting point, followed by measured optimization on this GPU.
5. **Scene transitions.** Detect hard cuts and hold the source interval instead of hallucinating a midpoint across unrelated scenes. Record thresholds/mode. A threshold needs validation on actual scene cuts and high-motion continuous shots.
6. **No silent substitute.** Missing weights, failed strict loads, unavailable GPU kernels or unsupported input formats must report a concrete unavailable/failure reason. Do not silently replace neural upscaling with bicubic or RIFE with frame duplication while retaining the neural method's label.
7. **Publication and recovery.** Encode into a job-specific temporary path; validate dimensions/frame count/FPS/duration and audio; write sidecar and embedded provenance; then publish atomically. Metadata failure must retain the processed video and a recoverable record, as generation already does.
8. **Runtime evidence required.** Run the official RIFE checkpoint on a small real GPU pair and require finite, correctly shaped output plus a nontrivial midpoint. Then run a short moving video with audio through upscale-only, interpolation-only and combined jobs. Check output geometry, exact timing, source preservation, cancellation, metadata, and restart history. Unit tests alone do not demonstrate neural execution.

## Remaining limits

No SeedVR2 model was downloaded or executed during this bounded review. The root agent independently ran the official RIFE inference weights after the narrow training-key filter: a 128×128 FP32 midpoint passed on the GPU, followed by timesteps 0.25, 1/3, 0.5, 2/3 and 0.75. A padded 1152×1920 FP32 forward was finite with approximately 34 ms synthetic forward time and 2.11 GB maximum reserved memory. These measurements do not include video I/O or audio/metadata work. The extracted checkpoint SHA-256 is `45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f`.

FP16 initially failed because upstream warp grids are created as float32 and the cache does not distinguish dtype. Do not claim FP16 support until dtype-aware grid creation/cache handling passes the actual GPU test; FP32 is the verified initial path and is a reasonable default.

The root agent also reports a real Real-ESRGAN x2plus checkpoint forward through Spandrel 0.4.2 in FP16: 544×960 to 1088×1920, finite output. Checkpoint SHA-256: `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`. Its reported warm synthetic forward was approximately 55 ms, which is not an end-to-end video throughput measurement.

The initial limitations and pending gates in this section are historical. Completed timing/audio, application integration and provenance evidence is recorded in the final sections below. Generalized full-length/4K quality and performance are not inferred from the bounded measurements.

## Application source audit in progress

The first backend pass identified the following concrete issues, communicated to Sol while implementation continues:

- Exported post-processing metadata needs the original prompt/settings and durable previous-stage lineage, not only a database job id and source hash.
- A header average frame rate does not establish CFR or perform temporal normalization; the reported timing mode must match actual processing.
- Blocking stdout reads and an undrained stderr pipe can prevent ffmpeg cancellation or deadlock. Process handles must be reaped before Windows temporary-file removal.
- Source probing errors should return a clear validation response. Output geometry must be bounded before queueing, rather than multiplying a 16k source into a 64k encoder request.
- The MP4-only upload boundary currently checks filename/MIME but lets PyAV auto-detect arbitrary contents. Astra independently generated a harmless 32×32 one-second Matroska clip named `.mp4`; `inspect_video` accepted it. Force the appropriate MP4/MOV parser before opening contents, including the later decoding path, and reject renamed formats/playlists.

These findings are not a final verdict on the unfinished implementation. Record fixes and actual post-processing evidence before approving the completed feature.

## Fix verification and actual post-processing evidence

Astra independently ran the maintained suite after the post-processing integration: **15 passed**. Additional bounded runtime checks passed:

- A 12-frame 24 FPS CFR clip is identified as CFR; a 12-frame clip with irregular presentation timestamps is identified as VFR.
- The earlier renamed-Matroska fixture is now rejected by the forced MOV demuxer.
- Embedded H3 JSON containing Japanese prompt text, seed zero, LoRA settings and inert command-like strings round-trips exactly through the metadata reader. This test validates data-only reading, not yet the complete upload-to-output provenance chain.
- A child process emitting no stdout was cancelled in 0.234 seconds; a child emitting 200 KB to stderr completed without a blocked pipe.

Source inspection confirms separate immutable origin snapshots, bounded intermediate lineage, async upload probing/hashing in a thread pool, O(1) timestamp validation, AI rejection of VFR, actual fps-filter normalization for Lanczos, validation responses and geometry limits. The RIFE CUDA warp cache is cleared on exit, its cut threshold is recorded, and output frame rates are limited consistently to 240 FPS. The model installer verifies pinned SHA-256 hashes and extracts only the known weight member; it does not run downloaded code.

The root agent independently exercised the actual local API/GPU pipeline: a 24-frame 192×128 source became a 384×256 Real-ESRGAN output, then separate RIFE 2×/3×/4× jobs produced 48/72/96 frames at 48/72/96 FPS. All outputs retained a one-second duration and an AAC stream. Original-grid mean absolute pixel errors after lossy encoding were approximately 1.774/1.914/1.954. This is real application-level neural execution, not a mock. Example artifacts are `outputs/2026-09-22/h3-upscale-f0061533.mp4` and RIFE jobs `ba0e7f7a`, `de04c037`, `2d69c829`.

The root test initially identified an empty post-processing top-level `prompt` and MP4 `description`, despite preservation inside nested origin metadata. This issue was corrected and its complete runtime verification is recorded below; the earlier reader-only test was not treated as sufficient evidence.

## Final post-processing review decision

The prompt-propagation issue above is now fixed and independently verified. Astra used a real temporary uploaded 24-frame MP4 with AAC and a 4,000-character Japanese/emoji/newline prompt containing hash, semicolon, equals and backslash characters. The upload-source resolver, separate upscale API handler, actual JobRunner, Lanczos processing and metadata finalization all ran. The final MP4's top-level JSON prompt and description matched the original exactly; the original embedded metadata, seed zero, LoRA settings and inert command-like text remained unchanged in origin provenance. The sidecar equaled the embedded JSON, and 24 frames plus one audio stream remained. This was a complete CPU processing path through the same application metadata boundary used by the AI stages.

The final maintained regression suite was independently rerun: **17 passed**. The later generic bounded JSON-comment reader preserves imported metadata as inert data even when it has no H3 schema; it does not interpret it as model paths or commands. Model readiness now verifies pinned weight hashes, and the bootstrap installer uses official data URLs and fixed hashes.

All material findings raised in this post-processing source review are resolved. The implemented Real-ESRGAN x2 and RIFE 2×/3×/4× paths have actual application GPU evidence from the root agent; the separate-stage/source-preservation/audio/metadata contracts have the runtime evidence documented above. **Astra approves this bounded post-processing implementation.** This approval does not imply that SeedVR2, latent two-pass upscaling, DLSS 5 or VC Attention are implemented/validated, nor does it approve the still-separate full H3 generation gate.

The root agent closed the fractional-FPS gate through the actual upload/API/GPU path: a 24-frame 192×128 source at exactly 24000/1001 FPS became 48 frames at exactly 48000/1001 FPS, with duration exactly 1001/1000 seconds and one audio stream. Frame rate and duration were compared as exact fractions.

The root agent also repeated the complete AI chain with an uploaded plain bounded JSON metadata record containing a prompt and seed 42: Real-ESRGAN x2 followed by RIFE 2× retained the entire original `embedded_metadata` dictionary exactly in origin provenance. Top-level JSON prompt and MP4 description matched exactly. SHA-256 of the AAC packet bytes was identical for the original, upscaled and interpolated clips. The resulting job prefixes were `aad90822` (upscale) and `85d4cabb` (interpolate).

The final runner source adds a publication validation gate before metadata finalization/completed status: decoded output dimensions, FPS, duration, audio presence and expected frame count are checked, and the probe is recorded in engine metadata. After restarting the application, the root agent confirmed that a real RIFE job completed with its `engine.output_validation` record present.

The root agent then cancelled actual Real-ESRGAN GPU processing during phase `Real-ESRGAN x2 · 1f`. The job reached `cancelled` with no published `output_path`. A subsequent RIFE replay completed successfully at `outputs/2026-09-22/h3-interpolate-30f653b3.mp4`, confirming that cancellation leaves the next GPU job usable. The final publication/cancellation/recovery gates are closed; no post-processing backend verification remains open in this review.

The post-processing fractional timing and AI provenance/audio gates are now closed. The outstanding major execution gate is **full native H3 generation**; unsupported optional engines remain explicitly unavailable rather than being represented as verified features. Performance figures above remain narrowly labeled synthetic, and no unmeasured full-length/4K throughput claim is approved.
