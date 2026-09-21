# Latent 2-pass generation

H3 Studio's optional Latent 2-pass path stays inside the MiniMax H3 generation job. It is distinct from the completed-video upscale modes.

1. Run the normal H3 denoiser on a lower 32-pixel-aligned canvas, omitting the decode block.
2. Pass Diffusers' normalized `B×24×T×H×W` video latent through the learned 3D spatial upscaler. The temporal length is unchanged.
3. Add deterministic noise at the selected refine sigma, using a fresh CPU generator seeded with the resolved generation seed.
4. Run a second H3 schedule from that sigma to zero. Only generated video rows are updated; clean audio latents stay at time 1 and are not stepped.
5. Decode video and the original generated audio once, then use the normal MP4 and metadata publication path.

The adapter is a clean-room standalone port of the Apache-2.0 `LatentResizer3D` network from `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`. It imports no ComfyUI code. The accepted release is pinned to the Hugging Face model repository revision `3f941d5` and exact v1 architecture (`24 → 512 → 24`, 12 input blocks, 12 output blocks, temporal convolution every two blocks with kernel 5, no attention). Only these released SafeTensors hashes are accepted:

- BF16: `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6`
- FP16: `043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2`

Put one checkpoint beneath `models/latent_upscalers`. Inventory discovery verifies its SafeTensors structure and SHA before presenting it as ready. Execution repeats this check before allocation and records the hash, architecture, input/output latent shapes, first and target canvases, sigma grid, evaluation counts, seed policy and audio policy in output metadata.

The implementation restores the original Diffusers modular block graph in `finally`, including cancellation and failure paths. A generation is rejected when the first canvas would not be smaller.

## Full-pipeline validation

The complete local H3 pipeline reached MP4 export in job `df767073-3ee5-4d86-9089-b199a504aec3`: 124 frames at 24 FPS and 512×512 from a 256×256 first pass, with 8 Turbo evaluations, learned `B×24×37×16×16 → B×24×37×32×32` latent enlargement, and 4 refinement evaluations at strength `0.18`. The 5.1667-second file contains generated audio and its embedded MP4 JSON exactly matches the sidecar JSON. **This run did not pass visual validation:** frame 60 contains a magenta grid artifact and no usable subject. Matching single-pass runs at both 256×256 and 512×512 render the expected scene, isolating the defect to the two-pass path.

Source comparison found that the standalone upscaler port omitted the released model's required 24-channel mean/std transform before inference and its inverse after inference. That transform is now restored and covered by a CPU contract test. The failed artifact remains evidence of the defect and of export/metadata plumbing only.

- Output: `outputs/2026-09-22/h3-generate-df767073.mp4`
- Generation receipt: `data/qa/h3_two_pass_export_verification.json`
- Load profile: `data/qa/h3_gpu_load_profile.json`

The corrected path then passed two real GPU generations:

- Job `8d623d35`: 256×256 → 512×512, 124 frames, 24 FPS, 8+4 evaluations, strength 0.18. Frames 0/60/123 were inspected; frames 60/123 retain the red mug and woven mat without the former grid. Frame 0 matches the first-pass opening state. The file has audio, decodes to 124 frames and has identical embedded/sidecar metadata. Receipt: `data/qa/latent_corrected_export_verification.json`.
- Job `e009d9e2`: the default landscape path, 480×288 → 960×544, with the same 124 frames, 24 FPS, 8+4 evaluations and strength 0.18. Frames 0/60/123 retain the red mug, table, chair and window through the camera move without grid artifacts. Audio and metadata match. Receipt: `data/qa/latent_landscape_export_verification.json`; output SHA256 `8af488ea3e38a493b72ad0e2a5545fbb5946c2ae5e1dec9bd951e2003ae000ab`.

Latent 2-pass is therefore available when a pinned checkpoint passes inventory validation, while remaining OFF by default. The validated defaults are strength 0.18 and four refinement evaluations. The GPU-staged INT8 model load reached a maximum process RSS of 73.37 GB while leaving at least 39.45 GB of system memory available; the sampled GPU peak was 68,935 MiB. Turbo8 registered 312 target modules and SageAttention was applied. The 1344×768 and 345-frame limits still require separate quality and memory measurements.
