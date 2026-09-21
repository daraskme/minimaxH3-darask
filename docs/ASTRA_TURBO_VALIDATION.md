# Astra Turbo and multiple-LoRA compatibility checks

2026-09-21, standalone project environment: Diffusers 0.40, Transformers 5.17, PEFT 0.21, TorchAO 0.18. No ComfyUI runtime was imported. This evidence establishes loader shape compatibility and a small quantized GPU adapter forward; it does not establish completed H3 video generation or its quality.

## Existing reusable Turbo weights

Read only safetensors JSON headers and small scalar alpha tensors, representing large weights as meta tensors. Converted through the **installed** Diffusers `_convert_non_diffusers_minimax_h3_lora_to_diffusers`; checked every converted adapter input/output dimension against an allocation-free default native `MiniMaxH3Transformer3DModel`.

| File under D:\comfyui-models\loras\minimaxH3 | Converted tensors | Shape errors |
|---|---:|---:|
| minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors | 624 / 312 target modules | 0 |
| minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors | 624 / 312 target modules | 0 |
| minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors | 624 / 312 target modules | 0 |

Metadata identifies a full FL2VA BF16 base and converted Diffusers PEFT source. These chosen files contain attention/MLP adapters and do not require pruned AdaLN shapes. The installed loader reverses the QKV fusion and SwiGLU row swap. Turbo8 uses training rank 128 and alpha 8; its fused-QKV rank 384/alpha 24 preserves the same 0.0625 scale. Per-key alpha is folded by the converter, so application adapter weight 1.0 is appropriate; do not multiply by 0.0625 again.

Use `pipe.load_lora_weights(directory, weight_name=filename, adapter_name="turbo8", use_safetensors=True, local_files_only=True)`. Set selected adapters and application weights with `pipe.set_adapters(names, adapter_weights=weights)`. Verify adapter registration because the loader can warn and return without loading unsupported layouts.

## Matching the real distilled schedule

[ModelTC's official Turbo specifications](https://github.com/ModelTC/Minimax-H3-Turbo) distinguish the adapters:

- The installed eight-step v1.0 **without** `_768p` was trained at 544p with video/audio shifts **12 / 3**, supports eight or four evaluations.
- The four-step v1.0 `_768p` was trained at 1344×768 with shifts **6 / 3** and four evaluations. Do not reuse shift 12 for this preset.
- The separate upstream eight-step `_768p` variant uses 6 / 3; it is not the chosen local eight-step file.
- Four-step v1.1 shape compatibility passed, but this review has not established its exact intended schedule from a primary version-specific model card. Prefer the verified v1.0 presets initially.

For N evaluations the official unshifted points are `(N-i)/N`, i from 0 through N-1, followed by zero. Native Diffusers `num_inference_steps` counts grid points including zero, so request N+1 points and verify the actual scheduler timesteps. Example four-evaluation video/audio shifts 12 / 3 produce video `[1, 0.9730, 0.9231, 0.8000] -> 0` and audio `[1, 0.9, 0.75, 0.5] -> 0`.

## Actual INT8 GPU multiple-adapter forward

Built a 128×128 BF16 CUDA linear layer, quantized with `Int8WeightOnlyConfig(version=2)`, then attached two independent PEFT rank-4 adapters. Both adapters independently changed the output, combined output was finite, and base weights remained `Int8Tensor`.

Observed maximum effects were 0.0234375 and 0.0390625 separately, 0.05859375 together. The combined-additivity error was 0.0078125 at BF16 precision. This validates the installed libraries' basic unfused multiple-LoRA path on this GPU, not the full H3 integration.

PEFT warned that manually quantized models lack metadata needed for merge/unmerge. Keep adapters **unfused**; do not assume `fuse_lora` works on this route. Use clear adapter names such as `turbo8` and `style001`, avoiding short names contained in the reserved `lora_` prefix.

## Remaining acceptance

Load the real native checkpoint with the selected Turbo8, verify all adapters registered, generate with 9 grid points and shifts 12/3, capture actual denoising count and audio/video output, and compare with the baseline. Then verify the four-step v1.0 preset at 5 grid points and shifts 6/3. No new Turbo model download is needed for these existing candidates.
