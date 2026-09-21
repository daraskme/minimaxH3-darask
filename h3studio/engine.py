from __future__ import annotations

import gc
import importlib.util
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any


ProgressCallback = Callable[[float, str], None]


class GenerationCancelled(RuntimeError):
    pass


class StandaloneH3Engine:
    """Persistent standalone engine boundary.

    Heavy torch/diffusers imports deliberately happen in ``load`` so the web UI
    starts quickly and can show a precise setup error without allocating VRAM.
    """

    def __init__(
        self, model_root: Path, lora_root: Path | None = None, upload_root: Path | None = None,
        latent_upscaler_root: Path | None = None,
    ):
        self.model_root = model_root
        self.lora_root = lora_root or model_root / "loras"
        self.upload_root = upload_root
        self.latent_upscaler_root = latent_upscaler_root or model_root / "latent_upscalers"
        self._pipeline: Any = None
        self._manager: Any = None
        self._base_identity: tuple[Any, ...] | None = None
        self._lora_identity: tuple[tuple[str, float], ...] | None = None
        self._adapter_module_counts: dict[str, int] = {}
        self._runtime_lock = threading.RLock()
        self._last_error: str | None = None

    @staticmethod
    def diagnostics() -> dict[str, Any]:
        result: dict[str, Any] = {"engine": "standalone-diffusers", "ready": False}
        try:
            import torch

            result.update({
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                "precision": "int8",
                "compute_dtype": "bfloat16",
                "memory_profiles": ["auto", "low_memory"],
                "recommended_memory_profile": "auto",
            })
        except Exception as exc:
            result["error"] = f"PyTorch is not installed in the project environment: {exc}"
            return result
        try:
            import diffusers
            import peft
            import torchao
            import transformers
            from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
            from diffusers.hooks import apply_group_offloading
            from torchao.quantization import Int8WeightOnlyConfig
            from transformers import Qwen3VLForConditionalGeneration
            from transformers import TorchAoConfig as TransformersTorchAoConfig

            # Constructor validation catches incompatible TorchAO releases
            # without allocating model weights or GPU memory.
            Int8WeightOnlyConfig(version=2)
            required = (
                ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig,
                apply_group_offloading, Qwen3VLForConditionalGeneration, TransformersTorchAoConfig,
            )
            if any(item is None for item in required):
                raise ImportError("a required H3 runtime class is unavailable")

            attention_backends = ["sdpa"]
            if importlib.util.find_spec("sageattention"):
                try:
                    __import__("sageattention")
                    attention_backends.append("sage")
                except Exception as exc:
                    result["sage_error"] = str(exc)

            result["diffusers"] = diffusers.__version__
            result["transformers"] = transformers.__version__
            result["torchao"] = torchao.__version__
            result["peft"] = peft.__version__
            result["attention_backends"] = attention_backends
            result["ready"] = bool(torch.cuda.is_available())
            if not torch.cuda.is_available():
                result["error"] = "CUDA GPU is not available to the standalone engine"
        except Exception as exc:
            result["error"] = f"Standalone H3 runtime validation failed: {exc}"
        return result

    def generate(
        self,
        settings: dict[str, Any],
        output_path: Path,
        progress: ProgressCallback,
        cancel: threading.Event,
    ) -> dict[str, Any]:
        repo = self._safe_path(self.model_root, str(settings["model"]), directory=True)
        loras = [item for item in settings.get("loras", []) if item.get("enabled", True)]
        base_identity = (str(repo), settings.get("attention", "sage"), settings.get("memory_profile", "auto"))
        lora_identity = tuple((item["path"], float(item["weight"])) for item in loras)
        if self._pipeline is None or base_identity != self._base_identity:
            progress(0.02, "INT8モデルを読み込んでいます")
            try:
                self._load(
                    repo, str(settings.get("attention", "sage")),
                    str(settings.get("memory_profile", "auto")), progress, cancel,
                )
            except BaseException:
                self._unload()
                raise
            with self._runtime_lock:
                self._base_identity = base_identity
        if self._lora_identity != lora_identity:
            progress(0.14, "LoRA構成を適用しています")
            self._apply_loras(loras, progress, cancel)

        try:
            return self._generate_loaded(settings, output_path, progress, cancel, loras, repo)
        except BaseException:
            # A cancelled/failed forward can leave scheduler indices, staged
            # modules or offload hooks half advanced. Rebuild for the next job.
            self._unload()
            raise

    def preload(
        self, settings: dict[str, Any], progress: ProgressCallback, cancel: threading.Event,
    ) -> dict[str, Any]:
        repo = self._safe_path(self.model_root, str(settings["model"]), directory=True)
        loras = [item for item in settings.get("loras", []) if item.get("enabled", True)]
        base_identity = (str(repo), settings.get("attention", "sage"), settings.get("memory_profile", "auto"))
        loading_base = self._pipeline is None or self._base_identity != base_identity
        try:
            if loading_base:
                self._load(
                    repo, str(settings.get("attention", "sage")),
                    str(settings.get("memory_profile", "auto")), progress, cancel,
                )
                with self._runtime_lock:
                    self._base_identity = base_identity
            self._apply_loras(loras, progress, cancel)
            with self._runtime_lock:
                self._last_error = None
            return self.runtime_status()
        except BaseException as exc:
            if loading_base:
                self._unload()
            with self._runtime_lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            raise

    def configure_loras(
        self, loras: list[dict[str, Any]], progress: ProgressCallback, cancel: threading.Event,
    ) -> dict[str, Any]:
        if self._pipeline is None:
            raise RuntimeError("モデルが未読み込みです。LoRA設定は保留してください")
        enabled = [item for item in loras if item.get("enabled", True)]
        self._apply_loras(enabled, progress, cancel)
        return self.runtime_status()

    def runtime_status(self) -> dict[str, Any]:
        with self._runtime_lock:
            model = None
            if self._base_identity:
                path = Path(self._base_identity[0])
                try:
                    model = path.relative_to(self.model_root.resolve()).as_posix()
                except ValueError:
                    model = str(path)
            return {
                "loaded": self._pipeline is not None and self._base_identity is not None,
                "model": model,
                "attention": self._base_identity[1] if self._base_identity else None,
                "memory_profile": self._base_identity[2] if self._base_identity else None,
                "loras": [
                    {"path": path, "weight": weight} for path, weight in (self._lora_identity or ())
                ],
                "lora_target_module_counts": dict(self._adapter_module_counts),
                "error": self._last_error,
            }

    def release(self) -> None:
        """Release H3 GPU/model state before another GPU engine is loaded."""
        self._unload()

    def _generate_loaded(
        self, settings: dict[str, Any], output_path: Path, progress: ProgressCallback,
        cancel: threading.Event, loras: list[dict[str, Any]], repo: Path,
    ) -> dict[str, Any]:
        import torch
        from diffusers.utils import load_image
        from diffusers.utils.export_utils import encode_video

        if cancel.is_set():
            raise GenerationCancelled("cancelled before inference")

        pipe = self._pipeline
        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            raise RuntimeError("The loaded t2va/fl2va pipeline has no transformer component")
        refine = settings.get("latent_refine") or {}
        refine_enabled = bool(refine.get("enabled"))
        expected_first = max(1, int(settings["steps"]))
        expected_refine = int(refine.get("steps", 0)) if refine_enabled else 0
        expected = expected_first + expected_refine
        counter = {"value": 0}

        def before_forward(_module, _args):
            if cancel.is_set():
                raise GenerationCancelled("cancelled during denoising")
            counter["value"] += 1
            progress(0.20 + 0.65 * min(counter["value"], expected) / expected, f"生成中 {counter['value']} / {expected}")

        hook = transformer.register_forward_pre_hook(before_forward)
        try:
            recipe = settings.get("acceleration") or {}
            video_shift = float(recipe.get("video_flow_shift", 12.0))
            audio_shift = float(recipe.get("audio_flow_shift", 3.0))
            pipe.scheduler.set_shift(video_shift)
            pipe.audio_scheduler.set_shift(audio_shift)
            kwargs: dict[str, Any] = {
                "prompt": settings["prompt"],
                "height": int(settings["height"]),
                "width": int(settings["width"]),
                "num_frames": int(settings["frames"]),
                # H3 counts the terminal zero as one grid point. The UI exposes
                # actual transformer evaluations, so add that terminal point.
                "num_inference_steps": expected_first + 1,
                "generator": torch.Generator(device="cpu").manual_seed(int(settings["seed"])),
                "output_type": "pil",
                "output": ["videos", "audio", "sampling_rate"],
            }
            if settings.get("first_frame"):
                kwargs["image"] = load_image(str(self._safe_upload(str(settings["first_frame"]))))
            if settings.get("last_frame"):
                kwargs["last_image"] = load_image(str(self._safe_upload(str(settings["last_frame"]))))
            if refine_enabled:
                from .latent_refine import add_refine_noise, first_pass_blocks, refine_blocks, validate_refine_result
                from .latent_upscale import H3LatentUpscaler

                original_blocks = pipe._blocks
                try:
                    low_width = int(refine["low_width"]); low_height = int(refine["low_height"])
                    pipe._blocks = first_pass_blocks(original_blocks)
                    first_kwargs = {
                        **kwargs, "width": low_width, "height": low_height,
                        "output": ["latents", "audio_latents"],
                    }
                    first = pipe(**first_kwargs)
                    if cancel.is_set():
                        raise GenerationCancelled("cancelled before latent upscale")
                    upscaler = H3LatentUpscaler(self.latent_upscaler_root)
                    target_latent_height = int(settings["height"]) // 16
                    target_latent_width = int(settings["width"]) // 16
                    lifted, upscale_info = upscaler.upscale(
                        first["latents"], str(refine["model"]), target_latent_height, target_latent_width,
                        lambda value, phase: progress(0.58 + value * 0.10, phase), cancel,
                    )
                    refine_generator = torch.Generator(device="cpu").manual_seed(int(settings["seed"]))
                    noised = add_refine_noise(lifted, float(refine["strength"]), refine_generator)
                    pipe._blocks = refine_blocks(original_blocks)
                    second_kwargs = {
                        **kwargs,
                        "latents": noised,
                        "audio_latents": first["audio_latents"],
                        "num_inference_steps": expected_refine + 1,
                        "refine_strength": float(refine["strength"]),
                        "output": ["videos", "audio", "sampling_rate", "audio_latents"],
                    }
                    results = pipe(**second_kwargs)
                    validate_refine_result(first, results, int(settings["height"]), int(settings["width"]))
                finally:
                    pipe._blocks = original_blocks
            else:
                results = pipe(**kwargs)
        finally:
            hook.remove()
        if counter["value"] != expected:
            raise RuntimeError(f"Expected {expected} denoiser evaluations, observed {counter['value']}")
        if cancel.is_set():
            raise GenerationCancelled("cancelled before encode")
        progress(0.90, "映像と音声をMP4に書き出しています")
        encode_video(
            results["videos"][0], fps=24, output_path=str(output_path),
            audio=results["audio"][0], audio_sample_rate=results["sampling_rate"],
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Video encoder returned without creating a valid MP4")
        return {
            "name": "Diffusers MiniMaxH3ModularPipeline",
            "repository": str(repo),
            "model_revision": self._model_revision(repo),
            "precision": "torchao-int8-weight-only-v2",
            "memory_profile": settings.get("memory_profile", "auto"),
            "attention_requested": settings.get("attention", "sage"),
            "attention_applied": settings.get("attention", "sage"),
            "denoiser_evaluations": counter["value"],
            "sigma_grid_points": expected_first + 1,
            "video_flow_shift": video_shift,
            "audio_flow_shift": audio_shift,
            "loras_applied": [{"path": item["path"], "weight": item["weight"]} for item in loras],
            "lora_target_module_counts": dict(self._adapter_module_counts),
            "diffusers": __import__("diffusers").__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **({
                "latent_two_pass": {
                    "enabled": True,
                    "first_pass": {"width": int(refine["low_width"]), "height": int(refine["low_height"]), "steps": expected_first},
                    "target": {"width": int(settings["width"]), "height": int(settings["height"])},
                    "refine_steps": expected_refine, "strength": float(refine["strength"]),
                    "refine_sigma_grid": [
                        float(refine["strength"]) * (1.0 - index / expected_refine)
                        for index in range(expected_refine + 1)
                    ],
                    "refine_noise_seed": int(settings["seed"]),
                    "refine_noise_policy": "fresh CPU generator; x=(1-sigma)*clean+sigma*noise",
                    "upscaler": upscale_info,
                    "audio_policy": "preserved-clean-latents",
                }
            } if refine_enabled else {}),
        }

    @staticmethod
    def _model_revision(repo: Path) -> str | None:
        metadata = repo / ".cache" / "huggingface" / "download" / "modular_model_index.json.metadata"
        if metadata.is_file():
            lines = metadata.read_text(encoding="utf-8").splitlines()
            return lines[0].strip() if lines else None
        return None

    def _safe_path(self, root: Path, relative: str, directory: bool = False) -> Path:
        target = (root.resolve() / relative).resolve()
        valid = target.is_dir() if directory else target.is_file()
        if not target.is_relative_to(root.resolve()) or not valid:
            raise FileNotFoundError(f"Local asset not found: {relative}")
        return target

    def _safe_upload(self, name: str) -> Path:
        if self.upload_root is None:
            raise RuntimeError("Upload root is not configured")
        return self._safe_path(self.upload_root, name)

    def _unload(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._manager = None
        with self._runtime_lock:
            self._base_identity = None
            self._lora_identity = None
        self._adapter_module_counts = {}
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load(
        self, repo: Path, attention: str, memory_profile: str,
        progress: ProgressCallback, cancel: threading.Event,
    ) -> None:
        import torch
        from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
        from diffusers.hooks import apply_group_offloading
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import Qwen3VLForConditionalGeneration
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        self._unload()
        if cancel.is_set():
            raise GenerationCancelled("cancelled before model load")
        model_id = str(repo)
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(model_id, components_manager=manager, local_files_only=True)
        # Loading the 61 GB BF16 checkpoints on CPU before TorchAO conversion
        # retains Windows safetensors shard storage beside the INT8 tensors and
        # peaks above 128 GB once Qwen starts.  Quantize each parameter directly
        # on the 96 GB GPU; only one source tensor is staged at a time and the
        # resident INT8 models fit with the configured 12 GB execution reserve.
        cuda_device = torch.device("cuda:0")
        cuda_map = {"": cuda_device}
        progress(0.04, "TransformerをINT8で読み込んでいます")
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            model_id, subfolder="transformer", dtype=torch.bfloat16, local_files_only=True,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
                ],
            ), low_cpu_mem_usage=True, device_map=cuda_map,
        )
        if cancel.is_set():
            raise GenerationCancelled("cancelled during model load")
        if memory_profile == "low_memory":
            offload = dict(onload_device=cuda_device, offload_device=torch.device("cpu"), use_stream=True)
            transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
        progress(0.08, "Qwen3-VLをINT8で読み込んでいます")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", dtype=torch.bfloat16, local_files_only=True,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=["model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"],
            ),
            low_cpu_mem_usage=True, device_map=cuda_map,
        )
        if memory_profile == "low_memory":
            apply_group_offloading(text_encoder.model, offload_type="leaf_level", **offload)
        pipe.update_components(transformer=transformer, text_encoder=text_encoder)
        pipe.load_components(
            workflow="t2va", pretrained_model_name_or_path=model_id,
            dtype=torch.bfloat16, local_files_only=True,
        )
        required_components = (
            "transformer", "text_encoder", "tokenizer", "processor", "vae", "audio_vae",
            "scheduler", "audio_scheduler",
        )
        missing = [name for name in required_components if getattr(pipe, name, None) is None]
        if missing:
            raise RuntimeError(f"H3 component loading failed: {', '.join(missing)}")

        if attention == "sage":
            if importlib.util.find_spec("sageattention") is None:
                raise RuntimeError("SageAttention was requested but is not installed")
            pipe.transformer.set_attention_backend("sage")
        elif attention == "sdpa":
            pipe.transformer.set_attention_backend("native")
        else:
            raise ValueError(f"Unsupported attention backend: {attention}")

        pipe.transformer.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        if memory_profile == "auto":
            manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
        elif memory_profile == "low_memory":
            pipe.vae.to("cuda")
            pipe.audio_vae.to("cuda")
        else:
            raise ValueError(f"Unsupported memory profile: {memory_profile}")

        self._manager = manager
        self._pipeline = pipe

    def _apply_loras(
        self, loras: list[dict[str, Any]], progress: ProgressCallback, cancel: threading.Event,
    ) -> None:
        pipe = self._pipeline
        if pipe is None:
            raise RuntimeError("H3 pipeline is not loaded")
        desired = tuple((str(item["path"]), float(item["weight"])) for item in loras)
        if desired == self._lora_identity:
            return
        try:
            if self._lora_identity is not None:
                pipe.unload_lora_weights()
            self._adapter_module_counts = {}
            adapter_names: list[str] = []
            adapter_weights: list[float] = []
            for index, spec in enumerate(loras):
                if cancel.is_set():
                    raise GenerationCancelled("cancelled during LoRA load")
                lora_path = self._safe_path(self.lora_root, str(spec["path"]))
                name = f"style{index + 1:03d}"
                progress(0.13 + 0.01 * index, f"LoRAを読み込んでいます {index + 1}/{len(loras)}")
                pipe.load_lora_weights(
                    str(lora_path.parent), weight_name=lora_path.name, adapter_name=name,
                    use_safetensors=True, local_files_only=True,
                )
                registered = getattr(pipe.transformer, "peft_config", {})
                if name not in registered:
                    raise RuntimeError(f"LoRA did not match any MiniMax H3 module: {spec['path']}")
                matched = sum(
                    1 for module in pipe.transformer.modules()
                    if name in getattr(module, "lora_A", {}) or name in getattr(module, "lora_B", {})
                )
                if matched == 0:
                    raise RuntimeError(f"LoRA registered but matched zero target modules: {spec['path']}")
                self._adapter_module_counts[name] = matched
                adapter_names.append(name)
                adapter_weights.append(float(spec["weight"]))
            if adapter_names:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
                active = pipe.get_active_adapters()
                if list(active) != adapter_names:
                    raise RuntimeError(f"LoRA activation order mismatch: expected {adapter_names}, got {active}")
            with self._runtime_lock:
                self._lora_identity = desired
                self._last_error = None
        except BaseException as exc:
            try:
                pipe.unload_lora_weights()
            finally:
                self._adapter_module_counts = {}
                with self._runtime_lock:
                    self._lora_identity = None
                    self._last_error = f"{type(exc).__name__}: {exc}"
            raise
