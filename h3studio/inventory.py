from __future__ import annotations

import json
import re
from pathlib import Path


MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"}


def _safe_files(root: Path, folder: str) -> list[dict[str, object]]:
    base = (root / folder).resolve()
    if not base.is_dir():
        return []
    result = []
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
            result.append({
                "name": path.relative_to(base).as_posix(),
                "size": path.stat().st_size,
            })
    return sorted(result, key=lambda item: str(item["name"]).lower())


def _repository_status(path: Path) -> tuple[bool, str | None]:
    index = path / "modular_model_index.json"
    if not index.is_file():
        return False, "modular_model_index.json がありません"
    try:
        manifest = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "modular_model_index.json が壊れています"
    if manifest.get("_class_name") != "MiniMaxH3ModularPipeline":
        return False, "MiniMaxH3ModularPipeline のリポジトリではありません"
    required = ("text_encoder", "tokenizer", "processor", "vae", "audio_vae", "transformer", "scheduler", "audio_scheduler")
    for name in required:
        spec = manifest.get(name)
        if not isinstance(spec, list) or len(spec) < 3:
            return False, f"必須コンポーネント {name} がmanifestにありません"
        subfolder = spec[2].get("subfolder", name) if isinstance(spec[2], dict) else name
        folder = path / subfolder
        if not folder.is_dir() or not any(folder.iterdir()):
            return False, f"ダウンロード中（{subfolder} が未完了）"
    required_files = {
        "text_encoder": ("config.json",),
        "tokenizer": ("tokenizer_config.json",),
        "processor": ("preprocessor_config.json", "tokenizer_config.json"),
        "vae": ("config.json",),
        "audio_vae": ("config.json",),
        "transformer": ("config.json",),
        "scheduler": ("scheduler_config.json",),
        "audio_scheduler": ("scheduler_config.json",),
    }
    for folder_name, names in required_files.items():
        for filename in names:
            if not (path / folder_name / filename).is_file():
                return False, f"ダウンロード中（{folder_name}/{filename} が未完了）"
    for folder_name in ("tokenizer", "processor"):
        folder = path / folder_name
        has_tokens = (folder / "tokenizer.json").is_file() or (
            (folder / "vocab.json").is_file() and (folder / "merges.txt").is_file()
        )
        if not has_tokens:
            return False, f"ダウンロード中（{folder_name} vocabulary が未完了）"
    for name in ("text_encoder", "transformer"):
        folder = path / name
        indices = list(folder.glob("*.safetensors.index.json"))
        singles = [
            item for item in folder.glob("*.safetensors")
            if not re.search(r"-\d{5}-of-\d{5}\.safetensors$", item.name)
        ]
        if not indices and not singles:
            return False, f"ダウンロード中（{name} weight index が未完了）"
    for name in ("vae", "audio_vae"):
        folder = path / name
        if not list(folder.glob("*.safetensors")) and not list(folder.glob("*.bin")):
            return False, f"ダウンロード中（{name} weights が未完了）"
    missing: set[str] = set()
    for component in ("text_encoder", "transformer"):
        for shard_index in (path / component).glob("*.safetensors.index.json"):
            try:
                weights = json.loads(shard_index.read_text(encoding="utf-8")).get("weight_map", {})
                if not isinstance(weights, dict) or not weights:
                    return False, f"壊れたindex: {shard_index.name}"
                for filename in set(weights.values()):
                    if not (shard_index.parent / filename).is_file():
                        missing.add(str((shard_index.parent / filename).relative_to(path)))
            except (OSError, ValueError):
                return False, f"壊れたindex: {shard_index.name}"
    if missing:
        return False, f"ダウンロード中（残り {len(missing)} shards）"
    return True, None


def _model_repositories(root: Path) -> list[dict[str, object]]:
    if not root.is_dir():
        return []
    result = []
    for index in root.rglob("modular_model_index.json"):
        if ".cache" in index.parts:
            continue
        repo = index.parent
        relative = repo.relative_to(root)
        if relative.parts and relative.parts[0].lower() in {
            "loras", "latent_upscalers", "recovered_comfyui", "seedvr2",
        }:
            continue
        ready, reason = _repository_status(repo)
        result.append({"name": repo.relative_to(root).as_posix(), "ready": ready, "reason": reason})
    return sorted(result, key=lambda item: str(item["name"]).lower())


def scan_inventory(
    root: Path, lora_root: Path | None = None, latent_upscaler_root: Path | None = None,
) -> dict[str, object]:
    from .latent_upscale import scan_latent_upscalers

    latent_root = latent_upscaler_root or root / "latent_upscalers"
    return {
        "root": str(root),
        "models": _model_repositories(root),
        "lora_root": str(lora_root or root / "loras"),
        "loras": _safe_files((lora_root or root), "") if lora_root else _safe_files(root, "loras"),
        "latent_upscaler_root": str(latent_root),
        "latent_upscalers": scan_latent_upscalers(latent_root),
    }


def resolve_asset(root: Path, category: str, relative: str) -> Path:
    if category == "repository":
        target = (root.resolve() / relative).resolve()
        if target.is_relative_to(root.resolve()) and target.is_dir() and (target / "modular_model_index.json").is_file():
            ready, reason = _repository_status(target)
            if not ready:
                raise FileNotFoundError(reason)
            return target
        raise FileNotFoundError(f"Unknown model repository: {relative}")
    folders = {
        "model": ("diffusion_models", "unet"),
        "text_encoder": ("text_encoders", "clip"),
        "vae": ("vae",),
        "lora": ("",),
    }[category]
    for folder in folders:
        base = (root / folder).resolve()
        target = (base / relative).resolve()
        if target.is_relative_to(base) and target.is_file():
            return target
    raise FileNotFoundError(f"Unknown {category}: {relative}")
