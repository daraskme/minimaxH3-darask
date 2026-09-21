import json
from pathlib import Path

import pytest

from h3studio.inventory import resolve_asset, scan_inventory


def test_repository_completeness_and_traversal(tmp_path: Path):
    repo = tmp_path / "models" / "H3"
    shard_dir = repo / "transformer"
    shard_dir.mkdir(parents=True)
    required = ("text_encoder", "tokenizer", "processor", "vae", "audio_vae", "transformer", "scheduler", "audio_scheduler")
    manifest = {"_class_name": "MiniMaxH3ModularPipeline"}
    for name in required:
        manifest[name] = ["library", "Class", {"subfolder": name}]
        (repo / name).mkdir(parents=True, exist_ok=True)
    (repo / "modular_model_index.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("text_encoder", "vae", "audio_vae", "transformer"):
        (repo / name / "config.json").write_text("{}", encoding="utf-8")
    (repo / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (repo / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (repo / "processor" / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (repo / "processor" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (repo / "processor" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (repo / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (repo / "audio_scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (repo / "vae" / "weights.safetensors").write_bytes(b"vae")
    (repo / "audio_vae" / "weights.safetensors").write_bytes(b"audio")
    (repo / "text_encoder" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "text.safetensors"}}), encoding="utf-8"
    )
    (repo / "text_encoder" / "text.safetensors").write_bytes(b"text")
    (repo / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "transformer.safetensors"}}), encoding="utf-8"
    )
    (repo / "transformer" / "transformer.safetensors").write_bytes(b"transformer")
    (shard_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    (shard_dir / "model-00001-of-00002.safetensors").write_bytes(b"one")
    inventory = scan_inventory(tmp_path / "models")
    assert inventory["models"][0]["ready"] is False
    assert "1 shards" in inventory["models"][0]["reason"]
    with pytest.raises(FileNotFoundError):
        resolve_asset(tmp_path / "models", "repository", "H3")
    (shard_dir / "model-00002-of-00002.safetensors").write_bytes(b"two")
    assert resolve_asset(tmp_path / "models", "repository", "H3") == repo.resolve()
    with pytest.raises(FileNotFoundError):
        resolve_asset(tmp_path / "models", "repository", "../outside")


def test_lora_root_is_not_doubled(tmp_path: Path):
    lora_root = tmp_path / "loras"
    target = lora_root / "minimaxH3" / "style.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"lora")
    assert resolve_asset(lora_root, "lora", "minimaxH3/style.safetensors") == target.resolve()
