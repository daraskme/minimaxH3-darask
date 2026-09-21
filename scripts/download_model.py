from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
destination = ROOT / "models" / "MiniMax-H3"
snapshot_download(
    repo_id="MiniMaxAI/MiniMax-H3",
    revision="42ed227ee7df40d41602854ae760620d6eb651fe",
    local_dir=destination,
    allow_patterns=[
        "model_index.json", "modular_model_index.json",
        "transformer/*", "text_encoder/*", "tokenizer/*", "processor/*",
        "vae/*", "audio_vae/*", "scheduler/*", "audio_scheduler/*",
    ],
)
print(destination)
