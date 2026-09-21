from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA_ROOT = ROOT / "models" / "loras"


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    host: str = "127.0.0.1"
    port: int = 7862
    model_root: Path = ROOT / "models"
    lora_root: Path = DEFAULT_LORA_ROOT
    latent_upscaler_root: Path = ROOT / "models" / "latent_upscalers"
    outputs: Path = ROOT / "outputs"
    uploads: Path = ROOT / "data" / "uploads"
    thumbnails: Path = ROOT / "data" / "thumbnails"
    state: Path = ROOT / "data" / "studio.sqlite3"
    ffmpeg: str | None = None

    @classmethod
    def load(cls) -> "Settings":
        path = ROOT / "config.json"
        values: dict[str, object] = {}
        if path.exists():
            values = json.loads(path.read_text(encoding="utf-8"))
        env = os.environ
        return cls(
            host=str(env.get("H3STUDIO_HOST", values.get("host", "127.0.0.1"))),
            port=int(env.get("H3STUDIO_PORT", values.get("port", 7862))),
            model_root=Path(env.get("H3STUDIO_MODEL_ROOT", values.get("model_root", ROOT / "models"))),
            lora_root=Path(env.get("H3STUDIO_LORA_ROOT", values.get("lora_root", DEFAULT_LORA_ROOT))),
            latent_upscaler_root=Path(env.get("H3STUDIO_LATENT_UPSCALER_ROOT", values.get("latent_upscaler_root", ROOT / "models" / "latent_upscalers"))),
            outputs=Path(env.get("H3STUDIO_OUTPUTS", values.get("outputs", ROOT / "outputs"))),
            uploads=Path(env.get("H3STUDIO_UPLOADS", values.get("uploads", ROOT / "data" / "uploads"))),
            thumbnails=Path(env.get("H3STUDIO_THUMBNAILS", values.get("thumbnails", ROOT / "data" / "thumbnails"))),
            state=Path(env.get("H3STUDIO_STATE", values.get("state", ROOT / "data" / "studio.sqlite3"))),
            ffmpeg=env.get("H3STUDIO_FFMPEG", values.get("ffmpeg")),
        )

    def public_dict(self) -> dict[str, object]:
        data = asdict(self)
        data.pop("root", None)
        return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}

    def ensure_dirs(self) -> None:
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.thumbnails.mkdir(parents=True, exist_ok=True)
        self.latent_upscaler_root.mkdir(parents=True, exist_ok=True)
        self.state.parent.mkdir(parents=True, exist_ok=True)


settings = Settings.load()
