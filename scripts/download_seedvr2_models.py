from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "seedvr2-standalone"
REVISION = "4490bd1f482e026674543386bb2a4d176da245b9"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

from h3studio.video import sha256_file
from h3studio.seedvr2 import DIT_MODEL, MODEL_SHA256, VAE_MODEL
from src.utils.downloads import download_weight


def main() -> int:
    revision_path = SOURCE / "UPSTREAM_REVISION"
    if not revision_path.is_file() or revision_path.read_text(encoding="ascii").strip() != REVISION:
        raise RuntimeError("Pinned SeedVR2 standalone source is missing or has the wrong revision")
    destination = ROOT / "models" / "seedvr2" / "SeedVR2-3B"
    if not download_weight(DIT_MODEL, VAE_MODEL, str(destination)):
        raise RuntimeError("Could not download the pinned SeedVR2 model bundle")
    for name, expected in MODEL_SHA256.items():
        actual = sha256_file(destination / name)
        if actual != expected:
            raise RuntimeError(f"SeedVR2 SHA256 mismatch for {name}: {actual}")
        print(f"verified {name}: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
