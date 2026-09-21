from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "downloads"

REALESRGAN_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
REALESRGAN_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
REALESRGAN_TARGET = ROOT / "models" / "upscalers" / "RealESRGAN_x2plus.pth"

RIFE_URL = "https://drive.usercontent.google.com/download?id=1gViYvvQrtETBgU1w8axZSsr7YUuw31uy&export=download"
RIFE_ARCHIVE_SHA256 = "c2452dd2b244947d4be580156bbead60d6b72af5736860f7d6b3f99648c9c4cc"
RIFE_WEIGHT_SHA256 = "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"
RIFE_MEMBER = "train_log/flownet.pkl"
RIFE_TARGET = ROOT / "models" / "interpolators" / "rife4.26" / "flownet.pkl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, expected: str) -> None:
    if target.is_file() and sha256(target) == expected:
        print(f"verified: {target.relative_to(ROOT)}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "H3-Studio/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination, length=1024 * 1024)
        actual = sha256(temporary)
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch for {target.name}: {actual}")
        os.replace(temporary, target)
        print(f"installed: {target.relative_to(ROOT)}")
    finally:
        temporary.unlink(missing_ok=True)


def install_rife() -> None:
    if RIFE_TARGET.is_file() and sha256(RIFE_TARGET) == RIFE_WEIGHT_SHA256:
        print(f"verified: {RIFE_TARGET.relative_to(ROOT)}")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "RIFEv4.26_0921.zip"
    download(RIFE_URL, archive, RIFE_ARCHIVE_SHA256)
    RIFE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    temporary = RIFE_TARGET.with_suffix(".pkl.part")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(archive) as package:
            info = package.getinfo(RIFE_MEMBER)
            with package.open(info) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        actual = sha256(temporary)
        if actual != RIFE_WEIGHT_SHA256:
            raise RuntimeError(f"SHA256 mismatch for RIFE weight: {actual}")
        os.replace(temporary, RIFE_TARGET)
        print(f"installed: {RIFE_TARGET.relative_to(ROOT)}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    download(REALESRGAN_URL, REALESRGAN_TARGET, REALESRGAN_SHA256)
    install_rife()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"postprocess model setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
