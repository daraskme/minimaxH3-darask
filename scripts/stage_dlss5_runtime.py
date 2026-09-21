from __future__ import annotations

"""Download and verify the pinned experimental DLSS 5 NR package.

This script only stages files.  It never imports or executes downloaded code.
Execution remains a separate, explicit validation step.
"""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ASSET_NAME = "dlss5-video-player-v0.24.0-win64.zip"
ASSET_URL = (
    "https://github.com/2600th/dlss5-video-player/releases/download/"
    "dlss5-video-player-v0.24.0/" + ASSET_NAME
)
ASSET_SIZE = 327_555_972
ASSET_SHA256 = "66947ca33b84459d13b3002cfea153f295c1d0543260713eced6b9766a136e78"
PACKAGE_DIRECTORY = "DLSSVideoPlayer-v0.24.0-win64"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256(path).lower() != digest.lower():
        raise RuntimeError(f"hash/size mismatch: {path}")


def verify_package(package: Path, *, pristine: bool) -> None:
    manifest = package / "PACKAGE_MANIFEST.txt"
    if not manifest.is_file():
        raise RuntimeError("PACKAGE_MANIFEST.txt is missing")
    lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    if not lines or lines[0] != "ProductVersion=0.24.0":
        raise RuntimeError("unexpected package version")
    entries = 0
    for line in lines[2:]:
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) < 3:
            raise RuntimeError("malformed package manifest")
        relative = PurePosixPath(fields[0])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("unsafe package manifest path")
        target = (package / Path(*relative.parts)).resolve()
        if not target.is_relative_to(package.resolve()):
            raise RuntimeError("package manifest escaped the destination")
        # The worker deliberately rewrites this local configuration during its
        # one-time repair restart.  Immutable binaries remain covered by both
        # the package manifest and the project's runtime lock.
        if not pristine and relative.as_posix() == "neural-runtime/ReShade.ini":
            if not target.is_file() or target.stat().st_size > 64 * 1024:
                raise RuntimeError("mutable ReShade.ini is missing or unexpectedly large")
        else:
            require_file(target, int(fields[1]), fields[2])
        entries += 1
    if entries < 40:
        raise RuntimeError("package manifest is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the pinned, unsigned community DLSS 5 NR package")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded exact release ZIP")
    args = parser.parse_args()
    downloads = ROOT / ".cache" / "downloads"
    destination_root = ROOT / ".cache" / "runtimes" / "dlss5-v0.24.0"
    destination = destination_root / PACKAGE_DIRECTORY
    archive = args.archive.resolve(strict=True) if args.archive else downloads / ASSET_NAME
    if destination.is_dir():
        verify_package(destination, pristine=False)
        print(f"Already verified: {destination}")
        return 0
    downloads.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        request = urllib.request.Request(ASSET_URL, headers={"User-Agent": "H3Studio/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            require_file(partial, ASSET_SIZE, ASSET_SHA256)
            os.replace(partial, archive)
        finally:
            partial.unlink(missing_ok=True)
    require_file(archive, ASSET_SIZE, ASSET_SHA256)

    destination_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dlss5-stage-", dir=destination_root) as temporary:
        stage = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as package_zip:
            for item in package_zip.infolist():
                relative = PurePosixPath(item.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("unsafe ZIP path")
                target = (stage / Path(*relative.parts)).resolve()
                if not target.is_relative_to(stage):
                    raise RuntimeError("ZIP member escaped the staging directory")
            package_zip.extractall(stage)
        extracted = stage / PACKAGE_DIRECTORY
        verify_package(extracted, pristine=True)
        os.replace(extracted, destination)
    print(f"Staged without execution: {destination}")
    print("Next: run validate_dlss5_nr_runtime.py with explicit acknowledgement and a short CFR MP4.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DLSS 5 runtime staging failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
