from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3studio.metadata import _ffmpeg_path
from h3studio.seedvr2 import SeedVR2Engine, capability
from h3studio.video import inspect_video


def main() -> int:
    before = capability(ROOT)
    if not before.get("installed"):
        raise RuntimeError(before.get("reason", "SeedVR2 prerequisites are not installed"))
    output_root = ROOT / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h3studio-seedvr2-selfcheck-", dir=output_root) as directory:
        work = Path(directory)
        source = work / "source.mp4"
        output = work / "restored.mp4"
        subprocess.run(
            [
                _ffmpeg_path(None), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24000/1001",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-frames:v", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(source),
            ],
            check=True,
        )
        source_info = inspect_video(source)
        settings = {"scale": 2, "seed": 123456, "source": source_info}

        def progress(value: float, phase: str) -> None:
            print(f"[{value:5.1%}] {phase}", flush=True)

        result = SeedVR2Engine(ROOT).selfcheck(
            source, output, settings, progress, threading.Event(),
        )
        verified = inspect_video(output)
        print({"engine": result, "output": verified, "capability": capability(ROOT)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
