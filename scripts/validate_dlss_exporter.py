from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3studio import dlss  # noqa: E402
from h3studio.metadata import _ffmpeg_path  # noqa: E402
from h3studio.video import inspect_video, sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the official DLSS SR exporter on a short local CFR MP4")
    parser.add_argument("source", type=Path, help="Short CFR SDR MP4 used only for local validation")
    args = parser.parse_args()
    source = args.source.resolve()
    info = inspect_video(source)
    if not info["is_cfr"] or info["frame_count"] < 2 or info["frame_count"] > 240:
        raise SystemExit("Validation source must be a short CFR MP4 with 2..240 frames")
    if info["width"] * 2 > 8192 or info["height"] * 2 > 8192:
        raise SystemExit("Validation source is too large")

    manifest = json.loads(dlss.BUILD_MANIFEST.read_text(encoding="utf-8-sig"))
    if dlss._sha256(dlss.SR_WORKER) != str(manifest.get("worker_sha256", "")):
        raise SystemExit("Exporter hash does not match its build manifest")
    if dlss._sha256(dlss.SR_RUNTIME) != dlss.OFFICIAL_DLSS_SR_SHA256:
        raise SystemExit("Official DLSS runtime hash mismatch")

    ffmpeg = Path(_ffmpeg_path(None)).resolve()
    (ROOT / ".cache").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h3studio-dlss-", dir=ROOT / ".cache") as folder:
        temporary = Path(folder)
        output = temporary / "validated.mp4"
        receipt = temporary / "receipt.json"
        command = [
            str(dlss.SR_WORKER), "--h3-dlss-sr", "--source", str(source), "--output", str(output),
            "--receipt", str(receipt), "--ffmpeg-dir", str(dlss.ffmpeg_helper_directory(ffmpeg)),
            "--target-width", str(int(info["width"]) * 2),
            "--target-height", str(int(info["height"]) * 2),
            "--expected-fps", format(float(info["fps"]), ".12g"),
            "--expected-frames", str(int(info["frame_count"])),
        ]
        timeout = max(180.0, float(info["duration_seconds"]) * 60.0)
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=timeout, shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not receipt.is_file():
            raise SystemExit(f"DLSS SR validation failed ({result.returncode}):\n{result.stderr[-8000:]}")
        evidence = json.loads(receipt.read_text(encoding="utf-8"))
        output_info = inspect_video(output)
        expected_frames = int(info["frame_count"])
        if (
            evidence.get("contract") != "h3studio-dlss-sr-v1" or evidence.get("ok") is not True
            or evidence.get("frames") != expected_frames or evidence.get("dlss_evaluations") != expected_frames
            or output_info["width"] != int(info["width"]) * 2
            or output_info["height"] != int(info["height"]) * 2
            or output_info["frame_count"] != expected_frames or not output_info["is_cfr"]
            or int(output_info["fps_numerator"]) != int(info["fps_numerator"])
            or int(output_info["fps_denominator"]) != int(info["fps_denominator"])
        ):
            raise SystemExit(f"DLSS SR output evidence mismatch: {evidence} / {output_info}")

    driver = dlss.driver_version()
    if not driver:
        raise SystemExit("NVIDIA driver version could not be read")
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=False,
        capture_output=True, text=True, timeout=4, shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    gpu = next((line.strip() for line in gpu_query.stdout.splitlines() if line.strip()), "unknown")
    manifest["smoke_validated"] = True
    manifest["smoke_validation"] = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "driver": driver,
        "gpu": gpu,
        "source_sha256": sha256_file(source),
        "source": {
            "width": info["width"], "height": info["height"], "fps_numerator": info["fps_numerator"],
            "fps_denominator": info["fps_denominator"], "frames": info["frame_count"],
        },
        "receipt": evidence,
    }
    dlss.BUILD_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["smoke_validation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
