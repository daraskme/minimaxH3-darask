from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the isolated, hash-locked community DLSS 5 Neural Rendering worker"
    )
    parser.add_argument(
        "--runtime-directory", type=Path,
        default=ROOT / ".cache" / "runtimes" / "dlss5-v0.24.0" / "DLSSVideoPlayer-v0.24.0-win64" / "neural-runtime",
    )
    parser.add_argument(
        "--allow-unsigned-runtime", action="store_true",
        help="Required acknowledgement: the pinned Neural Rendering runtime and proxy are unsigned community modifications",
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="Short local CFR SDR MP4 used for a real feature-18 file-export smoke test",
    )
    args = parser.parse_args()
    if not args.allow_unsigned_runtime:
        raise SystemExit("Refusing preflight without --allow-unsigned-runtime")
    runtime = args.runtime_directory.resolve(strict=True)
    os.environ["H3STUDIO_DLSS5_NR_DIRECTORY"] = str(runtime)
    os.environ["H3STUDIO_ALLOW_UNSIGNED_DLSS_NR"] = "1"

    from h3studio import dlss
    from h3studio.video import inspect_video, sha256_file

    dlss.verify_nr_locked_files()
    parsed_driver = dlss._parse_version(dlss.driver_version())
    if parsed_driver is None or parsed_driver < dlss.MIN_DRIVER:
        raise SystemExit(f"NVIDIA driver {dlss.driver_version() or 'unknown'} is below 610.47")
    preflight = dlss.run_neural_preflight()
    source = args.source.resolve(strict=True)
    source_info = inspect_video(source)
    if not source_info["is_cfr"] or source_info["duration_seconds"] > 10:
        raise SystemExit("Validation source must be a CFR MP4 no longer than 10 seconds")
    if source_info["width"] * source_info["height"] > 1920 * 1080:
        raise SystemExit("Validation source must be at most 1920x1080")
    source_info["sha256"] = sha256_file(source)
    dlss.NR_VALIDATION.unlink(missing_ok=True)
    temporary_root = ROOT / "data" / "qa"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dlss5-nr-validation-", dir=temporary_root) as folder:
        staging = Path(folder) / "neural-output.mkv"
        receipt = dlss.run_neural_rendering(
            source, staging, source_info, lambda *_: None, threading.Event(),
            require_export_validation=False,
        )
        export_validation = {
            "ok": True,
            "driver": dlss.driver_version(),
            "source_sha256": source_info["sha256"],
            "source": {
                key: source_info[key] for key in (
                    "width", "height", "fps", "fps_numerator", "fps_denominator",
                    "duration_seconds", "frame_count", "is_cfr",
                )
            },
            "output": receipt["output"],
            "receipt": {key: receipt[key] for key in (
                "feature18_armed_before_capture", "upscaling_off",
                "inline_interception_contract", "feature18_created",
                "feature18_evaluated", "later_failure", "frame_count",
                "native_evaluations", "verified_neural_frames",
                "highest_observed_evaluation", "peak_local_vram_mib",
            )},
        }
    validation = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "user_authorized_unsigned_runtime": True,
        "protocol_version": dlss.WIRE_VERSION,
        "player_commit": dlss.PLAYER_COMMIT,
        "worker_sha256": dlss._sha256(dlss.NR_WORKER),
        "runtime_lock_sha256": dlss._sha256(dlss.NR_LOCK),
        "driver": dlss.driver_version(),
        "preflight_ok": True,
        "preflight": preflight,
        "export_validation": export_validation,
    }
    temporary = dlss.NR_VALIDATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, dlss.NR_VALIDATION)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
