"""H3 Studio's narrow Windows launcher for the pinned SeedVR2 CLI.

The upstream loader intentionally uses ``strict=False``.  For the one exact 3B
FP16/VAE pair supported by H3 Studio, silently skipped tensors are unsafe: a run
must fail before inference if the checkpoint and architecture do not match.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in (str(ROOT), os.environ.get("PYTHONPATH", "")) if part
)

from src.core import model_loader  # noqa: E402


def _strict_standard_weights(model, state, used_meta, model_type, model_type_lower, debug=None):
    debug.start_timer(f"{model_type_lower}_state_apply")
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    unresolved = [name for name, value in model.named_parameters() if value.is_meta]
    if missing or unexpected or unresolved:
        raise RuntimeError(
            f"{model_type} checkpoint mismatch: missing={missing[:12]}, "
            f"unexpected={unexpected[:12]}, unresolved_meta={unresolved[:12]}"
        )
    action = "materialized" if used_meta else "applied"
    debug.end_timer(f"{model_type_lower}_state_apply", f"{model_type} weights {action}")
    debug.log(f"{model_type} weights strictly verified", category=model_type_lower, force=True)
    return model


model_loader._load_standard_weights = _strict_standard_weights
runpy.run_path(str(ROOT / "inference_cli.py"), run_name="__main__")
