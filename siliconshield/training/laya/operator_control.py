"""Read bounded operator requests only at optimizer-update boundaries."""

import json
import math
from pathlib import Path


def apply_control(output, run_id, progress, encoder_lr):
    path = Path(output) / "operator-control.json"
    if not path.exists():
        return None
    request = json.loads(path.read_text())
    if (
        request.get("run_id") != run_id
        or type(request.get("sequence")) is not int
    ):
        raise ValueError(
            "Operator request belongs to another run or has no sequence"
        )
    if request["sequence"] <= progress.get("operator_sequence", 0):
        return None
    action = request.get("action")
    if action == "adjust_lr":
        value = request.get("lr")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Operator learning rate must be finite")
        manifest = json.loads((Path(output) / "run-manifest.json").read_text())
        budget = manifest["intervention_budget"]
        low, high = budget["lr_bounds"]
        if (
            not low <= value <= high
            or progress.get("operator_lr_changes", 0)
            >= budget["max_lr_adjustments"]
        ):
            raise ValueError("Operator learning-rate budget exceeded")
        if encoder_lr <= 0:
            raise ValueError("Cannot scale a frozen encoder learning rate")
        progress["lr_scale"] = value / encoder_lr
        progress["operator_lr_changes"] = (
            progress.get("operator_lr_changes", 0) + 1
        )
    elif action != "stop":
        raise ValueError("Unsupported optimizer-boundary intervention")
    progress["operator_sequence"] = request["sequence"]
    return action
