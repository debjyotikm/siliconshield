"""A shared file lock keeps Glimmer training, serving, and evaluation separate."""

import fcntl
import os
from pathlib import Path


def acquire_gpu():
    path = os.environ.get("SILICONSHIELD_GPU_LOCK")
    if not path:
        raise ValueError(
            "Set SILICONSHIELD_GPU_LOCK to the shared host-mounted Glimmer GPU lock"
        )
    root = Path(path)
    root.parent.mkdir(parents=True, exist_ok=True)
    handle = root.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(
            "Glimmer GPU is already owned by training, serving, or evaluation"
        ) from exc
    return handle
