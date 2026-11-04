"""Restricted subprocess execution for untrusted inputs and agents.

Compiler, simulator, and coding-agent CLIs run with an allowlisted
environment and bounded resources so an arbitrary RTL file or agent
session cannot inherit operator secrets or consume the host without
limit. This is process-level hygiene, not a filesystem sandbox:
untrusted work still needs deployment-level isolation (a container
or separate host) before it may touch private files. Filesystem
access itself is unchanged here.
"""

import math
import os

# Variables a toolchain or login-based CLI needs. Everything else in
# os.environ — including exported API keys and service credentials —
# is dropped unless the caller passes it explicitly via ``extra``.
BASE_ENV_KEEP = frozenset(
    {
        "PATH",
        "SystemRoot",
        "SYSTEMROOT",
        "LANG",
        "LC_ALL",
        "LANGUAGE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USER",
        "LOGNAME",
    }
)


def restricted_env(extra=None):
    """Return an allowlisted environment for one child process.

    ``extra`` names additional variables the caller vouches for
    (for example a pinned runtime root the worker must import).
    """
    keep = set(BASE_ENV_KEEP)
    if extra:
        if isinstance(extra, str):
            raise ValueError("Sandbox env extras must be a name list")
        keep.update(extra)
    return {key: value for key, value in os.environ.items() if key in keep}


def limited_preexec(timeout_seconds, address_gb=16):
    """Return a preexec_fn bounding CPU, memory, output, and procs.

    Returns None where resource limits are unavailable so callers can
    pass the result straight to ``subprocess.run``. Limits are a
    backstop behind the caller's own timeout, not a replacement.
    """
    for name, value, minimum in (
        ("timeout_seconds", timeout_seconds, 0),
        ("address_gb", address_gb, 1),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < minimum
        ):
            raise ValueError("Sandbox limit needs %s" % name)
    if os.name != "posix":
        return None

    def _apply():
        import resource

        def _set(limit, *values):
            try:
                resource.setrlimit(limit, values)
            except (ValueError, OSError):
                pass

        headroom = int(timeout_seconds) + 120
        _set(resource.RLIMIT_CPU, (headroom, headroom + 300))
        address = address_gb * 2**30
        _set(resource.RLIMIT_AS, (address, address))
        _set(resource.RLIMIT_FSIZE, (2 * 2**30, 2 * 2**30 + 2**30))
        if hasattr(resource, "RLIMIT_NPROC"):
            _set(resource.RLIMIT_NPROC, (256, 512))

    return _apply
