"""Desktop GUI entry point: parse options before importing Qt.

This module never imports Qt, Torch, or CUDA at top level so that
``python -m siliconshield.gui --help`` works in lightweight
environments. Qt is imported only inside ``main`` after parsing.
"""

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    """Build the GUI command-line parser (Qt-free)."""
    parser = argparse.ArgumentParser(
        description="SiliconShield RTL security desktop analysis.",
    )
    parser.add_argument(
        "--models",
        help="Local registry file (JSON). Falls back to "
        "SILICONSHIELD_GUI_MODELS, then "
        "~/.config/siliconshield/models.local.json when it exists. "
        "When missing but --checkpoint/--python/--laya-root are given, "
        "the registry is created automatically on launch.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Write the registry and exit without opening the window "
        "(advanced; normally omit this and launch directly).",
    )
    parser.add_argument(
        "--checkpoint",
        help="Model export dir/file. Falls back to "
        "SILICONSHIELD_SECURITY_CHECKPOINT.",
    )
    parser.add_argument(
        "--python",
        help="LAYA environment python. Falls back to "
        "SILICONSHIELD_SECURITY_PYTHON.",
    )
    parser.add_argument(
        "--laya-root",
        help="Pinned LAYA runtime checkout. Falls back to LAYA_ROOT.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device: cpu or cuda:N (default: cuda:0). "
        "CPU is slower but needs no GPU. Overridable in the window.",
    )
    parser.add_argument(
        "--model-id", default="contract-1", help="Registry id to create."
    )
    parser.add_argument(
        "--display-name",
        default="SiliconShield-RTL Contract-1",
        help="Dropdown label to create.",
    )
    parser.add_argument(
        "--load-timeout",
        type=int,
        default=180,
        help="Seconds to wait for model load (default: 180).",
    )
    parser.add_argument(
        "--analyze-timeout",
        type=int,
        default=60,
        help="Seconds to wait for one analysis (default: 60).",
    )
    return parser


def resolve_models_path(
    flag_value: str | None,
) -> str | None:
    """Resolve the registry path without guessing model weights.

    Order: --models flag, SILICONSHIELD_GUI_MODELS, then the default
    user config file when it already exists. Returns None when no
    registry is configured; the caller may auto-create it from
    --checkpoint/--python/--laya-root.
    """
    if flag_value:
        return flag_value
    env_value = os.environ.get("SILICONSHIELD_GUI_MODELS")
    if env_value and env_value.strip():
        return env_value.strip()
    # Local import keeps this module Qt-free and avoids cycles.
    from siliconshield.gui.setup import default_registry_path

    default = default_registry_path()
    try:
        if default.is_file():
            return str(default)
    except OSError:
        return None
    return None


def has_location_args(args: object) -> bool:
    """Return True when checkpoint/python/laya-root are all available."""
    checkpoint = getattr(args, "checkpoint", None) or os.environ.get(
        "SILICONSHIELD_SECURITY_CHECKPOINT"
    )
    python = getattr(args, "python", None) or os.environ.get(
        "SILICONSHIELD_SECURITY_PYTHON"
    )
    laya_root = getattr(args, "laya_root", None) or os.environ.get("LAYA_ROOT")
    return bool(checkpoint and python and laya_root)


def ensure_registry_for_launch(args: object) -> str | None:
    """Return a registry path for launch, creating it when needed.

    Returns an existing file unchanged. When the resolved path is
    missing but checkpoint/python/laya-root are available, writes it
    and returns it. Returns None when neither is available.
    """
    from pathlib import Path

    existing = resolve_models_path(getattr(args, "models", None))
    if existing is not None:
        try:
            if Path(existing).is_file():
                return existing
        except OSError:
            return existing
        if not has_location_args(args):
            return existing
    elif not has_location_args(args):
        return None
    # Local import keeps --help usable without Qt installed.
    from siliconshield.gui.setup import init_from_args

    try:
        return str(init_from_args(args))
    except ValueError as exc:
        raise SystemExit("Cannot create registry: %s" % exc)


def main() -> None:
    """Parse options, then launch the Qt window (Qt import is local)."""
    args = build_parser().parse_args()
    if args.init:
        # Local import keeps --help usable without Qt installed.
        from siliconshield.gui.setup import init_from_args

        try:
            written = init_from_args(args)
        except ValueError as exc:
            raise SystemExit("Cannot create registry: %s" % exc)
        print("Wrote validated registry to %s" % written)
        print("Launch with: python -m siliconshield.gui")
        return
    models_path = ensure_registry_for_launch(args)
    # Local import keeps --help usable without Qt installed.
    from siliconshield.gui.application import run_window

    run_window(
        models_path=models_path,
        load_timeout_seconds=args.load_timeout,
        analyze_timeout_seconds=args.analyze_timeout,
    )


if __name__ == "__main__":
    main()
