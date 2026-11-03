"""Launch the optional loopback browser demo."""

import argparse
from pathlib import Path


def main() -> None:
    """Parse options before importing workers or starting any background task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="Local JSON settings, outside Git"
    )
    parser.add_argument("--port", type=int, help="Loopback port override")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate settings without starting services",
    )
    args = parser.parse_args()
    from siliconshield.web_demo.config import load_config

    try:
        config = load_config(args.config, port=args.port)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.check:
        print(
            f"Configuration valid: {len(config['entries'])} registered models; training enabled: {config['training_enabled']}"
        )
        return
    from siliconshield.web_demo.server import serve

    serve(config)


if __name__ == "__main__":
    main()
