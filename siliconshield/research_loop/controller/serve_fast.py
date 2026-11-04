"""Serve Glimmer with native vLLM kernels and explicit adapter identities."""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from siliconshield.research_loop.execution.gpu_lease import acquire_gpu
from siliconshield.training.glimmer.identity import (
    MODEL_REVISION,
    checkpoint_fingerprint,
    verify_snapshot,
)


@dataclass(frozen=True)
class ServerConfig:
    """Local BF16 inference settings; speculative decoding is opt-in."""

    model: str
    adapters: tuple[tuple[str, str], ...] = ()
    draft: str | None = None
    speculative_tokens: int = 15
    port: int = 18320
    max_context: int = 8192
    max_sequences: int = 8
    memory_fraction: float = 0.65
    max_batched_tokens: int = 8192

    def __post_init__(self) -> None:
        """Reject malformed server settings before any GPU work."""
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("Server needs a model name")
        for field in (
            "port",
            "max_context",
            "max_sequences",
            "speculative_tokens",
            "max_batched_tokens",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("Server %s must be an integer" % field)
        if isinstance(self.memory_fraction, bool) or not isinstance(
            self.memory_fraction, (int, float)
        ):
            raise ValueError("Server memory fraction must be a number")
        if self.draft is not None and not isinstance(self.draft, str):
            raise ValueError("Server draft model must be a string")
        if not isinstance(self.adapters, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
            for item in self.adapters
        ):
            raise ValueError("Server adapters must be (name, path) pairs")

    def command(self) -> list[str]:
        """Build an argv list, without shell evaluation or model mutation."""
        if not math.isfinite(self.memory_fraction) or not (
            0 < self.memory_fraction < 1
        ):
            raise ValueError("Memory fraction must be between zero and one")
        if not 0 < self.port < 65536:
            raise ValueError("Port must be between 1 and 65535")
        if (
            min(
                self.max_context,
                self.max_sequences,
                self.speculative_tokens,
                self.max_batched_tokens,
            )
            < 1
        ):
            raise ValueError(
                "Context, sequence, and speculative limits must be positive"
            )
        if self.draft and self.max_batched_tokens < self.max_sequences * (
            1 + self.speculative_tokens
        ):
            raise ValueError(
                "Batched-token budget must accommodate every draft position"
            )
        names = [name for name, _ in self.adapters]
        if len(names) != len(set(names)) or "glimmer-base" in names:
            raise ValueError(
                "Adapter names must be unique and differ from glimmer-base"
            )
        if any(
            not name or "=" in name or not path for name, path in self.adapters
        ):
            raise ValueError("Adapters must have nonempty names and paths")
        command = [
            "vllm",
            "serve",
            self.model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served-model-name",
            "glimmer-base",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(self.max_context),
            "--max-num-seqs",
            str(self.max_sequences),
            "--max-num-batched-tokens",
            str(self.max_batched_tokens),
            "--gpu-memory-utilization",
            str(self.memory_fraction),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "muse_glimmer",
            "--reasoning-parser",
            "muse_glimmer",
            "--generation-config",
            "auto",
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
            "--safetensors-load-strategy",
            "lazy",
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_capture_sizes": sorted(
                        {1, self.max_sequences}
                        | {
                            size
                            for size in (2, 4, 8, 16, 32, 64, 128, 256)
                            if size <= self.max_sequences
                        }
                    )
                }
            ),
        ]
        if self.adapters:
            command += [
                "--enable-lora",
                "--max-lora-rank",
                "16",
                "--max-loras",
                str(len(self.adapters)),
                "--max-cpu-loras",
                str(len(self.adapters)),
                "--lora-modules",
                *[f"{name}={path}" for name, path in self.adapters],
            ]
        if self.draft:
            command += [
                "--speculative-config",
                json.dumps(
                    {
                        "model": self.draft,
                        "method": "dflash",
                        "num_speculative_tokens": self.speculative_tokens,
                    }
                ),
            ]
        return command


def main() -> None:
    """Validate local artifacts and hold the shared GPU lease while serving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--adapter", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--draft", help="Optional local DFlash assistant checkpoint"
    )
    parser.add_argument("--speculative-tokens", type=int, default=15)
    parser.add_argument("--port", type=int, default=18320)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--max-sequences", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=0.65)
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    adapters = []
    for value in args.adapter:
        name, separator, path = value.partition("=")
        if not separator:
            parser.error("Each --adapter must be NAME=PATH")
        adapters.append((name, path))
    config = ServerConfig(
        args.model,
        tuple(adapters),
        args.draft,
        args.speculative_tokens,
        args.port,
        args.max_context,
        args.max_sequences,
        args.memory_fraction,
        args.max_batched_tokens,
    )
    command = config.command()
    if args.dry_run:
        print(json.dumps({"command": command}))
        return
    if not (Path(config.model) / "config.json").is_file():
        raise ValueError("--model must identify a local checkpoint directory")
    snapshot = verify_snapshot(
        config.model,
        os.environ.get("SILICONSHIELD_MODEL_FINGERPRINT"),
        revision=MODEL_REVISION,
    )
    fingerprints = {}
    for name, path in config.adapters:
        adapter_config = json.loads(
            (Path(path) / "adapter_config.json").read_text()
        )
        if not isinstance(adapter_config.get("r"), int) or not (
            1 <= adapter_config["r"] <= 16
        ):
            raise ValueError(
                "This server profile supports LoRA ranks 1 through 16"
            )
        fingerprints[name] = checkpoint_fingerprint(
            path, snapshot["fingerprint"]
        )
    draft = (
        verify_snapshot(
            config.draft, os.environ.get("SILICONSHIELD_DRAFT_FINGERPRINT")
        )
        if config.draft
        else None
    )
    # Replace the launcher so its lease and service lifetime match the server.
    # An ordinary child process could outlive a terminated launcher.
    with acquire_gpu() as lease:
        print(
            json.dumps(
                {
                    "phase": "starting_native_inference",
                    "base": snapshot["fingerprint"],
                    "adapters": fingerprints,
                    "draft": draft,
                    "command": command,
                }
            ),
            flush=True,
        )
        os.set_inheritable(lease.fileno(), True)
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
