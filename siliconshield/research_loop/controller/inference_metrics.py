"""Measure generated tokens without confusing prefill with decoding."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


class TokenBatch(Protocol):
    """The tensor operation needed by a Transformers generation streamer."""

    def numel(self) -> int:
        """Return the number of tokens in this batch."""
        ...


@dataclass
class TokenTiming:
    """Record single-request token arrivals, including speculative blocks.

    The first arrival can contain several tokens. Those tokens are excluded
    from the subsequent decode rate. Timing includes synchronization and
    transport when the caller measures arrivals outside the generation engine.
    """

    started: float
    first: float | None = None
    last: float | None = None
    first_tokens: int = 0
    generated_tokens: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.started):
            raise ValueError("Generation start time must be finite")

    def record(self, count: int, timestamp: float) -> None:
        """Record a nonempty block of generated tokens at a monotonic time."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Generated token count must be a positive integer")
        if not math.isfinite(timestamp) or timestamp < (
            self.last if self.last is not None else self.started
        ):
            raise ValueError("Token timestamps must be monotonic")
        if self.first is None:
            self.first = timestamp
            self.first_tokens = count
        self.last = timestamp
        self.generated_tokens += count

    def summary(self, finished: float) -> dict[str, int | float | None]:
        """Return seconds and token rates; undefined decode rates remain null."""
        if not math.isfinite(finished) or finished < (
            self.last if self.last is not None else self.started
        ):
            raise ValueError("Completion precedes the last generated token")
        elapsed = finished - self.started
        decode_seconds = (
            self.last - self.first
            if self.last is not None and self.first is not None
            else 0.0
        )
        return {
            "generated_tokens": self.generated_tokens,
            "first_token_seconds": (
                self.first - self.started if self.first is not None else None
            ),
            "generation_seconds": elapsed,
            "decode_seconds": decode_seconds,
            "decode_tokens": self.generated_tokens - self.first_tokens,
            "decode_tokens_per_second": (
                (self.generated_tokens - self.first_tokens) / decode_seconds
                if decode_seconds > 0
                else None
            ),
            "generation_tokens_per_second": (
                self.generated_tokens / elapsed if elapsed > 0 else None
            ),
        }


@dataclass
class GenerationStreamer:
    """Time Transformers output; its initial prompt callback is not output."""

    timing: TokenTiming
    clock: Callable[[], float] = time.perf_counter
    _prompt_seen: bool = field(default=False, init=False)

    def put(self, value: TokenBatch) -> None:
        """Skip the initial prompt, then record each generated token block."""
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        self.timing.record(value.numel(), self.clock())

    def end(self) -> None:
        """No text buffering is needed for timing."""
