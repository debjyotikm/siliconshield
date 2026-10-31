"""Large-file scan: deterministic windows, validation, and counting.

Qt-free helpers for the scan tab. Muse selects relevant windows (see
``muse_scan``); the decision model judges each excerpt through the
already-loaded worker; counting here is deterministic Python, never an
LLM tally. Exceeding the model input budget fails closed per excerpt.
"""

import re
from pathlib import Path
from typing import Any

SCAN_MAX_BYTES = 2 * 1024 * 1024
SCAN_WINDOW_LINES = 30
SCAN_OVERLAP_LINES = 6
SCAN_MAX_EXCERPTS = 20
SCAN_MAX_EXCERPT_CHARS = 4000
SCAN_SUPPORTED_SUFFIXES = (".v", ".sv")

_MODULE_START = re.compile(
    r"^\s*module\s+(?:automatic\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\b"
)
_MODULE_END = re.compile(r"^\s*endmodule\b")


def read_scan_source(path_str: str) -> tuple[str, dict[str, Any]]:
    """Read one large RTL file with an explicit bound for scanning.

    Accepts up to SCAN_MAX_BYTES of UTF-8 ``.v``/``.sv`` text. Raises
    ValueError with an actionable message otherwise; callers must
    surface the failure instead of silently truncating the design.
    """
    import hashlib

    if not path_str.strip():
        raise ValueError("Choose a large design file to scan.")
    path = Path(path_str).expanduser()
    if not path.exists():
        raise ValueError("Design file does not exist: %s" % path)
    if not path.is_file():
        raise ValueError("Design path is not a file: %s" % path)
    if path.suffix.lower() not in SCAN_SUPPORTED_SUFFIXES:
        raise ValueError(
            "Unsupported design suffix %r; use .v or .sv." % path.suffix
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError("Cannot read design file: %s" % exc)
    if len(raw) == 0:
        raise ValueError("Design file is empty: %s" % path)
    if len(raw) > SCAN_MAX_BYTES:
        raise ValueError(
            "Design file exceeds %d bytes; split it first." % SCAN_MAX_BYTES
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Design file is not UTF-8: %s" % exc)
    if not text.strip():
        raise ValueError("Design file holds no Verilog text.")
    return text, {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def split_scan_windows(
    text: str,
    window_lines: int = SCAN_WINDOW_LINES,
    overlap_lines: int = SCAN_OVERLAP_LINES,
    max_chars: int = SCAN_MAX_EXCERPT_CHARS,
) -> list[dict[str, Any]]:
    """Split source into overlapping line windows (no truncation of file).

    Each window is bounded so one excerpt plus requirement fits the
    decision-model budget; overlong windows carry an explicit marker
    and are still judged (coverage rejection stays authoritative).
    """
    if window_lines <= 0:
        raise ValueError("window_lines must be positive")
    if overlap_lines < 0 or overlap_lines >= window_lines:
        raise ValueError("overlap_lines must be in [0, window_lines)")
    lines = text.splitlines()
    windows: list[dict[str, Any]] = []
    step = window_lines - overlap_lines
    for start in range(0, len(lines), step):
        chunk = lines[start : start + window_lines]
        body = "\n".join(chunk) + "\n"
        truncated = False
        if len(body) > max_chars:
            body = body[:max_chars] + "\n// ...[window truncated]...\n"
            truncated = True
        windows.append(
            {
                "index": len(windows),
                "start_line": start + 1,
                "end_line": min(len(lines), start + window_lines),
                "text": body,
                "truncated": truncated,
            }
        )
        if start + window_lines >= len(lines):
            break
    if not windows:
        raise ValueError("No windows: source holds no lines.")
    return windows


def _mask_comments(text: str) -> str:
    """Replace comments with spaces while preserving lines and strings."""
    output: list[str] = []
    index = 0
    in_block = False
    in_string = False
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_block:
            if char == "*" and following == "/":
                output.extend((" ", " "))
                index += 2
                in_block = False
            else:
                output.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if in_string:
            output.append(char)
            if char == "\\" and following:
                output.append(following)
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == "/" and following == "*":
            output.extend((" ", " "))
            index += 2
            in_block = True
            continue
        if char == "/" and following == "/":
            output.extend((" ", " "))
            index += 2
            while index < len(text) and text[index] != "\n":
                output.append(" ")
                index += 1
            continue
        output.append(char)
        if char == '"':
            in_string = True
        index += 1
    return "".join(output)


def _complete_module_spans(text: str) -> list[tuple[int, int, str]]:
    """Return complete module spans or an empty list for unsafe syntax."""
    code_lines = _mask_comments(text).splitlines()
    spans: list[tuple[int, int, str]] = []
    open_module: tuple[int, str] | None = None
    for line_number, line in enumerate(code_lines):
        start = _MODULE_START.match(line)
        end = _MODULE_END.match(line)
        if start:
            if open_module is not None:
                return []
            open_module = (line_number, start.group(1))
        if end:
            if open_module is None:
                return []
            start_line, name = open_module
            spans.append((start_line, line_number, name))
            open_module = None
    if open_module is not None:
        return []
    return spans


def split_scan_excerpts(
    text: str,
    window_lines: int = SCAN_WINDOW_LINES,
    overlap_lines: int = SCAN_OVERLAP_LINES,
    max_chars: int = SCAN_MAX_EXCERPT_CHARS,
) -> list[dict[str, Any]]:
    """Prefer complete modules and fall back to bounded line windows.

    A complete module gives one security decision a coherent RTL scope.
    If module boundaries are absent, malformed, or a module exceeds the
    character budget, bounded overlapping windows retain coverage and
    carry explicit scope metadata.
    """
    if not text.strip():
        raise ValueError("No excerpts: source holds no lines.")
    lines = text.splitlines(keepends=True)
    spans = _complete_module_spans(text)
    if not spans:
        windows = split_scan_windows(
            text,
            window_lines=window_lines,
            overlap_lines=overlap_lines,
            max_chars=max_chars,
        )
        for window in windows:
            window.update(
                {
                    "kind": "line_window",
                    "module": None,
                    "complete": False,
                }
            )
        return windows

    excerpts: list[dict[str, Any]] = []
    for start, end, name in spans:
        body = "".join(lines[start : end + 1])
        if len(body) <= max_chars:
            excerpts.append(
                {
                    "index": len(excerpts),
                    "start_line": start + 1,
                    "end_line": end + 1,
                    "text": body,
                    "truncated": False,
                    "kind": "module",
                    "module": name,
                    "complete": True,
                }
            )
            continue
        windows = split_scan_windows(
            body,
            window_lines=window_lines,
            overlap_lines=overlap_lines,
            max_chars=max_chars,
        )
        for window in windows:
            window["index"] = len(excerpts)
            window["start_line"] += start
            window["end_line"] += start
            window.update(
                {
                    "kind": "module_window",
                    "module": name,
                    "complete": False,
                }
            )
            excerpts.append(window)
    return excerpts


def describe_excerpt_scope(excerpt: dict[str, Any]) -> str:
    """Describe whether an excerpt is a complete module or a fallback."""
    module = excerpt.get("module")
    if excerpt.get("complete") is True and isinstance(module, str):
        return "complete module %s" % module
    if isinstance(module, str):
        return "incomplete line window in module %s" % module
    return "incomplete line window"


def validate_scan_selection(
    excerpts: object, n_windows: int
) -> list[dict[str, Any]]:
    """Validate a Muse excerpt selection strictly.

    Returns sorted unique entries ``{index, reason}``. Raises ValueError
    on empty, non-integer, out-of-range, duplicated, or unexplained
    entries so a malformed selector can never silently drop windows.
    """
    if not isinstance(excerpts, list) or not excerpts:
        raise ValueError("Excerpt selection is empty.")
    seen: set[int] = set()
    cleaned: list[dict[str, Any]] = []
    for item in excerpts:
        if not isinstance(item, dict):
            raise ValueError("Excerpt must be an object.")
        index, reason = item.get("index"), item.get("reason")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Excerpt index must be an integer.")
        if index < 0 or index >= n_windows or index in seen:
            raise ValueError(
                "Excerpt index out of range or duplicate: %r." % (index,)
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Excerpt needs a non-empty reason.")
        seen.add(index)
        cleaned.append({"index": index, "reason": reason.strip()})
    return sorted(cleaned, key=lambda entry: entry["index"])


class ScanQueue:
    """Deterministic queue of excerpt indices with violation counting."""

    def __init__(self, indices: list[int]) -> None:
        """Bind the ordered excerpt indices to judge."""
        self._pending = list(indices)
        self._done: list[int] = []
        self._violations = 0

    @property
    def remaining(self) -> int:
        """Number of excerpts still waiting for a verdict."""
        return len(self._pending)

    @property
    def judged(self) -> int:
        """Number of excerpts with a recorded verdict."""
        return len(self._done)

    @property
    def violations(self) -> int:
        """Number of recorded `violates_requirement` verdicts."""
        return self._violations

    def advance(self, label: str) -> int | None:
        """Record one verdict; return the next index or None when done."""
        if not self._pending:
            return None
        finished = self._pending.pop(0)
        self._done.append(finished)
        if label == "violates_requirement":
            self._violations += 1
        if not self._pending:
            return None
        return self._pending[0]


def count_summary(queue: ScanQueue, errors: int) -> str:
    """Render the deterministic tally line.

    Counts model-flagged excerpts only; no independent verification
    was run, so the wording never claims confirmed findings.
    """
    return (
        "Potentially flagged: %d of %d excerpts "
        "(%d errors; model verdicts only, not independently verified)."
        % (
            queue.violations,
            queue.judged,
            errors,
        )
    )
