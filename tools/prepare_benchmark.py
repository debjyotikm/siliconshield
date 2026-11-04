"""Prepare the reviewed routing population from a pinned public benchmark."""

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import TypedDict

from siliconshield.benchmark.dataset import (
    LABELS,
    default_source,
    load_benchmark,
    validate_rows,
)

SOURCE_SHA256 = (
    "46d96ce09b203960a35d3e7790c1ae8ad2aa88c5cb410c3112a0b462ad6474f9"
)
REVIEWED_SHA256 = (
    "6b245e58570bcf51276ceeaee37134d1e0322a21c7d56f67a5a950baa5dc326c"
)
SOURCE_ROWS = 5589
REVIEWED_ROWS = 5580
EXCLUDED_PAIRS = tuple(f"47225d60c3a2f82dc0a15505:{i}" for i in range(3))
REASON = "Checked storage obligation is absent from the English requirement."


class PreparationReceipt(TypedDict):
    """Provenance for the reviewed benchmark and its excluded triplets."""

    source: str
    source_sha256: str
    source_rows: int
    reviewed_sha256: str
    reviewed_rows: int
    removed_rows: int
    excluded_pairs: list[str]
    reason: str
    role: str


def prepare(source: str, output: Path) -> PreparationReceipt:
    """Validate, filter and publish a new dataset directory with its receipt.

    The original source is never rewritten. Unknown source bytes, incomplete
    excluded triplets, a changed output identity, or an existing output refuse
    publication. Temporary files remain outside the destination until complete.

    Args:
        source: Pinned Hub source or path to an unchanged local copy.
        output: New directory for the reviewed dataset and provenance record.

    Returns:
        Source and output identities, exclusion reason, and population counts.

    Raises:
        FileExistsError: The destination already exists.
        ValueError: The source or reviewed population fails validation.
        OSError: The source cannot be read or output cannot be published.
    """
    if not isinstance(source, str) or not source:
        raise ValueError("Benchmark source must be a non-empty string")
    if not isinstance(output, Path):
        raise ValueError("Benchmark output must be a path")
    output = output.absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"Use a new output directory: {output}")
    original = load_benchmark(source)
    if (
        original["rows_sha256"] != SOURCE_SHA256
        or len(original["rows"]) != SOURCE_ROWS
    ):
        raise ValueError("Source differs from the reviewed benchmark revision")
    removed = [
        row for row in original["rows"] if row["pair_id"] in EXCLUDED_PAIRS
    ]
    for pair in EXCLUDED_PAIRS:
        labels = Counter(
            row["expected"]["security"]
            for row in removed
            if row["pair_id"] == pair
        )
        if labels != Counter(LABELS):
            raise ValueError("Exclusion must contain complete label triplets")
    kept = validate_rows(
        [
            row
            for row in original["rows"]
            if row["pair_id"] not in EXCLUDED_PAIRS
        ]
    )
    blob = (
        "\n".join(json.dumps(row, sort_keys=True) for row in kept) + "\n"
    ).encode()
    digest = hashlib.sha256(blob).hexdigest()
    if len(kept) != REVIEWED_ROWS or digest != REVIEWED_SHA256:
        raise ValueError("Filtered population differs from the reviewed result")
    receipt: PreparationReceipt = {
        "source": original["source"],
        "source_sha256": original["rows_sha256"],
        "source_rows": len(original["rows"]),
        "reviewed_sha256": digest,
        "reviewed_rows": len(kept),
        "removed_rows": len(removed),
        "excluded_pairs": list(EXCLUDED_PAIRS),
        "reason": REASON,
        "role": "reused development evidence, not independent qualification",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".reviewed-benchmark-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "benchmark-reviewed.jsonl").write_bytes(blob)
        (staging / "provenance.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        if os.path.lexists(output):
            raise FileExistsError(f"Use a new output directory: {output}")
        staging.rename(output)
    return receipt


def main() -> None:
    """Prepare the pinned benchmark without model inference or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=default_source(),
        help="Pinned Hub source or local copy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
