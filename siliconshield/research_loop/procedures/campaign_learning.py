"""Copy completed development observations into a separate procedure queue."""

import hashlib
import json
from pathlib import Path
from typing import Any

from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    digest,
    episode_from_summary,
    read_json,
    validate_episode,
)
from siliconshield.research_loop.episodes.laya_results import adapt_report
from siliconshield.schemas import PROCEDURE_REQUEST_SCHEMA


def queue_report(queue: Path, report_path: Path) -> Path:
    """Snapshot a completed development report; never expose the source tree.

    This is a retrospective checkpoint-review task. It is not a reconstructed
    decision at an earlier time, nor a label about intervention effectiveness.
    """
    raw = report_path.read_bytes()
    summary, provenance, group = adapt_report(
        json.loads(raw), source_sha256=hashlib.sha256(raw).hexdigest()
    )
    row = episode_from_summary(summary, digest(group), provenance)
    validate_episode(row)
    body: dict[str, Any] = {"schema": PROCEDURE_REQUEST_SCHEMA, "episode": row}
    identifier = digest(body)
    path = queue / (identifier + ".json")
    if path.exists():
        load_request(path)
    else:
        atomic_json(path, {**body, "digest": identifier})
    return path


def load_request(path: Path) -> dict[str, Any]:
    """Refuse damaged requests and observations outside development scope."""
    request = read_json(path)
    if not isinstance(request, dict):
        raise ValueError("Procedure request must be an object")
    body = {key: value for key, value in request.items() if key != "digest"}
    if (
        body.get("schema") != PROCEDURE_REQUEST_SCHEMA
        or request.get("digest") != digest(body)
        or path.stem != request["digest"]
    ):
        raise ValueError("Procedure request identity mismatch")
    row = body["episode"]
    validate_episode(row)
    if row["summary"].get("split_role") != "open_development":
        raise ValueError("Only development procedure learning is supported")
    return dict(row)
