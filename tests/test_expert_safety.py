"""Synthetic regression tests for reviewed router boundary failures."""

import copy
import json

from test_multi_expert import (
    _STATE,
    _assessment,
    _identity,
    _rated,
    _write_registry,
)

from siliconshield.experts import arbitration
from siliconshield.experts.router import route_scope


def _two_models(tmp_path, second_tags=None):
    path = _write_registry(
        tmp_path, _rated("expert-a", ["fifo"], {"fifo": 90}), {}
    )
    raw = json.loads(open(path).read())
    second = copy.deepcopy(raw["models"][0])
    second.update(id="expert-b", profile="profile-b.json")
    profile = _rated("expert-b", second_tags or ["fifo"], {"fifo": 50})
    from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA

    for filename, item in [
        ("profile-b.json", profile),
        ("profile.json", _rated("expert-a", ["fifo"], {"fifo": 90})),
    ]:
        (tmp_path / filename).write_text(
            json.dumps({**item, "schema": EXPERT_CAPABILITY_SCHEMA})
        )
    raw["models"].append(second)
    (tmp_path / "models.json").write_text(json.dumps(raw))
    return str(tmp_path / "models.json")


def test_arbitration_uses_current_counts_not_historical_profile():
    assessment = _assessment([{"expert_id": "expert-a"}])
    profile = _rated("expert-a", ["fifo"], {"fifo": 90})
    assessment["routing"] = route_scope(
        ["fifo"], [profile], {"expert-a": _identity()}
    )
    bound = arbitration.build_arbitration_input(
        assessment, _STATE, "synthetic requirement", {"expert-a": profile}
    )
    shown = bound["profiles"]["expert-a"]
    assert "measurements" not in shown
    assert shown["routing_evidence"]["areas"]["fifo"]["correct"] == 90
    assert shown["measurement_status"] == "matched to this execution"
    profile["routing_evidence"]["execution_identity"]["runtime_sha256"] = (
        "12" * 32
    )
    assessment["routing"] = route_scope(
        ["fifo"], [profile], {"expert-a": _identity()}
    )
    bound = arbitration.build_arbitration_input(
        assessment, _STATE, "synthetic requirement", {"expert-a": profile}
    )
    assert "routing_evidence" not in bound["profiles"]["expert-a"]
    assert (
        bound["profiles"]["expert-a"]["measurement_status"]
        == "unrated for this execution"
    )
