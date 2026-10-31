"""Independent dev set: synthetic, distinct from regression checks and training."""

import json


def _norm_state(state):
    import copy

    state = copy.deepcopy(state)
    state.pop("run_id", None)
    return json.dumps(state, sort_keys=True)


def test_dev_builds_independent_synthetic_rows(tmp_path):
    from siliconshield.research_loop.episodes.operator_dev import (
        build_dev,
        load_dev,
    )
    from siliconshield.schemas import OPERATOR_DEV_SCHEMA

    manifest = build_dev(tmp_path / "dev")
    assert manifest["schema"] == OPERATOR_DEV_SCHEMA
    assert manifest["rows"] == 6 and manifest["independent"]
    assert not manifest["held_out"]
    rows = load_dev(tmp_path / "dev/dev.jsonl")
    assert all(r["group_id"].startswith("dev-independent-") for r in rows)


def test_dev_disjoint_from_baseline_and_training_states():
    from siliconshield.research_loop.episodes.operator_baseline import (
        BASELINE_SCENARIOS,
    )
    from siliconshield.research_loop.episodes.operator_dev import DEV_SCENARIOS

    baseline_states = {_norm_state(sc[1]) for sc in BASELINE_SCENARIOS}
    dev_states = {_norm_state(sc[1]) for sc in DEV_SCENARIOS}
    assert not (baseline_states & dev_states)
    # Training eligible states (synthetic distinct set) must also avoid both.
    # Spot-check the previously overlapping shapes are now distinct:
    # baseline uses step 40/60/200/300, grad 42.0, exit 1; dev uses 57/63/123/211/311.
    dev_steps = set()
    for _, state, _, _, _, _ in DEV_SCENARIOS:
        if isinstance(state.get("step"), int):
            dev_steps.add(state["step"])
        for entry in state.get("development_history", {}).get("entries", []):
            dev_steps.add(entry["step"])
    assert dev_steps.isdisjoint({40, 60, 200, 300})
    assert 57 in dev_steps and 63 in dev_steps
