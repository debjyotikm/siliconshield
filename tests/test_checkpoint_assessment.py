"""Joint checkpoint constraints and external task selection."""

import copy
import sys
from types import SimpleNamespace

import pytest

from siliconshield.research_loop.episodes.operator_assessment import (
    assessment,
    assessment_issues,
)
from siliconshield.training.glimmer.loading import staged_weights
from siliconshield.training.glimmer.train import (
    checkpoint_improved,
    selection_mode,
)


def test_staging_preserves_values_and_restores_hook_on_failure(monkeypatch):
    """The CUDA path clones before transfer and restores the loader on error."""
    calls = []

    class Tensor:
        device = SimpleNamespace(type="cpu")

        def __getitem__(self, _):
            return self

        def clone(self):
            calls.append("clone")
            return self

        def to(self, **kwargs):
            calls.append(kwargs)
            return self

    original = object()
    loader = SimpleNamespace(_materialize_copy=original)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            device=lambda device: SimpleNamespace(type=device.split(":")[0])
        ),
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(core_model_loading=loader)
    )
    tensor = Tensor()
    with pytest.raises(RuntimeError):
        with staged_weights(True):
            assert loader._materialize_copy(tensor, "cuda:0", "bf16") is tensor
            assert calls == ["clone", {"device": "cuda:0", "dtype": "bf16"}]
            raise RuntimeError("fixture load failed")
    assert loader._materialize_copy is original


def observation():
    """An earlier checkpoint meets both limits; the selected one does not."""
    return {
        "selected_checkpoint_id": "later",
        "checkpoint_constraints": [
            {"view": "old", "metric": "errors", "relation": "max", "limit": 2},
            {"view": "new", "metric": "correct", "relation": "min", "limit": 8},
        ],
        "checkpoint_candidates": [
            {
                "id": "earlier",
                "views": {"old": {"errors": 1}, "new": {"correct": 8}},
            },
            {
                "id": "later",
                "views": {"old": {"errors": 3}, "new": {"correct": 8}},
            },
        ],
    }


def test_joint_constraints_and_wrong_population():
    """A passing transfer population cannot hide a retention regression."""
    state = observation()
    result = assessment(state)
    review = result["checkpoint_review"]
    assert review["feasible_ids"] == ["earlier"]
    assert review["selected_feasible"] is False
    assert review["failed_constraints"]["later"] == ["old.errors"]
    forged = copy.deepcopy(result)
    forged["checkpoint_review"]["feasible_ids"] = ["later"]
    assert assessment_issues(state, forged)


def test_missing_evidence_is_unknown_and_identifier_changes_are_neutral():
    """Names never imply quality; incomplete candidates cannot pass."""
    state = observation()
    state["checkpoint_candidates"][0]["id"] = "candidate-b"
    state["checkpoint_candidates"][1]["views"].pop("old")
    review = assessment(state)["checkpoint_review"]
    assert review["feasible_ids"] == ["candidate-b"]
    assert review["unknown_constraints"]["later"] == ["old.errors"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_invalid_evidence_is_refused(bad):
    """Non-finite values and booleans cannot masquerade as measurements."""
    state = observation()
    state["checkpoint_candidates"][0]["views"]["old"]["errors"] = bad
    with pytest.raises(ValueError):
        assessment(state)


def test_old_observation_contract_is_unchanged():
    """Existing operator episodes need no checkpoint-review field."""
    assert "checkpoint_review" not in assessment({"phase": "training"})


def test_external_selection_never_promotes_lower_loss():
    """A loss improvement cannot substitute for the external replay result."""
    args = SimpleNamespace(external_task_selection=True, task_eval_every=0)
    assert selection_mode(args) == "external_pending"
    assert not checkpoint_improved("external_pending", None, None, 0.01, 0.5)
    assert not checkpoint_improved("in_process", [2], [3], 0.01, 0.5)
    assert checkpoint_improved("in_process", [4], [3], 0.6, 0.5)
    args.task_eval_every = 10
    with pytest.raises(ValueError):
        selection_mode(args)
