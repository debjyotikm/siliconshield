"""Glimmer selection, serving, campaign, jobs, and operator tests: synthetic fixtures only. Torch- and harness2-gated suites stay in their own files so their module-level skips keep working, and the contracts module stays put because serving and harness2 import its helpers."""

import copy
import json
import re
from types import SimpleNamespace

from siliconshield.research_loop.episodes.contracts import (
    OPERATOR_ACTIONS,
)
from siliconshield.research_loop.episodes.operator_episodes import (
    SCENARIOS,
    episode_for,
)
from siliconshield.research_loop.execution.glimmer_operator import (
    run_cycle,
)
from siliconshield.research_loop.execution.operator import Operator
from siliconshield.training.glimmer.selection import (
    METRICS,
)


# ---- Merged from test_glimmer_selection.py ----
def report() -> dict:
    """Create a small, complete authored operator population."""
    rows = [
        {
            "episode_id": action + str(i),
            "group_id": action,
            "expected_action": action,
            "checkpoint_review": action == "leave_unchanged",
            "execution_complete": True,
            "execution_error": None,
            **dict.fromkeys(METRICS, i == 0),
        }
        for action in OPERATOR_ACTIONS
        for i in range(2)
    ]
    result = {
        "n": len(rows),
        "results": rows,
        "valid_execution": True,
        "test_evaluated": False,
        "source_identity": "source",
        "evaluator_sha256": "evaluator",
        "dataset_manifest_sha256": "dataset",
        "decoding": {"temperature": 0},
    }
    recount(result)
    return result


def recount(value: dict) -> None:
    """Recompute fixture totals after deliberate outcome changes."""
    value["totals"] = {
        key: sum(r[key] for r in value["results"]) for key in METRICS
    }


def outcome(value: dict, identity: str, passed: bool) -> None:
    """Change one fixture episode, preserving accurate summary counts."""
    next(r for r in value["results"] if r["episode_id"] == identity).update(
        dict.fromkeys(METRICS, passed)
    )
    recount(value)


# ---- Merged from test_glimmer_serving.py ----


# ---- Merged from test_glimmer_campaign.py ----
def arguments(root):
    return SimpleNamespace(
        root=str(root),
        run="fixture",
        training_container="owned-training",
        evaluation_container="owned-evaluation",
        image="fixture-image",
        examples=10,
    )


# ---- Merged from test_glimmer_jobs.py ----
PROFILE = {
    "python": "/env/bin/python",
    "source": "/source",
    "model": "/model",
    "data": "/data",
    "runs": "/runs",
    "defaults": {"max_updates": 100, "seed": 41},
}


# ---- Merged from test_glimmer_operator.py ----
ARMS = [
    {
        "name": "headonly",
        "overrides": {"freeze_encoder": True, "disable_adapters": True},
    },
    {
        "name": "adapters",
        "overrides": {"freeze_encoder": True, "adapter_lr": 1e-4},
    },
    {
        "name": "full",
        "overrides": {"freeze_encoder": False, "adapter_lr": 1e-4},
    },
]


class FakeJobs:
    def __init__(self, root, profile):
        self.profile, self.root = profile, root
        self.submissions = 0

    def status(self, experiment_id):
        return {}

    def submit(self, experiment_id, overrides, launch=False):
        self.submissions += 1
        return {"phase": "submitted"}


def make_operator(tmp_path):
    profile = {
        "python": "/usr/bin/python3",
        "source": "/source",
        "model": "/m",
        "data": "/d",
        "runs": str(tmp_path / "runs"),
        "gpus": [0, 1, 2],
        "defaults": {"max_updates": 10},
    }
    op = Operator.__new__(Operator)
    op.jobs = FakeJobs(tmp_path / "runs", profile)
    op.profile = profile
    op.runner = lambda argv: {"LoadState": "not-found"}
    from siliconshield.research_loop.execution.operator import (
        ResourceReservation,
    )

    op.reservation = ResourceReservation(op.jobs.root)
    return op


class ScriptedGlimmer:
    """Replays one authored operator trace for the requested action."""

    model = "fixture-glimmer"
    endpoint = "http://fixture.invalid"

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def complete(self, messages, tools=()):
        self.calls += 1
        scenario = next(s for s in SCENARIOS if s[3] == self.action)
        row = episode_for(
            scenario[0],
            scenario[1],
            scenario[2],
            scenario[3],
            scenario[4],
            "fx",
            0,
        )
        message = copy.deepcopy(row["messages"][len(messages)])
        run_id = re.search(r"Inspect run (\S+)", messages[1]["content"])[1]
        for call in message.get("tool_calls", []):
            call["function"]["arguments"]["run_id"] = run_id
        if message.get("role") == "assistant" and message.get("content"):
            final = json.loads(message["content"])
            final["run_id"] = run_id
            message["content"] = json.dumps(final)
        return message


def test_valid_decision_executes_through_operator(tmp_path):
    op = make_operator(tmp_path)
    manifest = op.prepare("exp-a", {})
    manifest["phase"] = "submit"  # state-machine position after a real submit
    manifest, executed = run_cycle(
        op, manifest, ScriptedGlimmer("leave_unchanged"), tmp_path
    )
    assert executed["action"] == "leave_unchanged"


class RecordingOperator:
    """Records intervene params without executing, to verify field propagation."""

    def __init__(self):
        self.interventions = []

        class _Jobs:
            def status(self, experiment_id):
                return {}

        class _Reservation:
            def owner(self):
                return None

        self.jobs, self.reservation = _Jobs(), _Reservation()

    def observe(self, manifest):
        return manifest

    def diagnose(self, manifest):
        return manifest

    def intervene(self, manifest, action, **params):
        self.interventions.append({"action": action, "params": params})
        manifest.setdefault("interventions", []).append(
            {"action": action, "params": params}
        )
        return manifest


def stub_manifest():
    return {
        "experiment_id": "exp-a",
        "phase": "submit",
        "history": [],
        "interventions": [],
    }


def _schedule_campaign(
    tmp_path, fitting_common, development_overrides, **extra
):
    from siliconshield.research_loop.execution.strategy import run_campaign

    def execute(config):
        folder = tmp_path / config["experiment_id"]
        folder.mkdir(exist_ok=True)
        if config["overrides"].get("limit_train_rows"):
            report = {
                "evaluation_role": "train_fitting",
                "train_accuracy": 0.8,
                "initial_train_accuracy": 0.3,
                "train": {"nll": 0.4},
                "selected_checkpoint": {"step": 6},
            }
        else:
            report = {
                "development": {"accuracy": 0.7},
                "development_acceptance": {"accepted": False},
            }
        (folder / "report.json").write_text(json.dumps(report))
        return {"manifest": {"terminal": "complete", "output": str(folder)}}

    config = {
        "experiment_prefix": "fixture",
        "output": str(tmp_path / "operator"),
        "profile": "unused",
        "timeout_seconds": 30,
        "operator": {"model": "fixture"},
        "fitting_common": fitting_common,
        "arms": ARMS,
        "development_overrides": development_overrides,
    }
    config.update(extra)
    return run_campaign(config, run=execute)


def test_model_status_rejects_malformed_records():
    import pytest

    from siliconshield.research_loop.execution import glimmer_operator as gop

    with pytest.raises(ValueError, match="must be an object"):
        gop.model_status(["x"])
    bad = {"evaluation_role": "development", "selected_checkpoint": "s7"}
    with pytest.raises(ValueError, match="checkpoint reference"):
        gop.model_status({"report.json": bad})
    bad = dict(
        bad,
        selected_checkpoint={"step": 1},
        development_acceptance={"accepted": True, "reasons": 5},
    )
    with pytest.raises(ValueError, match="reasons are malformed"):
        gop.model_status({"report.json": bad})
    bad["development_acceptance"] = {"accepted": True, "reasons": "ok"}
    with pytest.raises(ValueError, match="reasons are malformed"):
        gop.model_status({"report.json": bad})


def test_history_helpers_tolerate_bad_lines_reject_bad_shapes(tmp_path):
    import pytest

    from siliconshield.research_loop.execution import glimmer_operator as gop

    events = tmp_path / "events.jsonl"
    events.write_text(
        '[1, 2]\n{"event": "development", "step": 1, "metrics": {}}\n'
    )
    manifest = {"output": str(tmp_path)}
    assert gop.development_history(manifest)["total"] == 1
    assert gop._latest_view_metrics(manifest) == (None, None)
    with pytest.raises(ValueError, match="must be an object"):
        gop.development_history(None)
    with pytest.raises(ValueError, match="must be an object"):
        gop._latest_view_metrics(["x"])
    with pytest.raises(ValueError, match="needs a manifest"):
        gop.schedule_status(["x"], {})
    with pytest.raises(ValueError, match="needs a manifest"):
        gop.schedule_status({}, ["x"])
    with pytest.raises(ValueError, match="must be an object"):
        gop.stop_evidence({}, ["x"])


def test_dispatch_rejects_malformed_tool_calls(tmp_path):
    import pytest

    from siliconshield.research_loop.execution import glimmer_operator as gop
    from siliconshield.research_loop.execution.operator import (
        ResourceReservation,
    )

    class Jobs:
        def __init__(self, status):
            self._status = status

        def status(self, run_id):
            return self._status

    class FakeOperator:
        def __init__(self, status):
            self.jobs = Jobs(status)
            self.reservation = ResourceReservation(tmp_path)

    manifest = {"experiment_id": "e1", "output": str(tmp_path)}
    tools = gop.OperatorTools(FakeOperator({}), manifest)
    with pytest.raises(ValueError, match="must be an object"):
        tools.dispatch("operator.decide", ["x"])
    bad = gop.OperatorTools(
        FakeOperator({"status.json": ["x"]}), dict(manifest)
    )
    with pytest.raises(ValueError, match="records are malformed"):
        bad.dispatch("operator.inspect_run", {"run_id": "e1"})
    bad = gop.OperatorTools(
        FakeOperator({"execution.json": ["x"]}), dict(manifest)
    )
    with pytest.raises(ValueError, match="records are malformed"):
        bad.dispatch("operator.inspect_run", {"run_id": "e1"})


def test_glimmer_helpers_reject_malformed_inputs(tmp_path):
    import pytest

    from siliconshield.training.glimmer import dpo_pairs, identity, selection

    assert dpo_pairs.validate_pair(5) == ["pair schema incomplete"]
    with pytest.raises(ValueError, match="must be a path"):
        dpo_pairs.write_pairs([], 5)
    with pytest.raises(ValueError, match="list of pairs"):
        dpo_pairs.write_pairs(5, tmp_path / "p.jsonl")
    target = tmp_path / "p.jsonl"
    with pytest.raises(ValueError, match="Pair .* invalid"):
        dpo_pairs.write_pairs([{"id": "bad"}], target)
    assert not target.exists()
    with pytest.raises(ValueError, match="report object"):
        selection.compare_replay(5, {})
    with pytest.raises(ValueError, match="At least one reference"):
        selection.select_replay_candidate(5, {})
    with pytest.raises(ValueError, match="candidate reports"):
        selection.select_replay_candidate({"r": {}}, 5)
    with pytest.raises(ValueError, match="must be a path"):
        identity.adapter_manifest(5)
    with pytest.raises(ValueError, match="must be a path"):
        identity.snapshot_manifest(5)
    with pytest.raises(ValueError, match="must be a path"):
        identity.checkpoint_fingerprint(5)


def test_migrate_operator_row_rejects_malformed_rows():
    import pytest

    from siliconshield.research_loop.episodes import operator_episodes

    with pytest.raises(ValueError, match="episode object"):
        operator_episodes.migrate_operator_row(5)
    with pytest.raises(ValueError, match="message objects"):
        operator_episodes.migrate_operator_row({"messages": [5]})
    with pytest.raises(ValueError, match="tool calls"):
        operator_episodes.migrate_operator_row(
            {"messages": [{"tool_calls": [5]}]}
        )
    migrated, note = operator_episodes.migrate_operator_row(
        {
            "messages": [
                {"tool_calls": [{"function": {"name": "operator.decide"}}]}
            ]
        }
    )
    assert migrated is None
    assert note["status"] == "rejected"


def test_operator_builders_reject_malformed_inputs():
    import pytest

    from siliconshield.research_loop.episodes import (
        operator_curriculum,
        operator_dev,
        operator_episodes,
    )

    with pytest.raises(ValueError, match="for train"):
        operator_episodes.operator_coverage(5, [], [])
    with pytest.raises(ValueError, match="needs text lineage"):
        operator_episodes.episode_for("s", {}, {}, "a", "r", 5, 0)
    with pytest.raises(ValueError, match="integer index"):
        operator_episodes.episode_for("s", {}, {}, "a", "r", "l", "x")
    with pytest.raises(ValueError, match="spec object"):
        operator_curriculum.episode(5)
    with pytest.raises(ValueError, match="missing split"):
        operator_curriculum.episode(
            {
                "state": {},
                "resources": {},
                "scenario": "s",
                "action": "stop",
                "reason": "r",
                "family": "f",
                "index": 0,
            }
        )
    with pytest.raises(ValueError, match="episode object"):
        operator_curriculum.fixture_verify(5)
    with pytest.raises(ValueError, match="catalog and output paths"):
        operator_curriculum.build(5, "/tmp/x")
    with pytest.raises(ValueError, match="output path"):
        operator_dev.build_dev(5)
    with pytest.raises(ValueError, match="must be a path"):
        operator_dev.load_dev(5)


def test_training_row_loaders_reject_malformed_inputs():
    import pytest

    from siliconshield.training.glimmer import tokenization, train

    with pytest.raises(ValueError, match="must be a path"):
        train.load_rows(5)
    with pytest.raises(ValueError, match="must be a path"):
        train.training_rows(5)
    with pytest.raises(ValueError, match="episode rows"):
        train.dev_subset(5, 1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        train.dev_subset([], -1)
    with pytest.raises(ValueError, match="integer max length"):
        tokenization.encode_episode(None, {}, "x")
