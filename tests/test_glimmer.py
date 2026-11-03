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
