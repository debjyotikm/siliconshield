"""Synthetic regression tests for reviewed router boundary failures."""

import copy
import json
import queue
import threading
import time

import pytest
from test_multi_expert import (
    _STATE,
    _arb_profiles,
    _assessment,
    _identity,
    _profile,
    _rated,
    _result,
    _write_cli_registry,
    _write_registry,
)

from siliconshield.app.inference_worker import _run_analyze
from siliconshield.experts import arbitration, contracts, inspection
from siliconshield.experts.cli import build_parser, run_assess
from siliconshield.experts.router import route_scope
from siliconshield.experts.service import InferenceService
from siliconshield.experts.worker import WorkerPool


@pytest.mark.parametrize(
    "replacement",
    [
        _STATE.replace("assign q = d;", "assign q = 0;"),
        _STATE.replace("RTL\n", "ASSUMPTIONS\nReset ignored.\nRTL\n"),
    ],
)
def test_arbitration_refuses_other_input(replacement):
    with pytest.raises(ValueError):
        arbitration.build_arbitration_input(
            _assessment([{"expert_id": "expert-a"}]),
            replacement,
            "synthetic requirement",
            _arb_profiles(),
        )


@pytest.mark.parametrize(
    "requirement",
    ["synthetic", "requirement", "synthetic requirement plus reset"],
)
def test_arbitration_requires_full_section(requirement):
    with pytest.raises(ValueError, match="full judged"):
        arbitration.build_arbitration_input(
            _assessment([{"expert_id": "expert-a"}]),
            _STATE,
            requirement,
            _arb_profiles(),
        )


def test_arbitration_binds_assumptions_and_response_request():
    assessment = _assessment([{"expert_id": "expert-a"}])
    with pytest.raises(ValueError, match="Assumptions"):
        arbitration.build_arbitration_input(
            assessment,
            _STATE,
            "synthetic requirement",
            _arb_profiles(),
            assumptions="Reset completed",
        )
    assessment["responses"][0]["assessment_id"] = "other-request"
    with pytest.raises(ValueError, match="belong"):
        arbitration.build_arbitration_input(
            assessment, _STATE, "synthetic requirement", _arb_profiles()
        )


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


def test_failed_strong_expert_cannot_supply_coverage(tmp_path):
    service = InferenceService(_two_models(tmp_path))

    def transport(expert_id, state):
        if expert_id == "expert-a":
            raise RuntimeError("synthetic failure")
        return _result(state=state, model_id=expert_id)

    assessment = service.assess(["fifo"], _STATE, transport)
    assert assessment["planned_routing"]["thin_evidence"] is False
    assert assessment["routing"]["thin_evidence"] is True
    assert assessment["routing"]["experts"] == ["expert-b"]
    assert len(assessment["errors"]) == 1
    verdict = arbitration.arbitrate_assessment(
        assessment,
        _STATE,
        "synthetic requirement",
        service.profiles_for(["expert-b"]),
        client_factory=lambda: pytest.fail("Must not call Muse"),
    )
    assert verdict["conclusion"] == "unresolved"


def test_all_failed_means_no_coverage(tmp_path):
    service = InferenceService(_two_models(tmp_path))

    def transport(*args):
        raise RuntimeError("truncated")

    assessment = service.assess(["fifo"], _STATE, transport)
    assert assessment["routing"]["experts"] == []
    assert assessment["routing"]["thin_evidence"]
    assert assessment["responses"] == []
    assert len(assessment["errors"]) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "other"),
        ("checkpoint_sha256", "ef" * 32),
        ("question_contract", "other"),
        ("release", "other"),
    ],
)
def test_override_profiles_cannot_bypass_identity(tmp_path, field, value):
    path = _write_registry(tmp_path, _profile("expert-a", ["fifo"]), {})
    profile = _profile("expert-a", ["fifo"])
    profile[field] = value
    with pytest.raises(ValueError):
        InferenceService(path, profiles={"expert-a": profile})


@pytest.mark.parametrize(
    "second_tags,expected_loads",
    [(["counter"], ["expert-a"]), (["fifo"], ["expert-a", "expert-b"])],
)
def test_cli_lazy_load_and_failure_isolation(
    tmp_path, second_tags, expected_loads
):
    path = _two_models(tmp_path, second_tags)
    (tmp_path / "state.txt").write_text(_STATE)
    loaded = []

    class Pool:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def load(self, entry):
            loaded.append(entry["id"])
            if entry["id"] == "expert-b":
                raise RuntimeError("Cannot load expert-b")

        def transport(self):
            return lambda eid, state: _result(state=state, model_id=eid)

    args = build_parser().parse_args(
        [
            "assess",
            "--registry",
            path,
            "--scope",
            "fifo",
            "--state-file",
            str(tmp_path / "state.txt"),
        ]
    )
    report = run_assess(args, pool_factory=Pool)["assessment"]
    assert loaded == expected_loads
    assert len(report["responses"]) == 1
    assert len(report["errors"]) == (len(expected_loads) - 1)


@pytest.mark.parametrize("kind", ["progress", "invalid", "stale"])
def test_deadline_not_extended_by_noise(kind):
    pool = WorkerPool()
    box = queue.Queue()
    pool._queues["fixture"] = box
    stopped = threading.Event()

    def pump():
        while not stopped.wait(0.005):
            line = json.dumps(
                dict(
                    type="progress",
                    request_id="old" if kind == "stale" else "request",
                    protocol=contracts.PROTOCOL_VERSION,
                    session_id="session",
                )
            )
            box.put("invalid json" if kind == "invalid" else line)

    thread = threading.Thread(target=pump)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            pool._wait("fixture", "request", ("result",), 0.06, "analyze")
        assert 0.04 <= time.monotonic() - started < 0.3
    finally:
        stopped.set()
        thread.join()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "foreign"),
        ("device", "cuda:7"),
        ("question_contract", "other"),
        ("protocol", -1),
        ("session_id", ""),
    ],
)
def test_invalid_ready_reaps_child(tmp_path, field, value):
    from pathlib import Path

    from siliconshield.experts.registry import load_registry

    path = _write_cli_registry(tmp_path)
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        worker.read_text().replace(
            "    sys.stdout.write",
            "    reply[%r] = %r\n    sys.stdout.write" % (field, value),
        )
    )
    pool = WorkerPool(
        worker_module="fake_worker",
        extra_env={
            "PYTHONPATH": ":".join(
                (
                    str(tmp_path),
                    str(Path(__file__).parent),
                    str(Path(__file__).parents[1]),
                )
            )
        },
    )
    with pool:
        with pytest.raises(RuntimeError):
            pool.load(load_registry(path)[0])
        assert pool._workers == {}
        assert pool._ready == {}


@pytest.mark.parametrize(
    "key", ["question_sha256", "runtime_sha256", "export_sha256"]
)
def test_changed_execution_unrates_profile(key):
    profile = _rated("expert-a", ["fifo"], {"fifo": 95})
    identity = _identity()
    assert not route_scope(["fifo"], [profile], {"expert-a": identity})[
        "thin_evidence"
    ]
    identity[key] = "01" * 32
    assert route_scope(["fifo"], [profile], {"expert-a": identity})[
        "tiers"
    ] == {"expert-a": "unrated"}


def test_inference_protocol_change_unrates_profile():
    profile = _rated("expert-a", ["fifo"], {"fifo": 95})
    identity = _identity()
    identity["inference"]["batch_size"] = 16
    assert route_scope(["fifo"], [profile], {"expert-a": identity})[
        "thin_evidence"
    ]


def test_inspection_ignores_comments_and_initializer_symbols():
    rtl = """module top(input logic clk, a, output logic q);
logic gated = omitted_policy;
// if (secret) q <= bypass;
/* if (debug) q = hidden; */
always_ff @(posedge clk) if (a && (gated || q)) q <= a;
endmodule"""
    result = inspection.inspect_structure(rtl, [])
    assert result["unknown_symbols"] == ["omitted_policy"]
    assert "omitted_policy" not in result["state"]
    assert result["writers"]["q"] == [5]
    assert result["guards"] == ["a && (gated || q)"]
    assert not result["complete"]


def test_indexed_write_and_comparison_are_distinct():
    result = inspection.inspect_structure(
        "if (count <= limit) mem[index] <= value;",
        ["count", "limit", "mem", "index", "value"],
    )
    assert result["writers"] == {"mem": [1]}


@pytest.mark.parametrize(
    "dropped", [-1, float("nan"), float("inf"), 0.0, True, None]
)
def test_coverage_counts_are_nonnegative_integers(dropped):
    item = _result()
    item["coverage"]["state_tokens_dropped"] = dropped
    with pytest.raises(ValueError):
        contracts.validate_result(item)


def test_conflicting_runtime_usage_refused():
    item = _result()
    item["usage"] = {"state_tokens_dropped": 3}
    with pytest.raises(ValueError, match="contradicts"):
        contracts.validate_result(item)


@pytest.mark.parametrize("usage", [None, {}, {"truncated": False}])
def test_worker_does_not_fabricate_missing_coverage(usage):
    class Agent:
        device = "cpu"

        def predict_batch(self, *args, **kwargs):
            return [
                {
                    "answers": {
                        "security": {
                            "probabilities": _result()["probabilities"]
                        }
                    },
                    "usage": usage,
                }
            ]

    with pytest.raises(ValueError, match="complete coverage"):
        _run_analyze(Agent(), _STATE, {})


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
