"""Synthetic tests for Glimmer specialist measurement."""

import json

import pytest
from test_glimmer_specialists import (
    FINGERPRINT,
    _glimmer_registry,
)

from siliconshield.experts import measure


class _OracleChat:
    """Answer from an embedded marker; a synthetic test oracle."""

    def __init__(self):
        self.calls = []

    def complete(self, messages, tools=()):
        self.calls.append(messages)
        state = messages[-1]["content"]
        for label in (
            "satisfies_requirement",
            "violates_requirement",
            "insufficient_context",
        ):
            if "EXPECTED %s" % label in state:
                verdict = label
                break
        else:
            raise AssertionError("synthetic state lacks an EXPECTED marker")
        return {
            "content": (
                "Synthetic reasoning.\n\nVERDICT: %s\nCONFIDENCE: 0.9\n"
                "EVIDENCE: marker\nOPEN QUESTIONS: none\n" % verdict
            )
        }


def _dataset(tmp_path):
    rows = []
    for index, label in enumerate(
        [
            "satisfies_requirement",
            "violates_requirement",
            "insufficient_context",
        ]
    ):
        rows.append(
            {
                "id": "row-%d" % index,
                "state": "REQUIREMENT\nr\nASSUMPTIONS\na\nRTL\nEXPECTED %s\n"
                % label,
                "questions": ["security"],
                "expected": {"security": label},
                "benchmark_area": "counter",
                "benchmark_source": "synthetic-triplet",
                "pair_id": "trio-1",
                "mechanism_id": "counter-hold",
            }
        )
    path = tmp_path / "data.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    return str(path)


def _args(tmp_path, **overrides):
    parser = measure.build_parser()
    values = {
        "--registry": _glimmer_registry(tmp_path),
        "--endpoint": "http://127.0.0.1:1/v1/chat/completions",
        "--model": "synthetic-glimmer",
        "--glimmer-fingerprint": FINGERPRINT,
        "--dataset": _dataset(tmp_path),
        "--out-dir": str(tmp_path / "measured"),
    }
    values.update(overrides)
    argv = []
    for key, value in values.items():
        argv.extend([key, str(value)])
    return parser.parse_args(argv)


def test_measure_writes_valid_profile_and_receipt(tmp_path):
    args = _args(tmp_path, **{"--experts": "glimmer-a"})
    summary = measure.run_measure(args, client_factory=_OracleChat)
    assert summary["experts"]["glimmer-a"]["correct"] == 3
    profile = json.loads(
        (tmp_path / "measured" / "glimmer-a.profile.json").read_text()
    )
    assert profile["id"] == "glimmer-a"
    assert profile["checkpoint_sha256"] == FINGERPRINT
    assert profile["supported_mechanisms"] == ["counter"]
    assert profile["measurements"] == [
        {"eval": profile["measurements"][0]["eval"], "correct": 3, "total": 3}
    ]
    evidence = profile["routing_evidence"]
    assert evidence["areas"]["counter"] == {
        "correct": 3,
        "total": 3,
        "violations": 1,
        "false_safe": 0,
        "errors": 0,
    }
    assert evidence["execution_identity"]["checkpoint_sha256"] == FINGERPRINT
    receipt = json.loads((tmp_path / "measured" / "receipt.json").read_text())
    assert receipt["experts"]["glimmer-a"]["total"] == 3


def test_written_profile_loads_through_service(tmp_path):
    from siliconshield.experts.service import InferenceService

    args = _args(tmp_path, **{"--experts": "glimmer-a"})
    measure.run_measure(args, client_factory=_OracleChat)
    registry = json.loads((tmp_path / "models.json").read_text())
    for model in registry["models"]:
        if model["id"] == "glimmer-a":
            model["profile"] = str(
                tmp_path / "measured" / "glimmer-a.profile.json"
            )
            del model["qualification"]
    bound = tmp_path / "models.bound.json"
    bound.write_text(json.dumps(registry))
    service = InferenceService(str(bound))
    assert "glimmer-a" in service.expert_ids
    assert "glimmer-a" not in service.broken_experts


def test_limit_without_violations_refuses_profile(tmp_path):
    args = _args(tmp_path, **{"--experts": "glimmer-a", "--limit": "1"})
    with pytest.raises(ValueError, match="nothing ratable"):
        measure.run_measure(args, client_factory=_OracleChat)


def test_measure_refuses_unratable_evidence(tmp_path):
    class _Wrong:
        def complete(self, messages, tools=()):
            return {
                "content": (
                    "Synthetic reasoning.\n\nVERDICT: satisfies_requirement\n"
                    "CONFIDENCE: 0.9\nEVIDENCE: none\nOPEN QUESTIONS: none\n"
                )
            }

    args = _args(tmp_path, **{"--experts": "glimmer-a"})
    with pytest.raises(ValueError, match="nothing demonstrated"):
        measure.run_measure(args, client_factory=_Wrong)


def test_measure_refuses_bad_inputs(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    args = _args(first, **{"--experts": "nobody"})
    with pytest.raises(ValueError, match="unknown experts"):
        measure.run_measure(args, client_factory=_OracleChat)
    second = tmp_path / "second"
    second.mkdir()
    args = _args(second, **{"--glimmer-fingerprint": "cd" * 32})
    with pytest.raises(ValueError, match="pins weights"):
        measure.run_measure(args, client_factory=_OracleChat)


def test_measure_main_reports_errors_without_live_calls(tmp_path, capsys):
    assert (
        measure.main(
            [
                "--registry",
                str(tmp_path / "missing.json"),
                "--endpoint",
                "http://127.0.0.1:1/v1/chat/completions",
                "--model",
                "synthetic-glimmer",
                "--glimmer-fingerprint",
                FINGERPRINT,
                "--dataset",
                str(tmp_path / "missing.jsonl"),
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "measure" in capsys.readouterr().err
