"""GUI product qualification fixtures (no model weights)."""

import pytest

from siliconshield.app.inference_worker import questions_for_staged
from siliconshield.gui.qualification import (
    REQUIRED_CASES,
    build_case_states,
    qualify_states,
)
from siliconshield.training.laya.question_contracts import (
    LEGACY_CONTRACT,
    NEW_CONTRACT,
    match_version,
    write_export_record,
)


def test_required_cases_present_and_covered():
    states = build_case_states()
    assert set(states) == set(REQUIRED_CASES)
    report = qualify_states(states, contract=NEW_CONTRACT)
    assert report["contract"] == NEW_CONTRACT
    assert len(report["cases"]) == len(REQUIRED_CASES)


def test_fixtures_carry_distinct_functional_bodies():
    states = build_case_states()
    assert len(set(states.values())) == len(REQUIRED_CASES)
    bodies = {
        name: state.split("RTL\n", 1)[1] for name, state in states.items()
    }
    assert all(
        "always_ff" in body or "function" in body for body in bodies.values()
    )
    # Intentional same-RTL opposing-policy pair; all others differ.
    assert bodies["reset-exempt"] == bodies["reset-inclusive"]
    rest = [body for name, body in bodies.items() if name != "reset-inclusive"]
    assert len(set(rest)) == len(REQUIRED_CASES) - 1


def test_staged_questions_follow_export_record(tmp_path):
    legacy_questions, legacy_version = questions_for_staged(
        tmp_path / "missing"
    )
    assert legacy_version == LEGACY_CONTRACT
    assert match_version(legacy_questions) == LEGACY_CONTRACT
    export = tmp_path / "export"
    export.mkdir()
    write_export_record(export, NEW_CONTRACT)
    questions, version = questions_for_staged(export)
    assert version == NEW_CONTRACT
    assert match_version(questions) == NEW_CONTRACT


def test_staged_single_file_reads_parent_record(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    write_export_record(export, NEW_CONTRACT)
    weights = export / "model.pt"
    weights.write_bytes(b"fake-weights")
    _, version = questions_for_staged(weights)
    assert version == NEW_CONTRACT


def test_staged_corrupt_record_raises(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    (export / "question_contract.json").write_text("{corrupt")
    with pytest.raises(ValueError):
        questions_for_staged(export)
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    write_export_record(tampered, NEW_CONTRACT)
    record = tampered / "question_contract.json"
    record.write_text(record.read_text().replace("v1", "v0"))
    with pytest.raises(ValueError):
        questions_for_staged(tampered)
