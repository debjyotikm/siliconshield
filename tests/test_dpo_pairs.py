"""DPO pairs: same action, supported vs unsupported explanations; DPO loss math."""

import json

import pytest

from siliconshield.training.glimmer.dpo_pairs import (
    corrected_stop_pair,
    same_action_pair,
    training_pairs,
    validate_pair,
    write_pairs,
)


def _supported_continue_pair():
    state = {
        "run_id": "run-x",
        "phase": "training",
        "systemd": "running",
        "step": 40,
        "execution_failed": False,
    }
    resources = {"run_id": "run-x"}
    return same_action_pair(
        "run-x",
        state,
        resources,
        "continue",
        "Run is healthy and progressing; leaving it unchanged avoids an unnecessary intervention.",
        "This execution failure is a loading problem. Retry the job from scratch.",
        "test-pair-continue",
        "fixture",
    )


def test_corrected_stop_pair_validates_with_exact_wording():
    pair = corrected_stop_pair()
    assert validate_pair(pair) == []
    assert (
        "No execution failure was observed."
        in pair["chosen"]["decide"]["reason"]
    )
    assert "82.31%" in pair["chosen"]["decide"]["reason"]
    assert "memoriz" not in json.dumps(pair).lower()
    assert pair["provenance"]["kind"] == "authored_correction"


def test_same_action_pair_isolates_explanation():
    pair = _supported_continue_pair()
    assert validate_pair(pair) == []
    assert (
        pair["chosen"]["decide"]["action"]
        == pair["rejected"]["decide"]["action"]
        == "continue"
    )


def test_rejected_without_contradiction_is_refused(tmp_path):
    pair = _supported_continue_pair()
    pair["id"] = "bad-pair"
    # Make the rejected reason supported too: no contradiction remains.
    pair["rejected"]["decide"]["reason"] = pair["chosen"]["decide"]["reason"]
    with pytest.raises(ValueError, match="no explanation contradiction"):
        write_pairs([pair], tmp_path / "pairs.jsonl")


def test_write_pairs_roundtrip_and_hash(tmp_path):
    result = write_pairs(
        [corrected_stop_pair(), _supported_continue_pair()],
        tmp_path / "pairs.jsonl",
    )
    assert result["pairs"] == 2 and len(result["sha256"]) == 64
    assert len((tmp_path / "pairs.jsonl").read_text().splitlines()) == 2


def test_load_pairs_filters_regression_only_entry_point(tmp_path):
    from siliconshield.training.glimmer.dpo import load_pairs

    eligible = _supported_continue_pair()
    reg = corrected_stop_pair()
    assert reg["provenance"].get("regression_only") is True
    path = tmp_path / "pairs.jsonl"
    write_pairs([eligible, reg], path)
    # Shared loading path used by reference and training filters regression-only.
    loaded = load_pairs(str(path))
    assert [p["id"] for p in loaded] == [eligible["id"]]
    assert load_pairs(str(path), include_regression_only=True)[0]["id"] in {
        eligible["id"],
        reg["id"],
    }
    # Only regression-only file has no eligible pairs.
    only = tmp_path / "only.jsonl"
    write_pairs([reg], only)
    with pytest.raises(ValueError, match="No eligible"):
        load_pairs(str(only))


def test_pair_content_hash_detects_edits():
    import copy

    from siliconshield.training.glimmer.dpo_pairs import pair_content_hash

    pair = _supported_continue_pair()
    edited = copy.deepcopy(pair)
    edited["chosen"]["decide"]["reason"] += " Extra."
    assert pair_content_hash(pair) != pair_content_hash(edited)


def test_tokenizer_fingerprint_distinguishes_template():
    from siliconshield.training.glimmer.dpo import tokenizer_fingerprint

    class _Tok:
        def __init__(self, tmpl):
            self.chat_template = tmpl

        def __len__(self):
            return 100

    assert tokenizer_fingerprint(_Tok("a"), 2048) != tokenizer_fingerprint(
        _Tok("b"), 2048
    )
    assert tokenizer_fingerprint(_Tok("a"), 2048) != tokenizer_fingerprint(
        _Tok("a"), 1024
    )


def test_reference_requires_identity_and_gpu_lock(monkeypatch):
    from siliconshield.training.glimmer import dpo as dpo_mod

    monkeypatch.delenv("SILICONSHIELD_EXPECTED_CODE_IDENTITY", raising=False)
    monkeypatch.delenv("SILICONSHIELD_GPU_LOCK", raising=False)
    args = type(
        "A",
        (),
        {
            "model": "/nonexistent",
            "init_adapter": "/nonexistent",
            "pairs": "/nonexistent",
            "output": "/tmp/x",
            "seed": 41,
            "max_length": 2048,
        },
    )()
    with pytest.raises(
        ValueError, match="SILICONSHIELD_EXPECTED_CODE_IDENTITY"
    ):
        dpo_mod.reference_dpo(args)


def test_resume_requires_existing_checkpoint(tmp_path):
    from siliconshield.training.glimmer.dpo import check_resume_state

    latest = tmp_path / "latest"
    with pytest.raises(ValueError, match="requires an existing checkpoint"):
        check_resume_state(type("A", (), {"resume": True})(), latest)
    latest.mkdir()
    with pytest.raises(ValueError, match="requires --resume"):
        check_resume_state(type("A", (), {"resume": False})(), latest)


def test_dpo_loss_prefers_supported_response():
    torch = pytest.importorskip("torch")
    from siliconshield.training.glimmer.dpo import dpo_loss

    beta = 0.1
    good = dpo_loss(
        torch.tensor(-2.0),
        torch.tensor(-5.0),
        torch.tensor(-2.5),
        torch.tensor(-4.5),
        beta,
    )
    bad = dpo_loss(
        torch.tensor(-5.0),
        torch.tensor(-2.0),
        torch.tensor(-2.5),
        torch.tensor(-4.5),
        beta,
    )
    assert good.item() < bad.item()
    assert good.item() > 0
    # Larger beta sharpens the same margin.
    assert (
        dpo_loss(
            torch.tensor(-2.0),
            torch.tensor(-5.0),
            torch.tensor(-2.5),
            torch.tensor(-4.5),
            1.0,
        ).item()
        < good.item()
    )


def test_transfer_pairs_validate_and_exclude_overlap():
    from siliconshield.research_loop.episodes.transfer_episodes import SCENARIOS

    pairs = []
    for index, (name, state, resources, action, reason) in enumerate(
        SCENARIOS[:2]
    ):
        unsupported = (
            "All acceptance checks passed; continue."
            if action == "continue"
            else "The worker crashed, so stop training."
        )
        pair = same_action_pair(
            "example-transfer",
            dict(state, run_id="example-transfer"),
            resources,
            action,
            reason,
            unsupported,
            "example-pair-%d" % index,
            "Authored synthetic transfer example",
        )
        assert validate_pair(pair) == []
        pairs.append(pair)
    assert training_pairs(pairs + [corrected_stop_pair()]) == pairs
