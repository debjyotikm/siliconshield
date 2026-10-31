"""DDP coherence guard: single backward plus no_sync accumulation."""

from pathlib import Path

from siliconshield.training.laya.ddp_guard import (
    require_coherent_training,
    require_single_backward,
)


def test_trainer_uses_single_backward():
    source = (
        Path(__file__).resolve().parents[1]
        / "siliconshield/training/laya/security_train.py"
    )
    assert require_single_backward(source) == 1
    report = require_coherent_training(source)
    assert report["backward_calls"] == 1
    assert report["no_sync"] is True


def test_count_ignores_comments_and_strings(tmp_path):
    from siliconshield.training.laya.ddp_guard import count_backward_calls

    source = tmp_path / "fake_trainer.py"
    source.write_text(
        '"""Docstring mentioning loss.backward() twice: x.backward()."""\n'
        "# comment: y.backward() is not a call\n"
        "z = 'w.backward()'\n"
        "(loss * scale).backward()\n"
    )
    assert count_backward_calls(source) == 1
