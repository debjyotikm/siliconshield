"""A frozen teacher anchors replay without labeling new or evaluation rows."""

import pytest

from siliconshield.schemas import REPLAY_PRESERVATION_SCHEMA
from siliconshield.training.laya.replay_preservation import (
    preservation_diagnostics,
    preservation_loss,
    teacher_checkpoint,
    validate_preservation,
)


def config():
    return {
        "version": REPLAY_PRESERVATION_SCHEMA,
        "weight": 1.0,
        "temperature": 2.0,
        "pools": ["retained"],
    }


@pytest.mark.parametrize("invalid", [True, 0, -1, float("nan")])
def test_invalid_teacher_weight_refuses(invalid):
    value = config()
    value["weight"] = invalid
    with pytest.raises(ValueError, match="positive weight"):
        validate_preservation(value)


def test_teacher_configuration_is_optional_and_copied():
    assert validate_preservation(None) is None
    value = config()
    result = validate_preservation(value)
    result["pools"].append("additional")
    assert value["pools"] == ["retained"]


def test_consolidation_teacher_is_independent_of_student_initialization():
    value = config()
    assert teacher_checkpoint(value, "newly-fitted") == "newly-fitted"
    value["teacher_checkpoint"] = "previously-accepted"
    assert teacher_checkpoint(value, "newly-fitted") == "previously-accepted"


@pytest.mark.parametrize("invalid", [None, True, "", "  "])
def test_invalid_explicit_teacher_path_refuses(invalid):
    value = config()
    value["teacher_checkpoint"] = invalid
    with pytest.raises(ValueError, match="teacher checkpoint"):
        validate_preservation(value)


def test_only_correct_teacher_on_eligible_rows_has_gradient():
    torch = pytest.importorskip("torch")
    teacher = torch.tensor(
        [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        requires_grad=True,
    )
    student = torch.zeros(3, 3, requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    losses = preservation_loss(
        student, teacher, target, torch.tensor([True, False, True]), 2.0
    )
    assert losses[0] > 0
    assert losses[1:].tolist() == [0, 0]
    losses.sum().backward()
    assert teacher.grad is None
    assert student.grad[0, 0] < 0
    assert student.grad[1:].count_nonzero() == 0


def test_identical_logits_have_zero_loss_and_option_permutation_is_invariant():
    torch = pytest.importorskip("torch")
    teacher = torch.tensor([[4.0, 0.0, -2.0]])
    student = torch.tensor([[2.0, 1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.tensor([True])
    assert preservation_loss(teacher, teacher, target, mask, 2.0).item() == 0
    loss = preservation_loss(student, teacher, target, mask, 2.0)
    order = [2, 0, 1]
    changed = preservation_loss(
        student[:, order], teacher[:, order], target[:, order], mask, 2.0
    )
    torch.testing.assert_close(loss, changed)


def test_partial_accumulation_matches_a_single_weighted_batch():
    torch = pytest.importorskip("torch")
    student = torch.tensor([[1.0, 0.0, 0.0]] * 3, requires_grad=True)
    teacher = torch.tensor([[4.0, 0.0, 0.0]] * 3)
    target = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    mask = torch.ones(3, dtype=torch.bool)
    validity = torch.tensor([1.0, 1.0, 0.0])
    full = preservation_loss(student, teacher, target, mask, 2.0) * validity
    full.sum().div(2).backward()
    expected = student.grad.clone()
    student.grad = None
    for lo, hi in [(0, 2), (2, 3)]:
        part = preservation_loss(
            student[lo:hi], teacher[lo:hi], target[lo:hi], mask[lo:hi], 2.0
        )
        (part * validity[lo:hi]).sum().div(2).backward()
    torch.testing.assert_close(student.grad, expected)


def test_diagnostics_mirror_the_loss_mask():
    torch = pytest.importorskip("torch")
    student = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    teacher = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    target = torch.tensor([[1.0, 0.0, 0.0]] * 5)
    eligible = torch.tensor([True, True, True, False, True])
    valid = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])
    result = preservation_diagnostics(student, teacher, target, eligible, valid)
    assert result == {
        "pool_rows": 3,
        "teacher_match_rows": 2,
        "student_teacher_agree_rows": 1,
    }
    penalty = preservation_loss(student, teacher, target, eligible, 2.0)
    nonzero = {index for index, value in enumerate(penalty) if value > 0}
    assert nonzero == {0, 1, 4}
