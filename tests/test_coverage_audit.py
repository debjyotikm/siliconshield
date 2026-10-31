"""Tests for honest coverage simulation and triplet audits."""

from collections import Counter

from siliconshield.training.laya.coverage_audit import (
    quotas_for_round,
    simulate_coverage,
    triplet_integrity,
)


def make_rows():
    rows = []
    for pair in range(4):
        for member, label in enumerate(
            (
                "satisfies_requirement",
                "violates_requirement",
                "insufficient_context",
            )
        ):
            rows.append(
                {
                    "id": "r%d-%d" % (pair, member),
                    "pair_id": "p%d" % pair,
                    "expected": {"security": label},
                }
            )
    return rows


def test_triplet_split_and_broken():
    rows = make_rows()
    assignment = {i: ("a" if i % 3 else "b") for i in range(len(rows))}
    report = triplet_integrity(rows, assignment, set(range(len(rows))))
    assert report["triplets"] == 4
    assert report["split"] == ["p0", "p1", "p2", "p3"]
    assert report["broken"] == []
    clean = {i: "a" for i in range(len(rows))}
    report = triplet_integrity(rows, clean, set(range(len(rows) - 1)))
    assert report["split"] == []
    assert report["broken"] == ["p3"]


def test_quota_schedule_inherits_and_validates():
    schedule = {"1": {"new": 24, "replay": 24}, "3": {"new": 36, "replay": 12}}
    assert quotas_for_round(schedule, 1) == {"new": 24, "replay": 24}
    assert quotas_for_round(schedule, 2) == {"new": 24, "replay": 24}
    assert quotas_for_round(schedule, 4) == {"new": 36, "replay": 12}
    for bad in ({}, {"1": {}}, {"1": {"new": 0}}, {"x": {"new": 1}}):
        try:
            quotas_for_round(bad, 1)
        except ValueError:
            continue
        raise AssertionError("accepted bad schedule: %r" % (bad,))
    try:
        quotas_for_round(schedule, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("round before first entry accepted")


def test_simulation_mirrors_sampler_calls():
    rows = make_rows()
    assignment = {i: ("new" if i < 6 else "replay") for i in range(len(rows))}
    calls = []

    def fake_sampler(
        rows_arg,
        world,
        rank,
        seed,
        epoch,
        batch,
        pool_plan=None,
        _queue_state=None,
    ):
        calls.append((rank, epoch, seed, batch, _queue_state is not None))
        _queue_state["epochs"] = _queue_state.get("epochs", 0) + 1
        base = rank * 4
        return [base + (epoch + k) % 4 for k in range(4)]

    report = simulate_coverage(
        rows,
        (
            assignment,
            {"new": 0.5, "replay": 0.5},
            {"new": "flat", "replay": "flat"},
        ),
        77,
        4,
        world=3,
        global_batch=6,
        per_rank=2,
        grouped_epoch_indices=fake_sampler,
    )
    assert report["updates"] == 4
    assert report["epochs"] == 2
    assert sum(report["seen"].values()) == 24
    # Per-rank carried state: each rank saw distinct epoch sequence.
    by_rank = Counter(rank for rank, _, _, _, _ in calls)
    assert by_rank == {0: 2, 1: 2, 2: 2}, by_rank
    assert all(carried for _, _, _, _, carried in calls)
    assert all(sum(q.values()) == 6 for q in report["quotas"])


if __name__ == "__main__":
    test_triplet_split_and_broken()
    test_quota_schedule_inherits_and_validates()
    test_simulation_mirrors_sampler_calls()
    print("coverage_audit tests: 3 passed")
