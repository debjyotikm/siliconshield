"""Honest coverage simulation and triplet-integrity audits.

The simulator calls the trainer's real
:func:`grouped_epoch_indices` with per-rank carried queue state,
mirroring the epoch loop in ``security_train`` exactly, so the
simulation cannot drift from the sampler it predicts. Triplet
audits check that complete triplets sit in one pool and are
fully drawn within the horizon. Quota schedules allow bounded
per-round adjustments with validation.
"""

from collections import Counter, defaultdict


def triplet_integrity(rows, assignment, seen):
    """Check pool placement and draw closure of complete triplets.

    ``assignment`` maps row index to pool name; ``seen`` is the set
    of drawn row indices. Returns split triplets (members across
    pools), triplets with an undrawn member, and the triplet count.
    """
    triplets = defaultdict(list)
    for index, row in enumerate(rows):
        triplets[row["pair_id"]].append(index)
    split = sorted(
        pair
        for pair, members in triplets.items()
        if len({assignment[i] for i in members}) != 1
    )
    broken = sorted(
        pair
        for pair, members in triplets.items()
        if any(i not in seen for i in members)
    )
    return {"triplets": len(triplets), "split": split, "broken": broken}


def quotas_for_round(schedule, round_index):
    """Return validated per-pool quotas for a round index.

    ``schedule`` maps round indices (as strings or ints) to
    ``{pool: rows_per_update}`` maps. Later rounds inherit the
    latest earlier entry; unknown rounds before the first entry
    refuse. Quotas must be positive integers.
    """
    if not isinstance(schedule, dict) or not schedule:
        raise ValueError("Quota schedule must be a non-empty object")
    entries = {}
    for key, quotas in schedule.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            raise ValueError("Quota schedule keys must be round indices")
        if not isinstance(quotas, dict) or not quotas:
            raise ValueError("Quota entries must be non-empty objects")
        for pool, rows in quotas.items():
            if (
                not isinstance(pool, str)
                or not pool
                or type(rows) is not int
                or rows <= 0
            ):
                raise ValueError("Pool quotas must be positive integers")
        entries[index] = dict(quotas)
    earlier = [i for i in entries if i <= round_index]
    if not earlier:
        raise ValueError("No quota entry covers round %d" % round_index)
    return entries[max(earlier)]


def simulate_coverage(
    rows,
    pool_plan,
    seed,
    horizon_updates,
    world=3,
    global_batch=48,
    per_rank=16,
    grouped_epoch_indices=None,
):
    """Simulate sampler draws over a horizon, honestly.

    Calls the real ``grouped_epoch_indices`` (passed in to avoid a
    torch import here) once per epoch per rank with carried
    per-rank queue state, consuming per-rank shards in update
    order exactly like the trainer loop. Returns per-row draw
    counts, epochs consumed, and per-update pool quotas.
    """
    if grouped_epoch_indices is None:
        raise ValueError("A sampler callable is required")
    if horizon_updates <= 0:
        raise ValueError("Horizon must be positive")
    assignment = pool_plan[0]
    seen = Counter()
    states = [{} for _ in range(world)]
    quotas = []
    updates = 0
    epoch = 0
    while updates < horizon_updates:
        shards = [
            grouped_epoch_indices(
                rows,
                world,
                rank,
                seed,
                epoch,
                global_batch,
                pool_plan=pool_plan,
                _queue_state=states[rank],
            )
            for rank in range(world)
        ]
        if len({len(shard) for shard in shards}) != 1:
            raise ValueError("Unequal rank shards in simulation")
        for step in range(0, len(shards[0]), per_rank):
            chunk = [shard[step : step + per_rank] for shard in shards]
            if any(len(part) != per_rank for part in chunk):
                break
            quotas.append(
                Counter(assignment[i] for part in chunk for i in part)
            )
            for part in chunk:
                for index in part:
                    seen[index] += 1
            updates += 1
            if updates >= horizon_updates:
                break
        epoch += 1
        if epoch > horizon_updates + 2:
            raise ValueError("Horizon unreachable in simulation")
    return {
        "seen": dict(seen),
        "epochs": epoch,
        "quotas": quotas,
        "updates": updates,
    }


def reconstruct_seen(
    rows,
    pool_plan,
    seed,
    epoch,
    consumed,
    world=3,
    global_batch=48,
    grouped_epoch_indices=None,
):
    """Replay grouped draws exactly as the trainer loop consumed them.

    Replays complete epochs ``[0, epoch)`` fully plus the current
    epoch's consumed per-rank prefix, with carried per-rank queue
    state — the same deterministic replay the trainer uses to
    rebuild queues after resume. Every rank computes identical
    results with no communication. Returns per-row draw counts.
    """
    if grouped_epoch_indices is None:
        raise ValueError("A sampler callable is required")
    if epoch < 0 or consumed < 0:
        raise ValueError("Replay needs a nonnegative epoch and prefix")
    seen = Counter()
    states = [{} for _ in range(world)]
    for prior in list(range(epoch)) + [epoch]:
        shards = [
            grouped_epoch_indices(
                rows,
                world,
                rank,
                seed,
                prior,
                global_batch,
                pool_plan=pool_plan,
                _queue_state=states[rank],
            )
            for rank in range(world)
        ]
        if len({len(shard) for shard in shards}) != 1:
            raise ValueError("Unequal rank shards in replay")
        prefix = consumed if prior == epoch else len(shards[0])
        for shard in shards:
            for index in shard[:prefix]:
                seen[index] += 1
    return dict(seen)


def exposure_gate_passed(exposure: dict) -> bool:
    """Return whether delivered exposure covers a full first pass.

    Requires every row drawn at least once with no undrawn
    triplet member and no triplet split across pools.
    """
    return bool(
        exposure["unique_rows"] == exposure["total_rows"]
        and not exposure["broken_triplets"]
        and not exposure["split_triplets"]
    )


def delivered_exposure(seen, rows, assignment, checkpoint_step=None):
    """Summarize delivered exposure for selected and terminal steps.

    ``seen`` maps row index to draw counts from ``simulate_coverage``.
    Reports unique rows, full-triplet closure, and per-pool shares so
    an early selected checkpoint is never described as trained on
    rows only the terminal horizon reaches. ``checkpoint_step`` is
    recorded for receipts when the caller slices ``seen`` to that
    horizon.
    """
    total = len(rows)
    unique = len(seen)
    triplets = triplet_integrity(rows, assignment, set(seen))
    pools: dict[str, int] = {}
    for index, count in seen.items():
        pool = assignment.get(index, "unknown")
        pools[pool] = pools.get(pool, 0) + count
    draws = sum(seen.values())
    return {
        "total_rows": total,
        "unique_rows": unique,
        "draws": draws,
        "triplets": triplets["triplets"],
        "broken_triplets": triplets["broken"],
        "split_triplets": triplets["split"],
        "pool_draws": pools,
        "checkpoint_step": checkpoint_step,
    }
