"""Validate complete triplets and reconstruct deterministic training exposure."""

import collections
import json
import random
from pathlib import Path


def load_rows(path):
    """Read JSONL with id, domain, pair_id, split_group, state and label."""
    rows = [
        json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()
    ]
    if not rows:
        raise ValueError("Dataset is empty")
    ids = set()
    groups = collections.defaultdict(list)
    for row in rows:
        for key in ("id", "domain", "pair_id", "split_group", "state"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing or invalid {key}")
        if type(row.get("label")) is not int or row["label"] not in (0, 1, 2):
            raise ValueError("Label must be integer 0, 1 or 2")
        if row["id"] in ids:
            raise ValueError("Duplicate row id")
        ids.add(row["id"])
        groups[(row["domain"], row["pair_id"])].append(row)
    for group in groups.values():
        if len(group) != 3 or {r["label"] for r in group} != {0, 1, 2}:
            raise ValueError("Every pair_id must contain a complete triplet")
        if len({r["split_group"] for r in group}) != 1:
            raise ValueError("Triplet spans multiple lineage groups")
    return rows


def validate_splits(train, dev):
    """Reject overlap in identifiers, exact input text or coarse lineage."""
    for key in ("id", "state", "split_group"):
        if {r[key] for r in train} & {r[key] for r in dev}:
            raise ValueError(f"TRAIN/DEV overlap in {key}")
    if {r["domain"] for r in train} != {r["domain"] for r in dev}:
        raise ValueError("TRAIN/DEV domain sets differ")
    if {(r["domain"], r["pair_id"]) for r in train} & {
        (r["domain"], r["pair_id"]) for r in dev
    }:
        raise ValueError("TRAIN/DEV triplet overlap")


def epoch_rows(rows, seed, epoch, quotas):
    """Shuffle whole triplets, interleaving domains without replacement."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["domain"], row["pair_id"])].append(row)
    domains = sorted({r["domain"] for r in rows})
    if set(quotas) != set(domains) or any(
        type(n) is not int or n <= 0 for n in quotas.values()
    ):
        raise ValueError("Specify a positive triplet quota for every domain")
    pools = {d: sorted(k for k in groups if k[0] == d) for d in domains}
    rng = random.Random(seed + epoch)
    for pool in pools.values():
        rng.shuffle(pool)
    ordered = []
    while any(pools.values()):
        for domain in domains:
            for _ in range(quotas[domain]):
                if pools[domain]:
                    ordered.extend(groups[pools[domain].pop()])
    return ordered
