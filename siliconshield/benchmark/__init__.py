"""Area benchmark for RTL security decision models (Qt-free).

Scores any registry model on the public SiliconShield RTL benchmark
(11 design areas, balanced three-way labels, whole triplets) and
reports per-area capability: accuracy, per-class recall,
false-safes, and whole-triplet correctness. Model-neutral by
construction: future checkpoints qualify by adding a registry entry,
never by changing this package. The dataset revision is pinned;
pass an explicit revision to reproduce or to move forward.
"""

from siliconshield.benchmark import cli, dataset, metrics, runner

__all__ = ["cli", "dataset", "metrics", "runner"]
