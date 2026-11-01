# Area benchmark

Scores decision models on the public [SiliconShield RTL
benchmark](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-benchmark)
(11 design areas, 5,589 rows, pinned revision) and reports per-area
capability: accuracy, per-class recall, false-safes, and
whole-triplet correctness.

## Run it

```
python -m siliconshield.benchmark areas
python -m siliconshield.benchmark run --registry models.json \
    --out-dir bench-out
```

`run` scores every registry entry (or `--models a,b`), one model at
a time, and writes `<model>.json` plus `comparison.json`. `--areas`
and `--limit` select subsets for smoke runs. The default dataset is
the pinned Hub revision. Pass a local path to `--dataset` for
offline or CPU-test use (no `huggingface_hub` needed then).

## Future models

Any registry entry is benchmarkable: checkpoint, interpreter,
runtime, device, and file manifest. No model-specific code paths
exist, so a new checkpoint qualifies by adding one entry. Reports
carry model id, checkpoint hash, and benchmark hash. `compare`
refuses reports from different benchmark revisions.

## Pinned inference path

Rows go through the production worker subprocess unbatched, one
request each, with the benchmark's own pinned prompt. The research
campaign's batch-16 scorer may differ on ~2% of boundary rows
(sat/vio or ctx/vio flips, verified on Contract-1/2 fifo+counter
slices). Reruns through this path are deterministic. Compare
reports produced by this CLI against each other, not against
differently-batched campaign scores, when single rows matter.
