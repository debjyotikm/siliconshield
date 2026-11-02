# Benchmark models and routing

[Documentation index](README.md) · [Evaluation](evaluation.md) ·
[Decision-model qualification](expert-qualification.md)

The benchmark package scores compatible registry models through the production
worker protocol. It reports aggregate, per-area, and per-mechanism decisions,
including false-safes, false violations, abstentions, class recall, NLL, and
whole-triplet consistency. Model execution is separate from metric computation.

## Population and intended use

The default [RTL benchmark](https://huggingface.co/datasets/debjyotikm/siliconshield-rtl-benchmark)
uses the dataset revision pinned by the benchmark loader:
5,589 decisions in 1,863 triplets across 11 areas. The pin is owned by
[`dataset.py`](../siliconshield/benchmark/dataset.py).

| Area | Decisions | Design behavior exercised |
|---|---:|---|
| `local-safety-cwe` | 2,610 | CWE-provenance local update and hold contracts on registers |
| `requirement-clarity` | 756 | Policy-wording stress across lock, privilege, role, and address requirements |
| `transfer-pairs` | 621 | Implementation variants of authorization, window, and width checks |
| `write-paths` | 432 | Counterfactual write-path pairs judged under a fixed policy |
| `guard-semantics` | 432 | Authorization semantics under six stated policies |
| `interrupt-wake` | 234 | Interrupt pending, wake edges, enable gating, and completion races |
| `input-filter` | 162 | Glitch-filter streak counting, saturation, bypass, and rearm |
| `fifo` | 108 | Occupancy, backpressure hold, blanking, and word capture |
| `counter` | 108 | Saturation, sampled steps, and commit/hold priority |
| `ot-access-control` | 108 | Address-mask qualification on an I2C target path |
| `arbitration` | 18 | Grant handshake and arbiter-tree acceptance |

This is pooled **development evidence**. Contract-2 was selected on the focused
FIFO/counter/arbitration subset. Public availability does not make that subset
an independent qualification test. Compare models on identical populations,
then reserve new group-disjoint material for release qualification.

## Install and run

The controller process does not load Torch. Each model's registry entry names
a separate compatible inference interpreter, runtime, checkpoint, and device.
For Hub downloads, install the optional client:

```bash
python -m pip install -e . -r requirements/benchmark.txt
python -m siliconshield.benchmark areas
python -m siliconshield.benchmark run \
  --registry /path/to/models.local.json \
  --out-dir /path/to/benchmark-results
```

`run` scores every registry entry sequentially and writes `<model-id>.json`
and `comparison.json`. Use `--models contract-1,contract-2` for a subset when
those ids exist in your registry, and `--areas fifo,counter` for selected areas.
A `--limit` keeps complete triplets up to that row budget. It can return fewer
rows than requested and is a smoke check, not a full benchmark.

For an offline or reviewed alternative population, pass
`--dataset /path/to/benchmark.jsonl` to `areas`, `run`, or `router-eval`.
A local file needs no Hub client. Its content hash identifies it in the report.
Use a new output directory per comparison because reports use fixed filenames.

## Read the output

- `overall`, `by_area`, and `by_mechanism` contain counts and NLL.
- `predictions` records labels by row id. `row_errors` preserves execution and
  protocol failures. Error rows count against accuracy and fail their triplets.
- `comparison.json` compares the supplied models. The Python helper
  `metrics.compare()` rejects mismatched benchmark hashes. There is no separate
  `compare` CLI subcommand. Match filters and inference settings as well.
- Whole-triplet aggregation is computed overall and for area-contained
  triplets. Current per-mechanism records contain row/class diagnostics,
  not a separately accumulated whole-triplet statistic.

This package does not produce the trainer's weighted metrics, Brier/ECE,
source-stratified bootstrap intervals, or an automatic release decision.
Its reports also do not retain all probabilities or the complete worker-ready
receipt. For a capability profile, retain the observed execution identity
alongside the scoring run, as described in [decision-model qualification](expert-qualification.md).

## Evaluate the router

```bash
python -m siliconshield.benchmark router-eval \
  --registry /path/to/models.local.json \
  --per-area 5 --seed 7 \
  --out /path/to/router-evaluation.json
```

The default samples five whole triplets per area and checks routing and expert
outputs. `--per-area 0` takes every triplet. Add `--arbitrate-per-area 1` to
invoke Muse on sampled triplets, after configuring the Muse CLI and login.
The default zero invokes no arbitration. Compare missed violations, false
alarms, unsupported scopes, and execution errors, not just expert agreement.
The output parent directory must already exist.

## Inference and label integrity

Workers use the question contract recorded in each export. The public dataset
uses the v1 contract, so compare exports qualified for those same question
bytes. The runner sends each row's state to the worker. It does not override
an incompatible export contract with the row's `questions` field.

CLI scoring is one request per row. Published campaign/SDK measurements may use
batch 16, different padding, or a different runtime. Those choices can change
boundary predictions. Reproduce both arms under the same worker protocol before
attributing a difference to model quality. Do not promise bitwise determinism
across hardware.

A review identified three ambiguous FIFO triplets in the original population.
A separately reviewed 5,580-row measurement excludes those nine rows. The default
pin remains the original 5,589-row file and does not apply that exclusion
silently. Preserve original results, identify any corrected or quarantined
population by its own hash, and use `--dataset` to select it explicitly.
See [decision-model qualification](expert-qualification.md#capability-evidence-shape).

## Reproduce the reviewed population

From the repository root, use the
[controller environment and workspace](setup.md#controller-and-workspace),
then install the benchmark download dependency:

```bash
python -m pip install -r requirements/benchmark.txt
python tools/prepare_benchmark.py \
  --output-dir "$SILICONSHIELD_WORKSPACE/reviewed-benchmark"
```

This downloads the pinned public population, verifies its exact contents,
and removes the three previously reviewed FIFO triplets whose checked storage
obligation is absent from the English requirement. It validates the resulting
population before publishing a new directory containing:

- `benchmark-reviewed.jsonl`: the reviewed routing population.
- `provenance.json`: source identity, excluded triplets and reason, row counts,
  and the resulting file identity.

The command performs no model inference or training, leaves the original
population unchanged, and refuses an existing output directory. For an offline
copy of the pinned source, add `--source /path/to/original-benchmark.jsonl`.
An unknown source or a result that differs from the recorded review is refused.

Use the prepared file explicitly:

```bash
python -m siliconshield.benchmark areas \
  --dataset "$SILICONSHIELD_WORKSPACE/reviewed-benchmark/benchmark-reviewed.jsonl"
```

The result has 5,580 decisions, including 99 FIFO decisions. It remains reused
development material. Newly worded versions of the excluded cases are new inputs
and need semantic review and fresh predictions before admission. Historical
5,589-row results, including the published 234-row temporal subset, keep their
original denominators. Do not silently replace those results with this subset.
