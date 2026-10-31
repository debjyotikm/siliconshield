# SiliconShield-RTL security training

For the complete workflow, see [security-model training](../../../docs/security-model-training.md),
[the data pipeline](../../../docs/data-pipeline.md), and
[evaluation](../../../docs/evaluation.md). This page summarizes module entry points.

This directory contains dataset builders, a distributed trainer, and a campaign
runner for SiliconShield-RTL security decisions. Each example supplies RTL, a
requirement, and execution assumptions. The model predicts
`satisfies_requirement`, `violates_requirement`, or `insufficient_context`.

The trainer fine-tunes a LAYA checkpoint with residual adapters and runs on
the LAYA runtime, which is an external dependency (`LAYA_ROOT` or an
installed `laya` package).

The scripts ship in the Python package. The trainer also supports direct script
launch from the repository root:

```bash
python siliconshield/training/laya/security_train.py --help
```

Set `LAYA_ROOT` to a compatible LAYA checkout, or install a compatible `laya`
package. The trainer uses LAYA's residual-adapter support to produce the
specialized `siliconshield-rtl` checkpoint.

## Dataset definitions and recipes

Experiment definitions are loaded from local files. The public examples show the
formats. The synthetic fallbacks exercise the builders in tests.

| Setting | Input |
|---|---|
| `--cases-manifest` or `SILICONSHIELD_SECURITY_CASES` | JSON case catalog, split assignments, mutations, and optional source-specific cases |
| `SILICONSHIELD_EXTERNAL_CASES_MODULE` | Python module exporting `build_cases(sources)` and optionally `SPLIT_GROUP_ID` |
| `SILICONSHIELD_SEMANTIC_RECIPE_MODULE` | Python module exporting `scenario(seed, index)` |
| `SILICONSHIELD_SECURITY_RECIPE` | JSON source weights and checkpoint acceptance thresholds |

The source-backed dataset command also needs the referenced source files and a
Yosys executable. Use `--iverilog` to add compilation checks. The placeholder
catalog in [examples](../../../examples/README.md) is a format example, not a
complete training dataset.

Augmentation commands take an explicit case count. The guard augmentation's
`--seed` controls the order in which groups are selected.

For broader source collections, `source_contracts.candidates()` extracts local
transition requirements from small Verilog modules. Install the optional parser
with `pip install -r requirements/data.txt`. Extracted requirements describe
only the selected behavior. They do not certify every security requirement of a
source project. Keep the resulting case catalog outside Git.

Validate a prepared catalog with explicit split assignments using:

```bash
python -m siliconshield.training.laya.build_source_security \
  --catalog /path/to/local-catalog.json --output /path/to/new-dataset \
  --verification /path/to/solver-evidence \
  --yosys /path/to/yosys --iverilog /path/to/iverilog
```

The builder proves the satisfying implementation with a declared bound,
requires a reachable counterexample for its mutation, and checks that an
omitted expression has both completions. It compiles all three inputs,
including missing-context examples. Source families and their mutations must
share a partition. Check complete SiliconShield-RTL tokenization and the
pinned `--question-contract` before training, and enforce agreed template
reviews with `--review-ledger` when English requirements are involved.
Compilation and solver acceptance do not establish that the model's context
window contains the whole example. The DDP coherence gate refuses
dual-backward regressions at startup.

The source builder's manifest defines each population count under `counts`:
decision rows, source files, source-catalog records, extracted `base_design_id`
values, split groups, and exact RTL bodies. A parameter specialization or rewrite
does not become an independent source design. Use `split_group_id` when clustering
uncertainty estimates so related contracts and mutations stay together. Kish
effective sample size describes weight dispersion, not lineage independence.

Development reports include separate unweighted and weighted metrics, source and
mechanism breakdowns, and source-stratified bootstrap intervals over whole split
groups. The weighted view uses the declared source/mechanism/design/label row
weights. Missing sources are listed explicitly: results on the observed sources
do not estimate the full configured mixture. A source represented by one split
group receives point estimates and support counts, with its interval
unavailable. These diagnostics do not change checkpoint selection. Use
`--reporting-contract` to record a local reporting contract's contents and hash
in the run signature. `--report-bootstrap` controls the number of bootstrap
draws (default: 1,000).

## Training

To use a local recipe in a three-GPU run:

```bash
export LAYA_ROOT=/path/to/laya-next
export SILICONSHIELD_SECURITY_RECIPE=/path/to/security_recipe.json
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  siliconshield/training/laya/security_train.py \
  --model /path/to/laya-checkpoint \
  --data /path/to/security-dataset \
  --output /path/to/training-output \
  --run-id example-run --max-updates 200 --no-final-evaluation
```

The trainer uses BF16 and selects a checkpoint from development results. It
saves resumable state and exports the selected model. Calibration and test
scoring run separately through `security_campaign.py`. Use its `--help` for the
campaign entry point.

The systemd job command described in [setup](../../../docs/setup.md) is another
way to submit a run. It forwards the LAYA checkout path, GPU selection, and the
recipe path registered in the local profile.

For a fitting check, `--limit-train-rows N` selects complete TRAIN designs until
at least N rows are included. Counterfactuals and decision triplets stay
together. The trainer reads only the training partition and evaluates exactly
that selected population before and after fitting. Reports identify these as
training metrics. They are not admitted as development evidence by the episode
importer.

`--freeze-encoder` trains the decision layers and any residual adapters while
keeping encoder weights fixed. Add `--disable-adapters` for a head-only run from
an unadapted base. This creates no adapter branches, including in the exported
checkpoint. Resume signatures include both flags and the selected population.

With `--eval-every-updates N`, evaluation occurs at that update interval and at
the terminal update. With zero, it occurs at epoch boundaries and termination.
The terminal evaluation is saved in resumable state before the selected model
is exported. Logs distinguish gradients before and after clipping from actual
adapter and head parameter movement.

To continue an adapted checkpoint, pass `--continue-training`. An ancestor that
has already been evaluated on a final test requires an explicit
`--new-study-manifest`: it binds the parent files and all four new partitions by
hash, records the retired evaluation populations, and rejects reuse of their
inputs or lineages. The parent stays unchanged. Its old test score is historical
evidence, not a fresh test of the child model. Store this manifest outside Git.

`--development-views` defines populations that partition development rows
exactly, using source names or group selectors. The legacy mode ranks transfer
subject to retention accuracy. A frozen campaign reference adds joint
constraints and versioned ranking. Protected-view objectives set count
thresholds across every view. See [security-model
training](../../../docs/security-model-training.md) for selection modes, teacher
preservation, and continuation semantics.

For a fixed continuation budget, `--lr-schedule cosine` decays the learning rate
after warmup to one tenth of its peak at `--max-updates`. The default remains
constant after warmup. `--no-diagnostic-weights` avoids saving full alternative
checkpoints for accuracy and NLL. Resumable state and selected weights remain.

The Glimmer operator can run a local campaign configuration through `python -m siliconshield.research_loop.execution.strategy --config /path/to/campaign.json`.
That file supplies fitting arms, a measured training-gain threshold, the
permitted development budget, and the endpoint. Jobs execute sequentially on the
same three GPUs. The controller can reconnect to an owned job. Exhausting a
polling window leaves the job running. A completed run is never submitted again.
Set `ssh_host` in the operator configuration to reach a loopback endpoint
through a fresh SSH request instead of maintaining a forwarding tunnel.
