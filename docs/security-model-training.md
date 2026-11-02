# Training SiliconShield-RTL

[Documentation index](README.md)

The decision model learns requirement-conditioned security judgments using a
LAYA foundation and runtime. SiliconShield provides the dataset pipeline,
distributed trainer, continuation rules, checkpoint selection, and export
checks. Choose learning rates and sampling weights for the task, then record
them in the recipe.

## Prepare a run

Use a separate environment containing `requirements/core.txt` and a compatible
runtime checkout with residual-adapter support. Pin the runtime revision used
for the experiment. A moving checkout is not a reproducible dependency.

Before launch, freeze the parent export, source snapshot, dataset version,
sampling plan, question contract, review ledger, learning-rate schedule,
selection rules, evaluation configuration, and run budget. Check the model's
complete input encoding, not only raw RTL token length. Use a new output
directory for a new experiment. Training refuses rows whose question bytes
differ from `--question-contract` and refuses unreviewed templates when
`--review-ledger` is set. For new owned work, record this plan as a
[decision study](study-walkthrough.md) instead of assembling it by hand.
The campaign then enforces the frozen bytes at every launch.

The training and development files are normally `train.jsonl` and `dev.jsonl`.
The fitting mode reads TRAIN only. Calibration and test are reserved for a
separate finalization protocol.

## Launch the trainer

The following command illustrates the required arguments. Replace its values
with those from your reviewed recipe:

```bash
export LAYA_ROOT=/path/to/compatible-runtime
export SILICONSHIELD_SECURITY_RECIPE=/path/to/reviewed-recipe.json
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  siliconshield/training/laya/security_train.py \
  --model /path/to/parent-export \
  --data /path/to/accepted-dataset \
  --output /path/to/new-run --run-id example-continuation \
  --continue-training --sampling grouped \
  --pools /path/to/pools.json --development-views /path/to/views.json \
  --question-contract siliconshield-requirement-context-v1 \
  --micro-batch 4 --accumulation 4 \
  --max-updates 200 --eval-every-updates 25 \
  --no-final-evaluation
```

Use `--continue-training` when initializing from an adapted export. A new
unadapted foundation need not use it. Prefer the owned operator path described
in [campaigns](campaigns.md) when sharing training hardware. Direct `torchrun`
does not acquire the operator's job ownership automatically.

Distributed data parallel replicates the model per GPU. For three ranks,
microbatch four, and accumulation four, the nominal global batch is 48 decision
sequences. Three cards do not form a pooled memory allocation. Gradient
accumulation avoids unnecessary intermediate synchronization. The trainer folds
the preservation penalty into one supervised loss and calls `backward()` once
per microbatch. `no_sync()` wraps the non-final accumulated microbatches, while
the final one synchronizes across ranks.
`ddp_guard.require_coherent_training()` refuses a dual-backward regression at
startup. At runtime the trainer also
checks that DDP replicas agree and frozen parameters never move at early
updates, every development evaluation, and pre-finalization, and retains
every evaluated checkpoint's tensors under `eval_snapshots/` on rank 0.

## Training controls

| Setting | Behavior |
|---|---|
| `--encoder-lr`, `--adapter-lr`, `--head-lr` | Separate optimizer rates. Adapter LR defaults to head LR. |
| `--freeze-encoder` | Keep encoder weights fixed while training decision/adaptation parameters. |
| `--encoder-trainable-blocks N` | Train only the last N encoder blocks alongside the decision/adaptation parameters. Incompatible with `--freeze-encoder`. |
| `--disable-adapters` | Disable adapter branches for compatible unadapted initialization. Combine with `--freeze-encoder` for a true head-only control. |
| `--adapter-scale` | Gain for newly created residual branches. Inherited checkpoints retain their configuration. |
| `--checkpointing` / `--no-checkpointing` | Activation recomputation versus memory use. |
| `--sampling grouped`, `--pools` | Complete groups with declared pool quotas and per-pool `cycle`/`flat` modes. |
| `--max-updates`, `--epochs`, `--hours` | Update, epoch, and wall-clock bounds. |
| `--warmup`, `--lr-schedule` | `constant`, `cosine`, or `hold-decay`. Cosine ends at one tenth of peak. Hold-decay adds `--hold-updates` and a configurable `--lr-endpoint`. |
| `--eval-every-updates` | Development cadence. Zero uses epoch boundaries. Terminal weights are evaluated too. |
| `--eval-batch-size` | Evaluation batch per rank. Zero uses the training microbatch. |
| `--min-updates`, `--patience` | Minimum updates and unsuccessful evaluation count for early stopping. |
| `--checkpoint-interval` | Seconds between resumable saves. |
| `--question-contract` | Pinned `v0`/`v1` question bytes, validated on every row and recorded on export. |
| `--review-ledger` | JSONL English-to-assertion reviews, enforcing agreed template coverage. |
| `--review-packets` | Review-packet mapping for binding ledger entries to packet bytes (requires `--review-ledger`). |
| `--plateau-requires-full-exposure` | Refuse plateau interventions until delivered draws cover a full first pass with closed triplets. |
| `--no-diagnostic-weights` | Keep diagnostic metrics without saving alternative full model copies. |
| `--report-bootstrap`, `--reporting-contract` | Bootstrap replicate count (default 1,000) and a content-bound reporting contract. |

Read `--help` when preparing a command. The final recipe, not these example
values, belongs in the run signature. For v1 release data, explicitly pass
`--question-contract siliconshield-requirement-context-v1`, plus the reviewed
ledger, packets, pools, and development views required by the study. The trainer's default question contract is legacy
`v0`, so omitting that flag is not a compatible reproduction of either release.

Whole-triplet selection is implemented in the repository. The specialized
triplet-aware optimization and semantic-margin objectives in the historical
Contract-2 study used versioned study code. They are not enabled by a generic
trainer flag here. See [training recipes](training-recipes.md).

## Select on the full task

Development views partition rows exactly. Source-name lists support the older
layout. Structured selectors can include or exclude split groups, or name
exact `row_ids` so a reviewed, quarantined manifest maps onto the population.
`score_triplets` enables strict whole-triplet scoring grouped by `pair_id`.
A fresh single-view study uses `config_from_views_manifest()` to build an
exact-id selection view (`single_view_study`) with no retention population.
This predates decision-study manifests and is a different mechanism.
Use views to keep new capability and retained capability visible separately.

### Choose a selection path

For new work, freeze a decision-study manifest and use a protected-view objective.
The choice depends on the scientific priority declared before training:

| Need | Objective |
|---|---|
| Count-based protection with a target view | `protected-views-v1` |
| Safety-first ordering with whole-triplet scoring | `protected-views-v2-triplets` |
| Triplet-first ordering with false-safe safeguards | `protected-views-v3-balanced` |
| The balanced ordering plus exact source-specific floors | `protected-views-v4-sourced` |

The version is a policy choice, not a model-quality upgrade. Keep it frozen for
matched comparisons. Legacy transfer and incumbent-reference paths remain for
reproducing existing campaigns. They are not the recommended assembly path for
new decision studies. See the [walkthrough](study-walkthrough.md).

The [selection compatibility reference](reference.md#selection-compatibility-reference)
describes legacy campaign ranking and the implemented protected-view variants.

For protected-view objectives, `protected_passed` and
`quality_targets_passed` are separate fields. The sorting key prioritizes
protected feasibility, then the version's metric order. Target attainment is
reported but is not itself the first sorting component. A selected checkpoint
can therefore miss the study's quality target. Round assessment and promotion
must check the full frozen gates before accepting it.

The campaign creates a round recipe containing its selection reference. Jobs
must point at that recipe. Ranking or tolerance changes require explicit policy
registration. Do not compare runs with different rules as though selection were
unchanged. See [evaluation](evaluation.md) for metric definitions.

## Preserve earlier behavior

The optional `replay_preservation` recipe block declares its schema version,
positive weight and temperature, eligible TRAIN pools, and optionally a separate
`teacher_checkpoint`. The default teacher is the student's initialization. Set
`teacher_checkpoint` to use a different accepted model as the teacher.

Only eligible rows whose frozen teacher prediction matches the true target
contribute to the softened KL penalty. The caller applies normal sample and
accumulation weights. Teacher artifacts are hashed into preparation and resume
identity. Check protected development views for regressions even when this loss
is enabled.

### Optimization contract

The general trainer minimizes supervised cross-entropy with optional
preservation on eligible TRAIN rows. In schematic weighted-mean form:

$$
\mathcal{L}=\sum_i \bar w_i\left[-\log p_\theta(y_i\mid x_i)
+\lambda m_i T^2 D_{\mathrm{KL}}(p_{\mathrm{teacher}}^{(T)}
\Vert p_\theta^{(T)})\right].
$$

Here $x_i$ includes RTL, requirement, and assumptions. The normalized weights
$\bar w_i$ represent the configured sampling/loss contribution.
$m_i$ is one only for valid rows in eligible pools where the frozen teacher's
argmax matches the reviewed target. $T$ softens both distributions. The teacher
is detached and supplies no replacement labels. The actual implementation
applies validity and accumulation scaling within each distributed microbatch.
See [preservation loss](../siliconshield/training/laya/replay_preservation.py).

Loss share is not preservation strength: increasing $\lambda$ can reduce raw
KL enough that its fraction of total loss falls. Evaluate parameter movement,
teacher agreement on the eligible mask, and protected development performance
alongside CE and KL. Gradient norms, clip scale, and parameter deltas answer
different questions. None alone proves better generalization.

## Resume, continue, and finalize

| Operation | Purpose |
|---|---|
| `--resume` | Restore the same run's optimizer, RNG, data position, progress, and deterministically rebuilt sampler queues under compatible signatures. |
| New output + `--model` + `--continue-training` | Start a child experiment from an adapted export. |
| `--resume --finalize-development-only` | Export the selected development checkpoint without more optimizer updates. |
| Campaign finalization / evaluate-only flow | Separate exported-model calibration and final evaluation. |

A child should start from the selected export when that is the registered parent,
not automatically from terminal `latest.pt`. A selected checkpoint can remain
the initialization if every trained checkpoint fails the rules.

Loading a parent uses a temporary configuration copy so SDK load-time rewriting
does not mutate the sealed parent. Continuing a final-test-touched ancestor
requires an explicit new-study manifest that binds the parent and all four
partitions and retires prior evaluation populations. It does not make an old
test fresh again.

## Diagnose learning

Use TRAIN-only fitting to check whether the model can learn a complete fixed
subset: `--limit-train-rows N` retains complete designs, so the actual count can
exceed N. Evaluation uses that same fixed TRAIN population before and after
fitting. These reports are not admitted as development evidence by the episode
importer.

Inspect class recalls, false-safes, complete pairs, and per-source/mechanism
results. High missing-context recall can conceal poor satisfaction-versus-violation
discrimination. Compare initial, selected, and terminal weights separately.
Gradient norms before/after clipping and parameter deltas answer different
questions. A small loss on sampled batches does not establish coverage of the
entire training population.

Pool quotas, option-order exposure, and checkpoint selection are all part of the
recipe. The current row-order augmentation is seed/epoch/row dependent. Repeated
draws in one epoch need not expose new option orders. Evaluate order robustness
independently instead of assuming augmentation guarantees it.

The selected export records its question contract in
`question_contract.json`. Inference selects questions from that record.
Legacy exports without a record use the original contract.

Implementation: [trainer](../siliconshield/training/laya/security_train.py),
[selection](../siliconshield/training/laya/development_views.py),
[protected objectives](../siliconshield/training/laya/checkpoint_selection.py),
[teacher loss](../siliconshield/training/laya/replay_preservation.py),
[question contracts](../siliconshield/training/laya/question_contracts.py),
[review ledger](../siliconshield/training/laya/review_ledger.py),
[DDP guard](../siliconshield/training/laya/ddp_guard.py),
[coverage audit](../siliconshield/training/laya/coverage_audit.py),
[inventory](../siliconshield/training/laya/inventory.py).
