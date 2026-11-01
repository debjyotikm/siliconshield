# Training recipes: Contract-1 and Contract-2

How the two released decision models were trained, which public
rows each one actually saw, and how to rerun or extend the work
with the trainer in this repository.

Public data: the [combined training
set](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-training)
(19,050 rows, one v1 prompt). Public weights:
[Contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1)
(`f4130223…`) and
[Contract-2](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2)
(`f916444f…`).

## Which rows each model saw

The two subsets have different relationships to the released
weights. This distinction matters before any fine-tuning claim.

- **Contract-2** trained directly on its subset: all 2,700
  `contract-2-direct` rows (2,691 focused temporal rows plus 9
  solver-verified sibling rows) were the sprint's training
  population.
- **Contract-1** trained directly on 576 rows: the
  counterbalanced anchor population (192 complete triplets),
  which is included in the 16,350-row `contract-1-direct`
  subset. The other 15,774 rows are the follow-up
  improvement-study mixture (requirement, helper, expanded,
  and replay pools). Three study arms trained on that
  mixture. Every arm's selector held the parent, so no
  released weight comes from those rows. They are published
  for further training, not as rows Contract-1 saw.

Both subsets are exactly label-balanced and share the
`siliconshield-requirement-context-v1` prompt. Every model-visible
requirement is plain English. No RTL, label, or verdict hint
leaks into the questions.

## Contract-1: counterbalanced local specialist

Continued from the OpenTitan-expansion winner `u175`
(`b03c20a4…`) on DGX Spark. The run fitted a small corrected
population hard: 576 TRAIN rows in 192 triplets, with a
group-disjoint 288-row development set.

| Setting | Value |
|---|---|
| Trainable | Full encoder, rank-256 residual adapters, decision head |
| Frozen | Unused action head |
| Learning rate | 2e-6 everywhere, 10-update warmup, cosine to 0.1 |
| Batch | 48 rows (microbatch 6), grouped triplets |
| Seed / budget | 3301, 250 updates, evaluate and save every 25 |
| Objective | Row cross-entropy, no teacher, no margin |

Development gates admitted a checkpoint with at most 8
false-safes, at least 48/96 per class, at least 10/96 whole
triplets, and at least 159/288 correct. Ranking among eligible
checkpoints ran false-safes, minimum class count, whole
triplets, total correct, then NLL. Update 125 (257/288,
classes 96/95/66, zero false-safes, 66/96 triplets) is the
released Contract-1. The study's own later pick at update 200
(258/288) was not taken. Terminal TRAIN fit reached 576/576,
so the weights memorize their small population. Development
families are disjoint by design group, not by mechanism.

The RTX improvement study then assembled the 16,350-row
mixture around that anchor (per-update triplet quotas from
16 slots: requirement 5, helpers 3, anchor 1, short guards 1,
expanded replay 3, HSB/transfer replay 3, seed 501) and ran
three arms. None beat the parent under the frozen gates, and
the parent was preserved unchanged.

## Contract-2: temporal specialist

Continued from the four-arm sprint's F2 export (`d0a1696a…`),
itself a low-learning-rate margin continuation of the focused
temporal pilot. The sprint ran two single-GPU arms. The
triplet arm produced Contract-2.

| Setting | Value |
|---|---|
| Trainable | Existing adapters, decision head |
| Frozen | Encoder |
| Learning rate | 1e-7 adapter and head, 5-update warmup, hold-decay to 0.2 |
| Batch | 48 rows (microbatch 3, 16 accumulations), grouped triplets |
| Seed / budget | 7502, 300-update horizon, evaluate every 10 |
| Stopping | At least 150 updates, patience 3, full-exposure plateau rule |
| Objective | Row cross-entropy plus worst-member triplet loss (weight 0.5) |

Sampling pools over the 2,700 rows carried weights 4/3/1/7/1
for fifo, counter, arbitration, hard-triplet replay, and
siblings. Every pool cycles complete triplets except the
hard pool. The frozen promotion floor demanded at least 202
correct, 60 satisfies, 64 violates, all 78 context rows, at
most 14 false-safes, 50 triplets, bounded NLL, and per-domain
floors. The trainer's selector never left initialization.
step 70 (203/234, classes 64/61/78, 17 false-safes, 51/78
triplets) is an explicit owner research selection that fails
the violates, false-safe, and NLL gates. Fresh SDK scoring
reproduces its development predictions exactly.

## Reproducing with this repository

For new owned studies, follow [decision studies](decision-study.md).
That path binds the actual job to accepted rows, the exact recipe,
measured fitting qualification, and a cumulative time budget.
The direct trainer command below describes the lower-level interface.
It does not itself register an enforced study or qualify a recipe.


The public set plus the trainer below rerun the Contract-2
shape and the Contract-1 improvement study. The original
Contract-1 run executed on DGX Spark hardware with the same
trainer family.

**1. Fetch the rows.** Download `data/train.jsonl` from the
training set. Split out a development population before
training. Never select checkpoints on TRAIN fit. The
Contract-2 sprint selected on a 234-row development view
that now ships inside the public benchmark (tagged
`focused-temporal-dev-234`). Treat it as previously used
development data, not fresh qualification, since
Contract-2 was selected against it. The Contract-1
study's 288-row view does not ship publicly. Either way,
assemble a new group-disjoint development split first
and freeze its hash.

**2. Write a pools manifest.** The trainer samples grouped
triplets from a `siliconshield-sampling-pools-v1` manifest:
named pools of row ids with `mode` (`cycle` or `flat`) and
`weight`. Every pool must hold complete triplets, and row
assignment across pools must be mutually exclusive. The
loader refuses anything else. Verify a manifest with
`dataset_versions.verify_pools` before launching.

**3. Launch.** The trainer entry point is
`siliconshield.training.laya.security_train` under
`torch.distributed.run`, with `LAYA_ROOT` pointing at a
compatible runtime:

```bash
export LAYA_ROOT=/path/to/compatible-runtime
python -m torch.distributed.run --standalone --nproc_per_node=1 \
  siliconshield/training/laya/security_train.py \
  --model /path/to/parent-export \
  --data /path/to/dataset-dir \
  --pools /path/to/pools.json \
  --sampling grouped \
  --continue-training --freeze-encoder \
  --question-contract siliconshield-requirement-context-v1 \
  --development-views /path/to/views.json \
  --output /path/to/run --run-id example-01 \
  --max-updates 300 --eval-every-updates 10 \
  --min-updates 150 --patience 3 \
  --plateau-requires-full-exposure
```

Learning rates, warmup, schedule (`constant`, `cosine`, or
`hold-decay`), batch geometry, and seed are flags. See
`--help` and the [trainer
summary](../siliconshield/training/laya/README.md). A
`SILICONSHIELD_SECURITY_RECIPE` file adds frozen-teacher
preservation over named pools, which keeps correct parent
behavior without preserving parent mistakes.

**4. Select and export.** Development views partition the
development rows exactly. `checkpoint_selection` ranks
eligible checkpoints under a frozen protected-views
objective (`protected-views-v2-triplets` or later), never
on TRAIN fit. The selected export is reloaded through the
production worker and must reproduce the trainer's
predictions before it counts as a release candidate.

Two honest boundaries: the worst-member triplet training
loss and the semantic class margins behind Contract-2 ran
in versioned campaign snapshots and are not merged into
this trainer, which optimizes row cross-entropy with
optional teacher preservation. Whole-triplet checkpoint
*selection* is in the repository. Triplet-aware
*optimization* is the documented next merge. And the public
set intentionally excludes sealed qualification and test
populations, so headline numbers always need a fresh
group-disjoint evaluation to mean anything.
