# Decision-model training recipes: Contract-1 and Contract-2

[Documentation index](README.md) · [Training](security-model-training.md) ·
[Research white paper](whitepaper.md)

This guide records how the released decision models were trained, which public
rows contributed to their weights, and the limits of reproducing those studies.
Current launch procedures belong in the [trainer guide](security-model-training.md).

Public data: the [combined training
set](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-training)
(19,050 rows, one v1 prompt). Public weights:
[Contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1)
and [Contract-2](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2).

Both studies were directed with the help of Muse Wonder and Muse Spark 1.3
on local GPU hardware.

## Published populations and weight lineage

The two subsets have different relationships to the released
weights. This distinction matters before any fine-tuning claim.

- **Contract-2** trained directly on its subset: all 2,700
  `contract-2-direct` rows (2,691 focused temporal rows plus 9
  solver-verified sibling rows) were the final study’s registered
  training population.
- **Contract-1** trained directly on 576 rows: the
  counterbalanced anchor population (192 complete triplets),
  which is included in the 16,350-row `contract-1-direct`
  subset. The other 15,774 rows are the follow-up
  improvement-study mixture (requirement, helper, expanded,
  and replay pools). Three study arms trained on that
  mixture. Every arm's selector held the parent, so no
  released weight comes from those rows. They are published
  for further training, not as rows Contract-1 saw.

These are registered populations. Delivered exposure at a particular selected
update must be read from its sampling record, not inferred from the dataset
size or terminal run. Both subsets are exactly label-balanced and share the
`siliconshield-requirement-context-v1` prompt. Every model-visible
requirement is plain English. The three-way question is fixed across rows.
Requirements belong in the state text, and acceptance reviews check for
comments or wording that reveal the target without requiring a decision
about the RTL.

## Contract-1: counterbalanced local specialist

Continued from the OpenTitan-expansion winner on DGX Spark.
The specialization repeatedly sampled 576 reviewed TRAIN rows in 192 triplets,
with a group-disjoint 288-row development set.

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
triplets, total correct, then NLL. The checkpoint at (257/288,
classes 96/95/66, zero false-safes, 66/96 triplets) is the
Contract-1 release. A later study pick (258/288) was not taken.
Terminal TRAIN fit reached 576/576. That measures fitting on the
small training population, not independent generalization, and is
distinct from the Contract-1 export. Development families are
disjoint by design group, not by mechanism.

The RTX improvement study then assembled the 16,350-row
mixture around that anchor (per-update triplet quotas from
16 slots: requirement 5, helpers 3, anchor 1, short guards 1,
expanded replay 3, HSB/transfer replay 3, seed 501) and ran
three arms. None beat the parent under the frozen gates, and
the parent was preserved unchanged.

## Contract-2: temporal specialist

Continued from the temporal-pilot parent export,
itself a low-learning-rate margin continuation of the focused
temporal pilot. The study ran two single-GPU arms. The
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
The released Contract-2 (203/234, classes 64/61/78, 17 false-safes,
51/78 triplets) is an explicit owner research selection that fails
the violates, false-safe, and NLL gates. Fresh SDK scoring
reproduces its development predictions exactly.

## Research findings carried into the platform

| Observation | What it supports | Implementation response |
|---|---|---|
| A selected export can equal the parent even after many optimizer updates. | A completed run does not establish an improved released model. | Distinct initial, selected, terminal, and incumbent identities, with retained evaluation tensors. |
| Strong TRAIN fit can coexist with weak whole-triplet development results. | Fitting and transfer are separate problems. Low loss alone cannot distinguish them. | Delivered-exposure audit, TRAIN fitting qualification, class and triplet diagnostics. |
| Better total accuracy can accompany worse violation recall or more false-safes. | Aggregate ranking can reward the wrong trade-off. | Frozen class, source, family, and retention constraints before promotion. |
| The temporal study improved its focused triplet count but failed other frozen gates. | A useful research artifact need not be an automatic campaign promotion. | Explicit owner selection in the Contract-2 release record, with failed gates disclosed. |
| Teacher agreement and weighted KL share can move differently. | The coefficient or loss fraction alone cannot rank preservation strength. | Masked teacher-correct preservation and behavioral diagnostics. |

These are engineering lessons from the studies, not causal estimates of the
benefit of each mechanism. Causal comparisons need matched initial weights,
data, delivered exposure, optimization settings, and independent repetitions.
The [evaluation protocol](evaluation.md) defines what can be claimed from each
kind of comparison.

## Reproducing with this repository

Exact historical reproduction needs the original objective implementation as
well as the parent, data, delivered exposure, and optimizer settings. Contract-2's
worst-member triplet loss and semantic class margins ran in versioned study
snapshots. They are not merged into the general trainer. Matching learning rates
alone does not reproduce that study. Whole-triplet checkpoint selection is
implemented in the repository, independently of those training losses.

The 234-row `focused-temporal-dev-234` view used for Contract-2 selection is
published in the benchmark. Contract-1's 288-row development view is not public.
Previously used development data supports matched comparisons. New
generalization claims require fresh group-disjoint evaluation, and the public
collections exclude the reserved qualification and test populations.

For a new study, follow these maintained procedures:

1. [Qualify the dataset and sampling pools](data-pipeline.md#dataset-versions-and-acceptance).
2. [Freeze the study](decision-study.md#freezing-the-plan), including the complete
   recipe, evaluation views, fitting qualification, and resource budget.
3. Use the [trainer guide](security-model-training.md) for current objectives,
   launch arguments, continuation, and export. Use [campaigns](campaigns.md)
   when submitting owned jobs.
4. [Validate the exported model](evaluation.md#validate-the-exported-model)
   before accepting it as a release candidate.
