# Evaluation and evidence

[Documentation index](README.md)

For every result, name the checkpoint, evaluated population, inference settings,
and task. Successful execution, a valid report, and improved model quality are
different claims.

## Security-model metrics

| Metric | Interpretation |
|---|---|
| Accuracy | Correct three-way verdicts divided by decisions. |
| Macro-F1 / class recall | Label-balanced performance and which label is being missed. |
| False-safe count/rate | Violations predicted as satisfying. Divide by the number of actual violations for the rate. |
| Missing-context recall | Correct insufficient-context predictions divided by that class's support. |
| Complete-pair accuracy | Both related satisfying and violating decisions correct, divided by complete pairs. |
| NLL | Mean negative log-probability of the correct answer. Lower is better under the same evaluation protocol. |
| Per-source/mechanism/family results | Where aggregate gains or regressions occur. |
| Weighted metrics | Performance under declared source/mechanism/design/label weighting. |

A model can improve aggregate accuracy mainly by recognizing missing-context
examples while still failing complete designs. Report the confusion matrix,
class supports, pair count, and false-safe denominator. Do not use low training
loss as evidence of unseen-design generalization.

Reporting provides unweighted and weighted views and source-stratified
whole-split-group bootstrap uncertainty. Rewrites and mutations of one design
are not independent samples. A source with one group cannot support a
between-group uncertainty interval. Missing sources and unavailable intervals
must remain visible. Kish effective sample size describes weight dispersion, not
the number of independent designs.

Implementation: [security_reporting.py](../siliconshield/training/laya/security_reporting.py).

## Match comparisons

Compare initial, selected, and terminal checkpoints under the same dataset bytes,
pinned question-contract version, option construction, runtime, precision,
batch/padding behavior, and probability representation. Preserve per-row
outputs so disagreement can be inspected. A weights hash alone does not
specify the entire inference system. Report delivered sampler exposure at the
selected checkpoint, not only the terminal horizon.

The selected model can be the initialization. If so, unchanged results establish
preservation, not learning. The latest resumable state may contain later weights
that failed selection. Always name which artifact a table describes.

## Partition roles

| Role | Permitted use |
|---|---|
| TRAIN | Gradients, fitting checks, and training-side diagnosis. |
| Development | Checkpoint selection, bounded interventions, and iteration. |
| Calibration | A declared calibration procedure on separate groups. |
| Final test | Evaluation of a frozen candidate under a fixed protocol. |

A held-out diagnostic used to plan subsequent work is no longer independent
final evidence for that later work. Preserve its historical result and retire
it appropriately. Do not relabel consumed data to make it appear unseen.

`new_study.py` supports an explicit child study for an evaluated ancestor. It
binds parent and split hashes, checks retired IDs/states/groups, and keeps the
parent unchanged. It does not erase the parent's evaluation history.

## Validate the exported model

The training report measures the trainer path. `security_campaign.py` contains
export checks, grouped calibration, SDK scoring, order checks, and finalization
validation. Configure its campaign deliberately. It may evaluate reserved data.
Consult [training](security-model-training.md) and its entry-point help before
running finalization.

Match the chosen export with its selection receipt. Compare fresh SDK predictions
with the recorded development results before making deployment claims. Retained
per-evaluation tensors under `eval_snapshots/` let a reported development
number be rechecked against the exact evaluated weights. Keep calibration
fitted on its own partition. Test inference belongs to the frozen finalization
phase, not a per-update loop.

An export that passes development gates is not automatically a broadly validated
hardware-security model. State whether evaluation covers new variants, held-out
implementations, unseen mechanisms, or full external designs.

## Final evaluation protocol

Run final evaluation once, on a frozen candidate, after development selection
is closed. Use the pinned runtime, tokenizer, and question-contract bytes;
record their identities alongside the checkpoint manifest. Preregister the
reserved calibration and test partitions (separate split groups, no overlap
with TRAIN or development), then: fit calibration on its own partition, score
the frozen export on final test in one recorded pass, and compare fresh SDK
predictions with the recorded development results (plus byte-exact GUI checks:
a locked design must satisfy, a genuine bypass must violate, an unresolved
policy or missing helper must abstain without inheriting a verdict). Never
train on, select with, or iterate against the final partitions; a consumed
population stays consumed. Report confusion matrices, false-safe
denominators, paired and whole-triplet accuracy, per-source/mechanism support,
NLL with calibration, and delivered exposure at the selected checkpoint.

## Model-card release checklist

A release-ready [model card](siliconshield-rtl.md) names: parent checkpoint
and runtime revisions, trainable parameters, dataset version plus audit and
review-ledger digests, sampling exposure at the selected checkpoint,
selection policy with acceptance gates and the promotion receipt, per-view
results on development/calibration/final populations, and known limits
(including the GUI lock-hold failure mode and temporal-transfer behavior
where measured). No checkpoint is described as generally validated without
this evidence; development gates alone are not a release.

## Controller evaluation

The controller has different targets from the RTL model. It must read evidence,
choose an allowed action, calculate changes, and report consistently.

| Field or dimension | What it establishes |
|---|---|
| `action_correct` | Agreement with the episode's declared action policy. |
| `evidence_correct` | Structured numbers and evidence references satisfy the contract. |
| `explanation_correct` | Deterministic explanation checks did not identify unsupported claims. |
| `final_consistent` | Final structured response agrees with the recorded action/evidence. |
| Execution completion | Tool protocol completed without errors. |
| `verified` | The replay's combined checks passed. |

A deterministic grader is not an independent assessment of arbitrary scientific
reasoning. Report each component, per-action support, and lineage distribution.
Synthetic fixture replay tests interfaces. Live generation measures the model
under the chosen prompt, parser, runtime, and decoding settings.

`select_replay_candidate()` requires matched population, evaluator, source,
manifest, and decoding identities, coverage of all six operator actions, and
per-action/cohort non-regression against every supplied reference. It requires a
net verified gain. Selection does not deploy the adapter.

## Sealed controller evaluation

The evaluator's `--final` mode requires authenticated split membership and
cannot be combined with `--harness2-output`. The dataset identity machinery
records final use and blocks its reuse for training. Keep final evaluation
separate from any proposer or lesson-retention loop. Review both the selected
adapter and serving configuration before spending the reserved population.

Repeated development replay is useful for selection but should not be presented
as a fresh final benchmark. Neither correct arithmetic nor a completed job proves
that an intervention caused a better security model.
