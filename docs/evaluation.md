# Evaluation and evidence

[Documentation index](README.md)

For every result, name the checkpoint, evaluated population, inference settings,
and task. Successful execution, a valid report, and improved model quality are
different claims.

## Metric specification

Use class order **satisfies, violates, insufficient context**. Let $C_{ab}$
count true class $a$ predicted as $b$, $N$ be the population size, and $p_{iy_i}$
be the score assigned to row $i$'s reviewed target. The confusion matrix has
true labels on rows and predictions on columns.

| Metric | Definition and denominator |
|---|---|
| Accuracy | $\mathrm{tr}(C)/N$. Always include correct and total counts. |
| Class recall | $C_{kk}/\sum_b C_{kb}$. Name the class support. |
| Macro-F1 | Mean class F1 over supported classes in the trainer report. Report supports so absent classes are visible. |
| False-safe rate | $C_{\mathrm{violates},\mathrm{satisfies}}/N_{\mathrm{violates}}$. A violation predicted as context is a different error. |
| False-violation count | $C_{\mathrm{satisfies},\mathrm{violates}}$. Measures unnecessary violation flags. |
| Unnecessary abstention | Fully specified satisfies/violates cases predicted as insufficient context. |
| NLL | Mean $-\log p_{iy_i}$ with natural logarithms. Match score construction and numerical floors before comparing. |
| Brier | Mean summed squared error across the three class probabilities. The trainer reports the sum over classes, not its division by three. |
| ECE | Ten confidence bins, weighted by bin support, comparing average top score with empirical accuracy. Bin-sensitive and not a guarantee for an individual answer. |

The trainer also reports a false-safe fraction among predictions of satisfies.
That uses a different denominator from false-safe rate among actual violations.
Label the denominator explicitly rather than quoting a bare percentage.

### Linked decisions

For complete satisfying/violating pairs, both predictions must be correct.
For a triplet $g$ containing all three reviewed labels, define

$$
H_g=\prod_{i\in g}\mathbf{1}(\hat y_i=y_i),\qquad
\mathrm{WholeTripletAccuracy}=\frac{1}{G}\sum_{g=1}^{G}H_g.
$$

The linking key is the dataset's `pair_id`, despite the name. It must identify
exactly one member of each label for a complete triplet. Report complete-group
support and any malformed or execution-failed groups. A confusion matrix alone
cannot recover whole-triplet accuracy because it loses the row linkage.
Do not estimate it by cubing row accuracy: linked decisions are dependent.

For a purely illustrative three-member group, correctly classifying satisfies
and context but missing its violation gives 2/3 row accuracy and 0/1 whole
triplets. A model that shifts violations into context can therefore improve
one class while losing coherent handling of the underlying situation.

### Report owners and execution failures

| Report | Implemented measurements |
|---|---|
| Trainer and finalization | Confusion, supported-class recall/F1, false-safes, NLL, summed Brier, 10-bin ECE, configured development views and selection metrics. |
| Training reporting extension | Weighted/unweighted and per-source/mechanism metrics, complete-pair statistics, source-stratified split-group bootstrap intervals. |
| Area benchmark | Row/class, area/mechanism, false-safe, false-violation, abstention, whole-triplet, NLL, and execution-error counts. No Brier, ECE, weighted views, or bootstrap intervals. |

In the area benchmark, failed rows stay in row and class denominators and count
as incorrect. They make their linked triplet fail. NLL averages only rows with
usable predictions, with a target-score floor of `1e-12`. It is infinite when
none are usable. Report execution coverage alongside NLL so a model cannot
appear better by failing on hard rows. Such failures do not increment the
observed false-safe count because no label was produced.

The benchmark's `triplets_total` counts encountered groups, including incomplete
ones. `triplets_incomplete` and `triplets_failed` distinguish malformed groups
from complete groups containing execution errors. Publish strict whole-triplet
accuracy only with those counts disclosed, and qualify the population before
using it for selection. Per-mechanism triplet fields are not populated by the
current benchmark scorer, while overall and per-area triplets are.

Implementations: [trainer metrics](../siliconshield/training/laya/security_train.py),
[reporting extension](../siliconshield/training/laya/security_reporting.py), and
[benchmark scorer](../siliconshield/benchmark/metrics.py). Raw scores are not
calibrated risk estimates. See [confidence limitations](LIMITATIONS.md#confidence-is-not-reliability).

## Independence and uncertainty

The independent unit is the underlying lineage or `split_group_id`, not each
mutation, rewritten requirement, or sampled row. Separate decision count,
unique RTL bodies, mechanisms, and independent groups. Thousands of related
rows can still represent a small number of independent situations.

The reporting extension uses a **95% source-stratified whole-cluster percentile
bootstrap**. It samples split groups within source, retains their member rows,
and holds each observed source stratum's mass fixed. Record the bootstrap seed,
replicate count, cluster key, and per-source cluster counts. It returns
unavailable intervals when any included source has fewer than two groups or
when bootstrap is disabled. It does not invent an OpenTitan-wide interval from
one OpenTitan group.

Weighted accuracy uses fixed declared row weights. Their dispersion is
summarized by Kish effective sample size:

$$
N_{\mathrm{eff}}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

This is not the number of independent designs. Publish weighted and unweighted
results with the weighting definition. Aggregate source weighting must not
conceal a weak mechanism or class.

For an intervention comparison, use identical populations and paired per-group
outcomes. Paired uncertainty for differences needs an explicitly implemented
analysis. Overlap or separation of two independently reported intervals is not
itself that analysis. Multiple seeds can assess optimization variability, but
repeated selection on one development population still creates selection bias.

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

### What an experiment can establish

A recipe comparison supports a causal attribution only when relevant factors
are controlled. Apart from the registered factor under test, match the
initializer, data and split hashes, delivered sampling, optimizer/schedule,
trainable groups, inference protocol, and selection rule. If several factors change, report a revised-recipe outcome rather than
crediting one factor. Preserve unsuccessful arms and predeclared stopping rules.

Counterfactual requirements on the same RTL reduce code-only shortcuts. They do
not by themselves prove semantic reasoning: requirement templates and lineage
can also carry shortcuts. Use lineage-disjoint evaluation, identifier rewrites,
wording shifts, and appropriate single-surface baselines to test that claim.

A router study should compare each single expert with routing and with routing
plus arbitration on identical cases. Record unsupported scopes, failed calls,
model/agent time, and per-class errors. Current routing tiers are fixed
engineering heuristics, not learned probabilities or statistical confidence
bounds. The repository does not establish a measured speedup in bug discovery
or superiority over the best single expert merely by consulting both models.

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

The public 5,589-row benchmark is pooled development data. Contract-2's
234-row focused subset was used for selection. The benchmark's 756-row
`requirement-clarity` area is also a different population from Contract-1's
756-row shifted qualification in [limitations](LIMITATIONS.md). Equal row
counts do not identify a dataset. Compare hashes and provenance.

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

Before spending reserved data, confirm the selected export's identity and SDK
parity on development, and complete product-input checks on a separate reviewed
population. Exercise reset-exempt and reset-inclusive requirements, genuine
bypasses, and relevant versus irrelevant omissions. These checks do not imply
that every locked design will be classified correctly.

Then freeze the candidate, inference protocol, calibration procedure, and
final-test population. If calibration is part of that protocol, fit it only
on its separate partition. Score the frozen system on final test in one
recorded pass. Do not train on, select with, or iterate against those outcomes.
A consumed population stays consumed.

Report confusion matrices, false-safe denominators, pair and whole-triplet
accuracy, per-source/mechanism support, calibration status, NLL, and delivered
training exposure at the selected checkpoint. A new study can retire spent
evaluation populations. It cannot make them independent again.

## Model-card release checklist

A release-ready [model card](siliconshield-rtl.md) names: parent checkpoint
and runtime revisions, trainable parameters, dataset version plus audit and
review-ledger digests, sampling exposure at the selected checkpoint,
selection policy with acceptance gates and the promotion receipt, per-view
results on development/calibration/final populations, and known limits
(including the observed lock-hold false alarm and temporal-transfer behavior
where measured). No checkpoint is described as generally validated without
this evidence. Development gates alone are not a release.

Agent tool-use and action-policy checks are documented under
[Glimmer replay qualification](inference.md#controller-evaluation). They measure
the agent configuration, separately from decision-model accuracy.
