# SiliconShield-RTL capabilities and limitations

[Documentation index](README.md) · [Model-family card](siliconshield-rtl.md) ·
[Evaluation protocol](evaluation.md) · [Router](router.md)

Status: checkpoint-specific evidence summary  
Verification method: every aggregate below was recomputed from the
content-addressed evaluation receipts. No model was rerun and no solver was
rerun for this write-up.

The detailed matrices and examples describe the SiliconShield-RTL Contract-1
(C1) decision model. A separate [Contract-2 section](#contract-2-temporal-specialist)
records the focused temporal decision model's results and limits. Counts from
one release must not be assigned to another, and neither establishes general
RTL security competence.

In summary, this checkpoint is suited to explicit, local, synchronous rules
whose guard is visible in the RTL. It should not be relied upon where the
answer requires broad policy interpretation, missing-context reasoning,
unfamiliar state encodings, or whole-design analysis. Its raw scores are not
calibrated probabilities, so any security-relevant result still requires
executable verification or engineering review before action is taken.

## Model and evaluation identity

The numbers in this document apply only when the weights, question contract,
runtime behavior, and label order are kept together.

| Field | Evaluated value |
|---|---|
| Release | SiliconShield-RTL Contract-1 (C1) |
| Weights | [danieltombay/siliconshield-rtl-contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1) |
| Question contract | `siliconshield-requirement-context-v1` |
| Adapter metadata | 28 residual-adapter layers, rank 256, scale 8 |
| Evaluation precision | BF16 model inference, FP32 probabilities and metrics |
| Input budgets | `max_len=1536`, `head_max_len=256` for the question and options |

The model returns one of three labels:

- `satisfies_requirement`
- `violates_requirement`
- `insufficient_context`

The label is conditional on the supplied RTL, requirement, and assumptions.
Changing any of those inputs can legitimately change the answer. Loading the
same weights with a different question contract is also a different inference
system and is outside the evidence reported here.

## How to read the evidence

Four kinds of evidence appear below. They answer different questions.

| Evidence | What it can support | What it cannot support |
|---|---|---|
| Frozen development views | Performance on populations used during model development | Independent generalization |
| Family-disjoint qualification | Transfer to a small number of unseen authored families | Broad real-world RTL coverage |
| Requirement-context test | Behavior under a deliberate wording and representation shift | Performance on every semantic shift |
| Six-example authored local-guard set | Correct behavior on those exact examples | An accuracy estimate |

The two unseen evaluations were frozen before inference. Both were balanced
across the three labels, solver-checked, and group-disjoint from their matching
training and development data. They are still authored local-contract datasets,
not a random sample of production hardware. Both are now spent. Their rows,
labels, errors, and logits must not feed further training or checkpoint
selection.

The historical final test and unrelated sealed OpenTitan and HardSecBench
partitions were not evaluated for this study. There is therefore no final-test
claim for this checkpoint.

In this document, a false-safe is a true requirement violation predicted as
`satisfies_requirement`. It is more dangerous than an ordinary classification
error because it can make a faulty design look acceptable.

## What the model does well

### Explicit local guards

The strongest use case is a requirement that names a local state update and a
guard that can be read directly from one small RTL region. Examples include:

- a register write that requires a visible privilege signal
- a clear operation that requires both a request and an enable
- a write that is legal at one exact address
- a sticky bit that may clear only under a declared condition
- a local mask, strobe, owner-width, or default-deny rule

The six authored local-guard examples form three counterfactual pairs. Each pair keeps the
requirement fixed and changes one guard in the RTL.

| Pair | Violating design | Satisfying design |
|---|---|---|
| Privilege | Write checks only `wr_en` | Write checks `wr_en && privileged` |
| Clear authorization | Clear checks only `clear_req` | Clear checks `clear_req && clear_enable` |
| Address qualification | Write accepts an address range | Write accepts only `8'hA0` |

For example, the following requirement is narrow and directly connected to the
implementation:

> After reset, `cfg_reg` may change only on a write performed while
> `privileged` is 1. A write while `privileged` is 0 must leave `cfg_reg`
> unchanged.

The violating branch is:

```systemverilog
else if (wr_en)
  cfg_reg <= wdata;
```

The satisfying branch is:

```systemverilog
else if (wr_en && privileged)
  cfg_reg <= wdata;
```

In one matched batch, the checkpoint classified all six authored examples
correctly. Scores below are raw SDK values:

| Example | Expected decision | Winning model score |
|---|---|---:|
| Missing privilege guard | Violates | 99.9889% |
| Enforced privilege guard | Satisfies | 94.9334% |
| Omitted clear enable | Violates | 98.4911% |
| Enforced clear enable | Satisfies | 99.9990% |
| Overbroad address check | Violates | 99.9998% |
| Exact address check | Satisfies | 99.9993% |

These six examples were selected to demonstrate a measured strength rather
than to estimate one. A 6/6 result is not a benchmark and must never be
reported as 100% accuracy. The table describes that measured batch only.

### Unseen counterbalanced write-path families

The strongest independent result came from 288 decisions arranged as 96
complete triplets across four unseen structural families. Each triplet contains
a satisfying case, a violating case, and an insufficient-context case tied to
the same underlying situation.

| Metric | Result |
|---|---:|
| Overall accuracy | 249/288, 86.46% |
| Satisfies recall | 81/96, 84.38% |
| Violates recall | 92/96, 95.83% |
| Insufficient-context recall | 76/96, 79.17% |
| False-safes | 0/96 violations |
| Whole triplets correct | 59/96, 61.46% |
| Negative log likelihood | 0.7566 |
| Brier score, sum over classes | 0.2360 |
| Expected calibration error, 10 bins | 12.13% |

The confusion matrix makes the error directions explicit. Rows are true labels
and columns are model predictions.

| True \ predicted | Satisfies | Violates | Insufficient context |
|---|---:|---:|---:|
| Satisfies | 81 | 0 | 15 |
| Violates | 0 | 92 | 4 |
| Insufficient context | 16 | 4 | 76 |

Accuracy ranged from 83.33% to 90.28% across the four families. This reduces the
chance that the aggregate result came from one unusually easy family. It does
not remove the main statistical limitation. Four families are too few to claim
coverage of arbitrary RTL.

The row-level Wilson interval for accuracy is 82.02% to 89.93%. Rows within a
triplet and family are correlated, so this is descriptive only. It is not a
family-clustered population interval.

A zero false-safe count on this population is encouraging but carries no
implication for other populations. The shifted evaluation below did produce
false-safes.

### Familiar development populations

The checkpoint was also evaluated on a frozen 4,095-decision development
population. These results are useful for describing retained capabilities, but
they are not independent evidence because development data informed the broader
model-development process.

| Development view | Correct | False-safes | Whole triplets |
|---|---:|---:|---:|
| HardSecBench-derived | 2,601/2,610, 99.66% | 7 | 862/870, 99.08% |
| OpenTitan-derived | 180/180, 100% | 0 | 60/60, 100% |
| Authored transfer | 561/567, 98.94% | 0 | 183/189, 96.83% |
| Requirement-understanding stress | 340/738, 46.07% | 22 | 15/246, 6.10% |

Across all four views, the checkpoint reached 3,682/4,095 correct, or 89.91%.
Its satisfies, violates, and insufficient-context recalls were 86.67%, 95.75%,
and 87.33%. It made 29 false-safe errors among 1,365 true violations and got
1,120/1,365 whole triplets correct. NLL was 0.6541 and 10-bin ECE was 8.40%.

The contrast within this table matters more than the aggregate score. The model
retains very high accuracy on familiar source and contract styles, while the
requirement-understanding stress view exposes a large weakness. Reporting only
the first three rows would give a misleading picture.

## Where the model fails

### Requirement and representation shifts

The second unseen evaluation deliberately changed both the wording and the RTL
representation. It contained internal storage, inverted storage, and
rotate/debug update structures from two unseen origins. The model did not
transfer well.

| Metric | Result |
|---|---:|
| Overall accuracy | 330/756, 43.65% |
| Satisfies recall | 80/252, 31.75% |
| Violates recall | 225/252, 89.29% |
| Insufficient-context recall | 25/252, 9.92% |
| False-safes | 25/252 violations, 9.92% |
| Whole triplets correct | 12/252, 4.76% |
| Negative log likelihood | 3.9405 |
| Brier score, sum over classes | 1.0421 |
| Expected calibration error, 10 bins | 50.16% |

| True \ predicted | Satisfies | Violates | Insufficient context |
|---|---:|---:|---:|
| Satisfies | 80 | 159 | 13 |
| Violates | 25 | 225 | 2 |
| Insufficient context | 49 | 178 | 25 |

The model predicted `violates_requirement` on 562 of 756 rows, or 74.34% of the
population. High violation recall here does not mean the model understood the
requirements. It frequently predicted a violation regardless of the correct
label. That behavior accounts for the low satisfies recall, the very low
insufficient-context recall, and the near-zero whole-triplet consistency.

No tested representation transferred well:

| Shifted group | Accuracy | Satisfies recall | Violates recall | Insufficient recall |
|---|---:|---:|---:|---:|
| Internal storage | 45.24% | 38.10% | 85.71% | 11.90% |
| Inverted storage | 45.24% | 34.52% | 91.67% | 9.52% |
| Rotate/debug updates | 40.48% | 22.62% | 90.48% | 8.33% |

This is the clearest reason not to market the checkpoint as a general RTL
security reasoner.

The row-level Wilson interval for accuracy is 40.16% to 47.21%. The 756 rows
come from only two independent origin groups. That is too little support for a
reliable group-clustered uncertainty interval.

### Missing context

`insufficient_context` is the most fragile output. The correct label requires
more than noticing an unfamiliar identifier. The omitted implementation or
policy must be capable of changing whether the requirement holds.

It is a semantic label, not a general abstention mechanism. Low confidence,
unfamiliar syntax, or model difficulty does not by itself make
`insufficient_context` correct. This checkpoint has no separately qualified
out-of-distribution detector.

Consider this requirement:

> After reset, `cfg_reg` may change only on a write performed while
> `privileged` is 1.

Now suppose the RTL gates the write with `policy_allow`, but the input does not
define the relationship between `policy_allow` and `privileged`:

```systemverilog
else if (wr_en && policy_allow)
  cfg_reg <= wdata;
```

This fragment omits the internal definition of `policy_allow`. Compatible
definitions can make it satisfy or violate the requirement, so its reviewed
label is `insufficient_context`. This differs from a complete module that
declares `policy_allow` as an unconstrained primary input: an allowed assignment
that bypasses privilege would then be a violation. In a small diagnostic screen, the checkpoint instead returned
`satisfies_requirement` with a score of 99.9512%.

The same screen contained six newly authored insufficient-context examples and
the model got 0/6 correct. A second six-example screen using more familiar
omitted-implementation patterns reached 2/6. These are small diagnostics, not
population estimates. The failure is nonetheless consistent with the 9.92%
insufficient-context recall observed in the larger shifted evaluation.

Do not trust a definite `satisfies_requirement` or `violates_requirement`
answer when any of the following is missing:

- the mapping between policy terms and RTL signals
- a helper module or task that controls the update
- an encoding definition for privilege, role, lock, or ownership
- the reset and observation semantics needed by the requirement
- an indirect update path that can reach the protected state

### Multi-path authorization and missing conjuncts

The model can recognize many direct missing guards, but it does not reliably
inspect every path. A representative diagnostic had a normal write protected
by `privileged` and a debug write without that guard:

```systemverilog
else if (dbg_wr)
  key_reg <= dbg_data;
else if (wr_en && privileged)
  key_reg <= wdata;
```

The requirement applied the privilege rule to both paths. The expected answer
was `violates_requirement`. The checkpoint returned
`satisfies_requirement` with a score of 99.7409%. This is a false-safe.

A single correct path is therefore not evidence that the model checked every
assignment to the protected state. Use structural search and executable checks
to enumerate all writers.

### Correct lock guards can be rejected

The model also makes the opposite error. It can label a correct lock guard as a
violation. In a diagnostic case, this local branch satisfied the requirement:

```systemverilog
else if (wr_en && !locked)
  control_reg <= wdata;
```

The checkpoint returned `violates_requirement` with a score of 99.9550%.
Another case placed two write paths under one outer `if (!locked)` block. It was
also incorrectly classified as a violation, with a score of 98.8771%.

These false violations make the model unsuitable as an automatic rejection
gate. A flagged design still needs a counterexample or manual confirmation.

### Whole-triplet consistency

Row accuracy can hide contradictory behavior. A whole triplet counts as correct
only if the model gets the satisfying, violating, and insufficient-context
versions of one situation right.

- The unseen counterbalanced qualification reached 61.46% whole-triplet
  accuracy despite 86.46% row accuracy.
- The shifted requirement-context evaluation reached only 4.76% whole-triplet
  accuracy.
- The requirement-understanding development view reached 6.10%.

This gap means the model may appear convincing on one example while failing a
nearby counterfactual. Demos and regression tests should include both sides of a
requirement, not only a known violation.

### Confidence is not reliability

Raw softmax scores are model scores. They are not calibrated probabilities that
the answer is correct.

On the unseen counterbalanced qualification:

- mean confidence was 99.28% on correct answers
- mean confidence was 91.62% on wrong answers
- 28 of 39 errors had confidence at or above 90%

On the shifted requirement-context evaluation:

- mean confidence was 93.64% on correct answers
- mean confidence was 92.30% on wrong answers
- 346 of 426 errors had confidence at or above 90%
- 17 of 25 false-safes had confidence at or above 90%

The almost identical confidence on correct and incorrect shifted examples is a
serious calibration failure. Reports should retain all three scores. Neither
the highest score nor a value near 99% should be described as certainty.

## Coverage that has not been established

The current evidence does not establish dependable behavior for:

- arbitrary OpenTitan blocks or complete OpenTitan subsystems
- arbitrary HardSecBench designs or every CWE represented in its provenance
- full-chip, multi-module, or deeply hierarchical information flow
- liveness, long temporal sequences, or unbounded properties
- clock-domain crossings, asynchronous logic, or metastability
- four-state `X` and `Z` semantics when the contract assumes two-state logic
- analog behavior, side channels, timing leakage, power leakage, or fault
  injection
- firmware and hardware interactions beyond the supplied RTL and requirement
- synthesis, place-and-route, or physical implementation effects
- vulnerability localization or automatic root-cause identification
- general CWE classification
- automatic repair correctness

OpenTitan and HardSecBench are source material, not blanket certifications. A
perfect result on a small OpenTitan-derived development view means that the
checkpoint got those 180 decisions right. It does not mean that it understands
all OpenTitan IP or all hardware security requirements.

### Labels represent scoped contracts

Training and evaluation labels describe one registered property under declared
assumptions. They do not label an entire module as secure or insecure. The
solver-backed data focuses on local, bounded, reset-conditioned contracts under
the checker's supported semantics.

Several decisions can come from one source design through different
requirements, mutations, rewrites, or missing-context completions. Decision
counts are therefore not counts of independent chips or modules. Whole-triplet
and family results are more informative than treating every row as an
independent sample.

CWE identifiers attached to HardSecBench-derived material describe source
provenance. They were not trained or evaluated as a general CWE classification
target. Likewise, an OpenTitan-derived row tests its local contract. It does not
carry an OpenTitan-wide security label.

## Product and interface limitations

### Model assessment is not verification

The model does not execute a proof. `satisfies_requirement` means that the model
assessed the supplied input as satisfying the requirement. It does not certify
the design, and it says nothing about requirements that were not supplied.
`violates_requirement` identifies a candidate issue. It is not a confirmed
vulnerability until a requirement-bound check produces a counterexample or an
engineer verifies it.

Use the [investigation workflow](investigation.md) when executable checks are
available. A passing bounded simulation means that no counterexample was found
within that check. It is not automatically a proof for all time.

### Assessments are bounded by the supplied RTL

The router assesses the state supplied by its caller. It does not automatically
elaborate a repository or prove that every writer, helper, generated block, or
module instance is included. For a large design, supply a region containing the
logic needed to judge one requirement and preserve its source locations. Each
verdict applies to that region and its assumptions, not the entire design.

### Worker token-budget truncation receives no verdict

The evaluated model used a maximum sequence length of 1,536 tokens and a
256-token budget for the question and options. Inputs that do not fit the
configured contract must be rejected as truncated. They must not be silently
shortened and must not be relabeled as `insufficient_context`.

This guarantee covers truncation detected by the inference worker. It cannot
detect source removed before a request was constructed. Integrators must
preserve the relevant RTL and disclose any excerpt boundaries. A short input
that fits the budget is not evidence that the full semantic context is present.

### Requirements must be precise

The model works best when the requirement:

- names the state being protected
- identifies the allowed update condition
- states the reset and observation scope
- uses signal meanings supplied in the input
- covers one local property rather than a broad policy document

Better wording cannot recover missing design information. If the policy-to-RTL
mapping is unknown, the right product behavior is to request that information
or run a check over explicit alternatives.

## Contract-2 (temporal specialist)

A second release covers FIFO, counter, and arbitration behavior:
[Contract-2 weights](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2).
Its public-benchmark measurements show a narrow operating envelope.
FIFO areas reach 97 of 108 and counter areas 92 of 108. Outside that
scope the model degrades sharply: local-safety areas fall to 933 of
2610 with 281 false-safe errors, transfer pairs to 275 of 621, and
access-control areas to 77 of 108. Never route local-safety,
transfer, requirement, or access-control scopes to this checkpoint
alone. These are published development measurements, not independent qualification.
The public benchmark's FIFO/counter/arbitration subset was used in Contract-2
selection.

| Focused development metric | Contract-2 |
|---|---:|
| Correct decisions | 203/234, 86.75% |
| Satisfies / violates / context correct | 64/78 / 61/78 / 78/78 |
| False-safes | 17/78 violations |
| Whole triplets correct | 51/78, 65.38% |

Commit/hold, arbiter-tree, empty-data-blanking, and stored-word-hold families
remain unreliable. The release is an owner-selected research checkpoint. It
did not pass the frozen violation-recall, false-safe, and NLL promotion gates,
and the trainer's selector retained its parent. Public release and automatic
promotion are distinct decisions.

The [router](router.md) derives tiers from each deployment's measured profile
and observed inference identity. It does not turn the historical table into
current evidence for a different runtime. Arbitration quality must also be
measured against individual experts. Agreement alone is not verification.

## Recommended use

| Use | Recommendation |
|---|---|
| Triage a direct register guard against a precise local requirement | Appropriate with review |
| Compare an explicitly guarded and unguarded local update | Appropriate with review |
| Prioritize likely violations for executable checking | Appropriate |
| Declare a design safe from the model label alone | Not appropriate |
| Reject a design automatically after a model violation | Not appropriate |
| Resolve an undefined external policy or helper module | Not appropriate |
| Certify a full IP block or chip | Not appropriate |
| Treat the largest score as a calibrated confidence | Not appropriate |
| Use the model as a general CWE classifier | Not appropriate |

For supported triage, use this sequence:

1. Supply the exact RTL region, requirement, and assumptions.
2. Confirm that all writers and relevant helpers are present.
3. Read all three model scores rather than only the winning label.
4. Compare a satisfying and violating counterfactual when possible.
5. Bind the requirement to simulation or formal checks.
6. Treat the executable counterexample and engineering review as the deciding
   evidence.

## Claims that the evidence supports

The following wording is supported:

> The evaluated Contract-1 release is strongest on explicit local RTL guards. It
> reached 86.46% accuracy and 95.83%
> violation recall with zero false-safes on a 288-decision, four-family
> counterbalanced qualification set. It classified all six curated privilege,
> clear-enable, and address-guard demo examples correctly in one matched
> batch.

The following wording is not supported:

- “The model is 100% accurate.”
- “The model verifies RTL.”
- “A 99% score means the answer is 99% likely to be correct.”
- “The model generalizes across OpenTitan.”
- “The model understands missing context.”
- “The model finds all write-path bypasses.”
- “The model is a general hardware vulnerability detector.”

## Release boundary

This checkpoint is suitable for a bounded product demonstration of local guard
analysis. It is not a generally qualified hardware-security model. A broader
release would require, at minimum:

- a frozen final evaluation on independent source families
- stronger and more varied missing-context performance
- evaluation across multi-path and cross-module update structures
- calibration or an explicit refusal policy for unfamiliar inputs
- family-clustered uncertainty over more independent families
- reproducible SDK parity for the released export
- a release model card with the exact parent, runtime, dataset, and evaluation
  identities

Until those gates are met, the honest product position is decision support for
well-scoped local contracts, followed by executable verification.

## Reproducibility notes

The aggregate measurements above were transcribed from content-addressed
receipts kept outside the public source tree. Model weights and the development
benchmark are now distributed separately through Hugging Face. The
[benchmark guide](benchmark.md) describes the pinned public population.
Private qualification/shifted-evaluation rows and complete historical logits
are not supplied by the source repository. The public benchmark cannot
reproduce every historical table above.

Published campaign/SDK scores and single-row worker scores use different
inference paths. Keep their batch, padding, runtime, question, and population
identities attached rather than mixing their counts in one comparison.

The first counterbalanced qualification attempt completed inference but failed
before writing metrics because its reporting recipe did not declare a source
mass for the authored source. The retry used the same model, data, precision,
option order, and runtime inputs with the reporting recipe bound. The failed
attempt produced no score and was not combined with the retry.

On a separate development reconciliation, the local SDK and local trainer paths
agreed on all 288 rows. A cross-machine BF16 comparison differed on one
low-margin insufficient-context row. The broader 4,095-row view agreed on every
prediction. This is a reminder that marginal decisions can change across GPU
architectures even when model and input identities match. Reproducible claims
must name the runtime, precision, question contract, and checkpoint rather than
the weights alone.

