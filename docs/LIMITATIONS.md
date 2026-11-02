# SiliconShield-RTL capabilities and limitations

[Documentation index](README.md) · [Model-family card](siliconshield-rtl.md) ·
[Evaluation protocol](evaluation.md) · [Router](router.md)

SiliconShield's decision models assess a specific RTL implementation against
an explicit requirement. Contract-1 specializes in visible local guards and
register updates. Contract-2 specializes in focused FIFO, counter, and
arbitration contracts. They supply requirement-specific evidence for engineering
review and Muse-assisted investigation.

This guide explains where each model has demonstrated useful behavior, which
errors have been observed, and how to use the results. The tables reproduce
recorded evaluations of the released checkpoints.

[Operating scope](#choose-the-decision-model-by-the-property) ·
[Local-guard examples](#explicit-local-guards) ·
[Temporal specialist](#contract-2-temporal-specialist) ·
[Failure patterns](#known-failure-patterns-and-engineering-response) ·
[Evaluation record](#contract-1-evaluation-record) ·
[Reproducibility](#reproducibility-notes)

## Choose the decision model by the property

| Decision model | Demonstrated operating scope | Review focus |
|---|---|---|
| Contract-1 | Visible privilege and lock guards, clear enables, and exact-address qualification around local writes | Include every relevant writer and the meaning of each guard. Helper logic and changed representations need additional scrutiny. |
| Contract-2 | Focused FIFO occupancy, counter updates, and arbitration contracts | Distinguish accepted operations from requests and occupancy from stored data. Commit/hold priority and arbiter-tree behavior need additional scrutiny. |

The decision labels are `satisfies_requirement`, `violates_requirement`, and
`insufficient_context`. Each applies to the supplied RTL, requirement, and
assumptions. A model assessment helps prioritize inspection. Design acceptance
requires engineering review or an executable check of the property. The three
scores describe model preference and have not been calibrated as probabilities
of correctness.

The [router](router.md) selects specialists using measured deployment profiles.
Domain membership identifies a relevant model, but individual predictions can
still be wrong. The examples below show both successful use and observed errors
within these domains.

## Measured strengths

### Explicit local guards

Contract-1 is strongest when the protected state and its enabling condition are
visible in one small RTL region. Evaluated contracts include privilege checks,
clear authorization, exact-address writes, sticky-bit clear conditions, masks,
strobes, owner-width comparisons, and default-deny branches.

On counterbalanced development, it classified 191/192 fully specified cases
correctly (99.48%). The complete three-class population, including 96
insufficient-context cases, scored 257/288 (89.24%). A separate four-family
qualification scored 249/288 (86.46%), with 92/96 violation recall and no
false-safes observed among its 96 violations. The
[evaluation record](#contract-1-evaluation-record) separates these populations
and reports the other classes and whole-triplet results.

A register-write example makes the task concrete:

> After reset, `cfg_reg` may change only on a write performed while
> `privileged` is 1. A write while `privileged` is 0 must leave `cfg_reg`
> unchanged.

The violating branch omits the privilege check:

```systemverilog
else if (wr_en)
  cfg_reg <= wdata;
```

The satisfying branch includes it:

```systemverilog
else if (wr_en && privileged)
  cfg_reg <= wdata;
```

Six authored examples form three matched pairs. Each pair keeps the requirement
fixed and changes one guard:

| Pair | Violating design | Satisfying design |
|---|---|---|
| Privilege | Write checks only `wr_en` | Write checks `wr_en && privileged` |
| Clear authorization | Clear checks only `clear_req` | Clear checks `clear_req && clear_enable` |
| Address qualification | Write accepts an address range | Write accepts only `8'hA0` |

Contract-1 classified all six correctly in one matched batch. The scores are
raw SDK outputs for that batch:

| Example | Expected decision | Winning model score |
|---|---|---:|
| Missing privilege guard | Violates | 99.9889% |
| Enforced privilege guard | Satisfies | 94.9334% |
| Omitted clear enable | Violates | 98.4911% |
| Enforced clear enable | Satisfies | 99.9990% |
| Overbroad address check | Violates | 99.9998% |
| Exact address check | Satisfies | 99.9993% |

These curated examples illustrate measured behavior. Their 6/6 outcome is not
an estimate of accuracy on unseen designs.

## Contract-2 (temporal specialist)

[Contract-2](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-2)
assesses focused FIFO, counter, and arbitration contracts. Its development
results were 97/108 for FIFO, 92/108 for counters, and 14/18 for arbitration.
Those areas form the same 234-decision population used during selection:

| Focused development metric | Contract-2 |
|---|---:|
| Correct decisions | 203/234, 86.75% |
| Satisfies / violates / context correct | 64/78 / 61/78 / 78/78 |
| False-safes | 17/78 violations |
| Whole triplets correct | 51/78, 65.38% |

These are development measurements, rather than independent qualification.
Contract-2 is an **owner-selected research release**. The candidate missed the
frozen violation-recall, false-safe, and NLL promotion gates, so the trainer
retained its parent. The publication decision did not change that selection
outcome.

Performance varies within the temporal domain. Commit/hold, arbiter-tree,
empty-data-blanking, and stored-word-hold families remain observed weaknesses.
Broader development measurements also limit its use as a generalist:

| Development area outside the focused temporal population | Correct decisions | Recorded false-safes |
|---|---:|---:|
| Local safety | 933/2,610 | 281 |
| Transfer | 275/621 | Not listed in this summary |
| Access control | 77/108 | Not listed in this summary |

Use a separately qualified specialist for these scopes. Muse arbitration can
interpret expert evidence and disagreements, but an accuracy improvement over
individual models still requires a matched evaluation.

### Contract-2 examples: distinguish occupancy from stored data

The following fragments explain the property boundaries. They are not newly
scored model examples. Assume reset has completed and the displayed branch is
the relevant writer.

For the requirement "A push while full must leave occupancy unchanged", the
update must distinguish the request from an accepted operation:

```systemverilog
if (push && !full) occupancy <= occupancy + 1'b1;
```

That branch holds occupancy when full. A separate data writer can nevertheless
violate the requirement "A rejected push must leave the stored word unchanged":

```systemverilog
if (push) stored_word <= write_data;
```

The data writer lacks the acceptance guard. Inspect it independently of the
occupancy update. Stored-word hold and empty-data blanking are among the
measured weak families, so include all relevant writers and check the exact
property before accepting the assessment.

Counter priority needs the same precision. For the requirement "A hold cycle
must preserve the count, including when commit is asserted", this ordering
violates the contract whenever `commit` changes the value during hold:

```systemverilog
if (commit) count <= next_count;
else if (hold) count <= count;
```

Another specification may intentionally give commit priority. State the required
ordering explicitly and check the simultaneous-control case. Contract-2's
commit/hold errors show that the model label alone does not resolve this case
reliably.

## Known failure patterns and engineering response

These cases distinguish errors within the intended scope from weaknesses under
changed input conditions. They are diagnostic examples, not estimates of how
often each error occurs in production RTL.

### Missing context

An `insufficient_context` label means that omitted implementation or policy
information can change whether the requirement holds. Unfamiliar syntax or low
confidence alone does not establish this condition. Contract-1 has no separately
qualified out-of-distribution detector.

Consider the requirement:

> After reset, `cfg_reg` may change only on a write performed while
> `privileged` is 1.

The supplied fragment instead uses an internal guard whose definition is
omitted:

```systemverilog
else if (wr_en && policy_allow)
  cfg_reg <= wdata;
```

Compatible definitions of `policy_allow` can make this fragment satisfy or
violate the requirement. Its reviewed label is therefore `insufficient_context`.
Contract-1 returned `satisfies_requirement` with a score of 99.9512% in the
recorded diagnostic. A complete module declaring `policy_allow` as an
unconstrained primary input is a different case: an allowed input assignment
that bypasses privilege would establish a violation.

The diagnostic screen scored 0/6 on newly authored missing-context examples.
A second screen with more familiar omission patterns scored 2/6. These small
screens are consistent with the lower context recall in the
[shift evaluation](#requirement-and-representation-shifts), but do not establish
a population error rate.

**Engineering response:** supply the policy-to-signal mapping, helper or task
definition, encoding, and observation assumptions needed by the property. If an
indirect writer or controlling definition is absent, request it before treating
a definite model label as a usable conclusion. Better phrasing cannot restore
missing implementation detail.

### Multi-path authorization and missing conjuncts

A normal write can be protected while a debug write bypasses the same rule:

```systemverilog
else if (dbg_wr)
  key_reg <= dbg_data;
else if (wr_en && privileged)
  key_reg <= wdata;
```

The diagnostic requirement applied privilege protection to both paths. The
expected decision was `violates_requirement`, but Contract-1 returned
`satisfies_requirement` with a score of 99.7409%.

**Engineering response:** enumerate every assignment to the protected state,
including debug, test, clear, and indirect paths. Review each writer against
the requirement. A correctly assessed normal path does not establish coverage
of the other writers.

### Correct lock guards can be rejected

In another diagnostic, this branch satisfied the local lock requirement:

```systemverilog
else if (wr_en && !locked)
  control_reg <= wdata;
```

Contract-1 returned `violates_requirement` with a score of 99.9550%. A case with
two write paths under one outer `if (!locked)` also received a false violation,
with a score of 98.8771%.

**Engineering response:** confirm a reported violation by inspecting the update
condition or obtaining a counterexample. These are observed false alarms within
the local-guard domain, so routing by domain alone does not eliminate them.

### Changed requirements and representations

The reserved shift evaluation changed requirement phrasing, omitted context,
and storage representation. Contract-1's accuracy and whole-triplet consistency
fell substantially, with errors across internal storage, inverted storage, and
rotate/debug updates. The tested variations include local access-control
requirements, so this is a transfer limitation within a relevant application
area.

**Engineering response:** qualify a deployment on representative RTL and
wording, including equivalent forms and satisfying/violating contrasts.
A familiar mechanism name or a high model score does not establish transfer
to a new representation. The [full results](#requirement-and-representation-shifts)
retain the counts and error directions.

## Recommended use

1. Specify one property and the state it protects, including reset and
   observation scope where relevant.
2. Supply the RTL region with all writers, relevant helper definitions, and
   the meaning of policy signals.
3. Route using a capability profile measured for the deployed checkpoint and
   inference configuration.
4. Inspect all three scores, execution errors, and expert disagreements. Keep
   the requirement and source locations with the result.
5. Confirm consequential findings through engineering review or a
   requirement-bound executable check. Include both satisfying and violating
   counterfactuals when evaluating a model or preparing a demonstration.

Muse can organize the validated specialist evidence and explain disagreements.
Expert agreement alone does not verify a property. The
[investigation workflow](releases.md#real-muse-repair) provides a separate path for
executing checks. A passing bounded simulation establishes the observed outcome
within that check, not an unbounded proof.

### Assessments are bounded by the supplied RTL

The router assesses the state supplied by its caller. It does not automatically
elaborate a repository or establish that every generated block, helper, and
module instance is present. Keep excerpt boundaries and source locations
explicit. The conclusion applies to that region and its assumptions.

### Worker token-budget truncation receives no verdict

Contract-1's evaluated limits are 1,536 tokens for the input sequence and
256 tokens for the question and options. The inference worker rejects detected
truncation without assigning a decision label. It cannot detect source removed
before the request was constructed. Integrators must preserve relevant context
and report excerpt boundaries rather than treating a short input as complete.

## Contract-1 evaluation record

### How to read the evidence

| Evidence | Interpretation |
|---|---|
| Frozen development views | Performance on populations used in model development and selection |
| Family-disjoint qualification | Transfer to four unseen authored structural families |
| Requirement and representation shift | Robustness under deliberate changes to wording, context, and RTL representation |
| Six-example authored local-guard set | Measured behavior on those selected examples |

A **false-safe** is a true violation predicted as `satisfies_requirement`.
A violation predicted as `insufficient_context` also reduces violation recall,
but does not count as a false-safe. A **whole triplet** is correct only when all
three linked satisfying, violating, and insufficient-context decisions are
correct. This measures consistency across related cases rather than accuracy
on an isolated example.

The two reserved evaluations were frozen before inference, balanced across
labels, solver-checked, and group-disjoint from their matching training and
development data. They cover authored local contracts, not a random production
hardware sample. Both have now been evaluated. Their rows, errors, and logits
remain excluded from further training and checkpoint selection.

The historical final test and unrelated sealed OpenTitan and HardSecBench
partitions were not evaluated for this study. The results below do not make a
claim about those populations.

### Counterbalanced development

| Evaluation | Correct decisions | Violation recall | False-safes | Whole triplets |
|---|---:|---:|---:|---:|
| Counterbalanced development | 257/288 (89.24%) | 95/96 (98.96%) | 0/96 | 66/96 |

The fully specified subset contains 96 satisfying and 96 violating decisions.
Contract-1 classified 96 and 95 correctly, respectively, giving 191/192
(99.48%). The full three-class result includes another 96 insufficient-context
decisions. This development population is separate from the 4,095-decision
retention views below.

### Unseen counterbalanced write-path families

This qualification contains 288 decisions in 96 complete triplets across four
unseen structural families:

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

Rows in the confusion matrix are true labels and columns are predictions:

| True \ predicted | Satisfies | Violates | Insufficient context |
|---|---:|---:|---:|
| Satisfies | 81 | 0 | 15 |
| Violates | 0 | 92 | 4 |
| Insufficient context | 16 | 4 | 76 |

Family accuracies ranged from 83.33% to 90.28%. The row-level Wilson interval
is 82.02% to 89.93%, reported descriptively because rows within a triplet and
family are correlated. It is not a family-clustered population interval. Four
families provide limited evidence about variation across new design lineages.
The observed zero false-safes applies to these 96 violations and does not
establish a zero error rate elsewhere.

### Familiar development populations

A separate frozen development population measures retained capabilities:

| Development view | Correct | False-safes | Whole triplets |
|---|---:|---:|---:|
| HardSecBench-derived | 2,601/2,610, 99.66% | 7 | 862/870, 99.08% |
| OpenTitan-derived | 180/180, 100% | 0 | 60/60, 100% |
| Authored transfer | 561/567, 98.94% | 0 | 183/189, 96.83% |
| Requirement-understanding stress | 340/738, 46.07% | 22 | 15/246, 6.10% |

Across these four views, Contract-1 scored 3,682/4,095 (89.91%), with class
recalls of 86.67% satisfies, 95.75% violates, and 87.33% insufficient context.
It recorded 29 false-safes among 1,365 violations and 1,120/1,365 correct whole
triplets. NLL was 0.6541 and 10-bin ECE was 8.40%.

The source-derived and authored-transfer views show strong retention of familiar
contracts. The requirement-understanding stress view is substantially weaker.
Reporting these views separately preserves that distinction. All four are
development evidence rather than independent generalization measurements.

### Requirement and representation shifts

The second reserved evaluation contains 756 decisions from two unseen origins.
It changes wording and RTL representation through internal storage, inverted
storage, and rotate/debug update structures:

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

The model predicted `violates_requirement` on 562/756 rows (74.34%). This
prediction bias accompanies low satisfies and context recall. Violation recall
alone therefore overstates performance across the three-way task.

| Shifted group | Accuracy | Satisfies recall | Violates recall | Insufficient recall |
|---|---:|---:|---:|---:|
| Internal storage | 45.24% | 38.10% | 85.71% | 11.90% |
| Inverted storage | 45.24% | 34.52% | 91.67% | 9.52% |
| Rotate/debug updates | 40.48% | 22.62% | 90.48% | 8.33% |

All three groups show a substantial transfer limitation. Whole-triplet accuracy
falls from 61.46% in the four-family qualification to 4.76% here. The separate
requirement-stress development view reaches 6.10%. These measurements motivate
testing related satisfying, violating, and incomplete cases together.

The row-level Wilson interval is 40.16% to 47.21%. Only two independent origin
groups contribute the 756 rows, which is insufficient for a reliable
group-clustered uncertainty interval.

### Confidence is not reliability

Correct and incorrect predictions can both receive high raw scores:

| Score diagnostic | Four-family qualification | Requirement/context shift |
|---|---:|---:|
| Mean winning score on correct decisions | 99.28% | 93.64% |
| Mean winning score on incorrect decisions | 91.62% | 92.30% |
| Errors with winning score at least 90% | 28/39 | 346/426 |
| False-safes with winning score at least 90% | No false-safes observed | 17/25 |

The shift population has little separation between the mean scores of correct
and incorrect predictions. The NLL, Brier, and ECE values in the population
tables quantify this calibration limitation. Retain all three scores in reports
and do not use a high winning score as a substitute for property verification.
No confidence threshold has been qualified here as a safe acceptance rule.

## Coverage that has not been established

The following areas have not been qualified by these evaluations. They are
coverage gaps, distinct from the measured errors above:

| Area | Scope requiring separate evidence |
|---|---|
| Design scale | Arbitrary OpenTitan IP, arbitrary HardSecBench designs, complete subsystems, full-chip and hierarchical information flow |
| Temporal and clock behavior | Liveness, long or unbounded sequences, CDC, asynchronous logic, and metastability |
| Logic and implementation semantics | Four-state `X`/`Z` behavior under two-state contracts, synthesis, place-and-route, and physical effects |
| Broader security mechanisms | Analog behavior, timing and power leakage, side channels, fault injection, and firmware/hardware interactions beyond the input |
| Additional tasks | General CWE classification, vulnerability localization, automatic root-cause identification, and repair correctness |

Training labels describe one registered property under declared assumptions.
The solver-backed data covers local, bounded, reset-conditioned contracts under
the checker's supported semantics. Related decisions can share a source design
through mutations, rewrites, and missing-context completions. Row counts are
therefore not counts of independent designs. CWE identifiers describe source
provenance rather than a separately trained classification head. Likewise,
OpenTitan-derived measurements apply to their extracted contracts, not an
OpenTitan-wide security assessment.

Extending the operating scope requires independent source families,
representative missing-context and multi-writer cases, and family-level
uncertainty estimates. A broader release should also qualify its confidence or
refusal policy, demonstrate SDK parity, and publish the corresponding parent,
runtime, dataset, and evaluation identities. The
[study walkthrough](study-walkthrough.md) describes the training and acceptance
controls for that work.

## Reproducibility notes

### Model and evaluation identity

Contract-1's measurements bind these inference settings to its released weights:

| Field | Evaluated value |
|---|---|
| Release | SiliconShield-RTL Contract-1 (C1) |
| Weights | [debjyotikm/siliconshield-rtl-contract-1](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-1) |
| Question contract | `siliconshield-requirement-context-v1` |
| Adapter metadata | 28 residual-adapter layers, rank 256, scale 8 |
| Evaluation precision | BF16 model inference, FP32 probabilities and metrics |
| Input budgets | `max_len=1536`, `head_max_len=256` for the question and options |

Changing the question contract, label order, runtime behavior, or checkpoint
changes the evaluated system. Contract-2 has its own release and measured
profiles. The [release guide](releases.md) records the loading configuration.
The router derives capability tiers from the deployed inference identity and
matching measurements rather than inheriting them from a historical table.

The recorded aggregates were reconciled against saved evaluation receipts kept
outside the source tree. Public weights and the development benchmark are
available through Hugging Face. The [benchmark guide](benchmark.md) identifies
the public population. Private qualification and shift rows and complete historical logits
are not included, so the public benchmark cannot reproduce every table here.

Campaign/SDK scoring and single-row workers can use different batch, padding,
and runtime paths. Report those settings with each result. In a development
reconciliation, the local SDK and trainer agreed on all 288 rows. A comparison
across GPU architectures using BF16 differed on one low-margin context row.
The broader 4,095-row view agreed on every prediction. Marginal differences
therefore need to be assessed under matched inference conditions.

<details>
<summary>Qualification reporting provenance</summary>

The first counterbalanced qualification attempt completed inference but could
not write metrics because its reporting recipe omitted a source mass for the
authored source. The retry kept the model, data, precision, option order, and
runtime inputs unchanged and supplied the reporting recipe. The first attempt
produced no score and was not combined with the retry.

</details>
