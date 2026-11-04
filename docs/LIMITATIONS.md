# Tested domains and operating scope

[Documentation index](README.md) · [Router](router.md) ·
[Writing RTL requirements](rtl-requirements.md) · [Research paper](whitepaper.md)

SiliconShield uses Muse to investigate RTL against explicit requirements.
Its strongest evidence concerns small digital-design regions where the relevant
state, update paths, and execution assumptions are available together. This
page describes what has been exercised, what each specialist is intended to
review, and which extensions still need evaluation.

An area marked **not evaluated** is an open measurement boundary. It is not a
claim that Muse cannot reason about that area. Likewise, a skill's review
instructions describe its intended scope, not an accuracy guarantee for every
mechanism named in those instructions.

## Three distinct sources of evidence

| Component | Role | Evidence to use |
|---|---|---|
| Muse Glimmer with Contract-1 and Contract-2 skills | The shipped specialist path. Versioned review procedures guide Glimmer's analysis and structured response. | Live skill assessments, deployment-specific capability profiles, and router integration checks. |
| Fine-tuned Muse Glimmer RTL adapter | A separately published research decision model covering local guards and focused temporal contracts. | Its frozen-checkpoint development measurements, authored demonstrations, and reload checks. |
| Earlier retained decision models | Historical weight-based specialists used in the research leading to this project. | Their own model cards and historical evaluation records. Their scores do not measure the current Glimmer skills. |

The fine-tuned adapter and the Contract skills are different interfaces. The
adapter scores three answer tokens. The skills ask Glimmer to reason through a
review procedure and return a verdict, confidence, evidence, and open questions.
The adapter's measurements do not automatically qualify the skill path, and
successful skill calls do not validate the adapter.

## Contract-1 skill: local access-control review

[Contract-1](../.agents/skills/contract-1/SKILL.md) organizes review around the
writers and guards of a protected register or state element. It directs
Glimmer to enumerate update paths, trace authorization conditions, inspect
branch priority, and identify a concrete triggering case for a violation.

| Intended review domain | What the requirement should identify |
|---|---|
| Privilege-qualified register writes | The protected register and the privilege condition that must hold for an update. |
| Lock protection | Which updates are prohibited while locked, including debug and test paths. |
| Clear authorization | The clear request and the permission required to act on it. |
| Address qualification | The address or range allowed to update the target register. |
| Write strobes and masks | Which bits may change and under which authorization. |
| Owner checks and default-deny behavior | The owner comparison, relevant widths, and behavior when no authorized case matches. |

**Recorded live evidence.** The browser-backend verification consulted the
Contract-1 skill through a real Glimmer service for an authored guarded-register
case. It returned `satisfies_requirement` with validated skill identity and a
self-reported confidence of 0.95. The [demo evidence](web-demo.md#recorded-evidence)
identifies the scope of that check. It establishes a working assessment path on
that example, not a domain-wide accuracy estimate.

The skill also includes review instructions for multiple writers and hidden
helpers. That procedural coverage is useful, but no complete per-family quality
evaluation for the current skill deployment is reported here. In particular,
large helper graphs, indirect storage representations, and extensive debug or
test logic require representative measurements before carrying an established
reliability claim.

## Contract-2 skill: focused state-transition review

[Contract-2](../.agents/skills/contract-2/SKILL.md) organizes review around the
transition that occurs for each input combination constrained by a requirement.
It distinguishes a requested operation from an accepted operation and asks
Glimmer to check which branch governs simultaneous controls.

| Intended review domain | What to inspect |
|---|---|
| FIFO occupancy and backpressure | Push/pop acceptance, full/empty behavior, and the resulting occupancy update. |
| Stored-word capture and blanking | Data-path updates separately from occupancy and flag updates. |
| Counters | Increment, decrement, load, clear, saturation, and wrap conditions. |
| Commit and hold priority | The specified outcome when controls are asserted together. |
| Arbitration and grant handshakes | Request eligibility, grant acceptance, overlap, and release conditions. |
| Local sequencing | The visible state transitions and observation edges covered by the property. |

The repository provides authored temporal examples and a structured review
procedure. This document does not assign the historical Contract-2 model's
accuracy, or the fine-tuned adapter's accuracy, to the current Contract-2 skill.
A deployed skill needs its own measured capability profile.

The intended scope is focused transition reasoning. It does not establish
unbounded liveness, arbitrary protocol correctness, fairness over every possible
execution, or correctness of an entire FIFO simply because one occupancy
requirement is satisfied.

## Fine-tuned Muse Glimmer: measured domains

The separately released
[SiliconShield Muse Glimmer RTL adapter](https://huggingface.co/debjyotikm/siliconshield-muse-glimmer-rtl-lora)
was trained on accepted local-guard and focused temporal examples. Its frozen
checkpoint was evaluated on 288 local-guard decisions and 234 temporal
decisions. These are existing development populations, not a new sealed test.

| Evaluated population | Correct decisions | What the measurement covers |
|---|---:|---|
| Counterbalanced local guards | 280/288, 97.22% | Small write-path contracts assessed against explicit requirements, including satisfying, violating, and omitted-context cases. |
| Focused temporal contracts | 223/234, 95.30% | The selected FIFO, counter, and arbitration population under its stated assumptions. |

The model completed 88/96 local-guard triplets and 67/78 temporal triplets
correctly. A triplet is credited only when all three linked labels are correct.
This measures consistency across related requirements or design variants.

The remaining errors define useful follow-up work:

- Local guards: five satisfying cases and three violations were classified as
  insufficient context. No violation was classified as satisfying in this
  96-violation population.
- Focused temporal contracts: three satisfying cases were classified as
  violating. Five violations were classified as satisfying and three as
  insufficient context.
- All 174 omitted-context development cases were classified correctly. Broader
  omission patterns still need independent evaluation.

These counts describe the evaluated samples. They do not establish zero risk
for local guards or universal context detection. Detailed results are retained
in the [release evaluation record](https://huggingface.co/debjyotikm/siliconshield-muse-glimmer-rtl-lora/blob/main/evaluation.json)
and [research paper](whitepaper.md).

### Authored examples exercised with the frozen adapter

| Example | Requirement focus | Recorded result |
|---|---|---|
| Privilege guard present | Every non-reset write requires privilege. | Correct satisfying prediction. |
| Privilege guard removed | An unprivileged write must not change the register. | Correct violating prediction. |
| Authorization helper omitted | The missing definition can change whether the write is authorized. | Correct insufficient-context prediction. |
| Simultaneous push/pop holds | Both requests together must preserve the count. | Correct satisfying prediction. |
| Simultaneous push/pop increments | An explicit increment on both requests breaks the hold requirement. | Correct violating prediction. |
| Counter-update helper omitted | The missing definition can change the specified transition. | Correct insufficient-context prediction. |

All six passed. They are authored demonstrations, not an unseen benchmark or
an estimate of the frequency of errors in production RTL.

### Areas without qualification in this adapter study

| Area | What remains to be established |
|---|---|
| New design lineages and substantially different requirement wording | Independent group-disjoint evaluation beyond the reused development populations. |
| Large modules and cross-module behavior | Whether relevant context can be assembled and assessed without losing dependencies. |
| Deep helper chains, indirect writers, and encoded storage | Transfer across those representations rather than reliance on familiar local forms. |
| Long temporal sequences | Multi-cycle protocol properties, eventual responses, fairness, and liveness beyond the bounded local contracts. |
| Clock-domain and reset-domain crossings | Synchronizer behavior, metastability assumptions, asynchronous interactions, and cross-domain ordering. |
| Four-state semantics | X/Z propagation and simulator-dependent unknown-value behavior. The task data uses declared two-state assumptions. |
| Complete security subsystems | End-to-end access control, lifecycle transitions, cryptographic implementations, and hardware/firmware interactions. |
| Physical behavior | Timing closure, power or electromagnetic leakage, analog effects, and physical fault injection. |
| Additional output tasks | CWE classification, vulnerability localization, generated-repair correctness, and calibrated risk estimates. |

OpenTitan-derived contracts are evidence about those extracted behaviors. They
are not an assessment of OpenTitan as a whole. Similarly, a source category or
CWE provenance tag does not establish a separately trained classification task.

## Requirements determine the assessment boundary

A useful assessment begins with a precise property, not a request to declare a
design generally secure. State the protected object, relevant event, and
permitted behavior. Include the assumptions that affect the answer.

For example, this branch protects a register against unprivileged writes:

```systemverilog
else if (wr_en && privileged)
  cfg_reg <= wdata;
```

For the requirement “Outside reset, `cfg_reg` may change only when `privileged`
is asserted,” this branch is consistent with the policy if it is the relevant
non-reset writer. A second debug path must be checked separately. If the
requirement also covers reset, the reset assignment needs to satisfy it too.
That change in scope can change the correct answer without changing the RTL.

For a FIFO, “a push while full must preserve occupancy” is different from
“a rejected push must preserve the stored word.” A correct count update does
not settle the data-path requirement. Similarly, checking behavior when an
input named `full` is asserted does not establish that the logic generating
`full` is correct.

A missing helper definition matters only when it can change the property.
A complete design with an unconstrained primary input is different from an
excerpt with omitted implementation. Unfamiliar code, low confidence, an
execution failure, and semantic insufficient context are distinct conditions.
See [Writing requirements for RTL analysis](rtl-requirements.md) for examples.

## What the router and Muse analysis establish

The router uses caller-supplied scope and measured capability profiles to select
relevant specialists. Muse can examine their assessments alongside the RTL,
explain disagreements, and identify the information or checks needed next.
This makes the review traceable to a requirement and implementation region.

The [router guide](router.md#driving-assessments-from-glimmer) records live
Glimmer-driven assessment sessions. Fixture checks exercise routing, response
validation, identity mismatches, malformed verdicts, and recovery. These are
valuable integration results. They do not establish an accuracy advantage over
the best individual specialist or unattended reliability across arbitrary RTL.

Two consultations of the same Glimmer model under different skills are not
independent model votes. Agreement can help organize an investigation, but it
does not multiply confidence or verify the property. Disagreements and weak
coverage should remain visible in the resulting assessment.

The router does not automatically elaborate a whole repository or establish
that every relevant generated block and helper is included. Lexical inspection
can assist scope selection. It does not replace elaboration or supply missing
implementation detail.

## Scores, context budgets, and deployment identity

The two current interfaces report different kinds of scores:

- **Skill-assisted Glimmer:** confidence is supplied by the model in its
  structured response. The transport assigns that value to the selected label
  and splits the remaining mass between the other two labels. This is not a
  measured three-class token distribution.
- **Fine-tuned adapter:** scores come from normalization over the three
  registered answer-token logits. They express relative model preference under
  that prompt, not calibrated correctness probabilities.

Preserve this distinction in integrations. Neither interface has a measured
calibration guarantee for arbitrary engineering inputs.

The published adapter loader rejects inputs exceeding its 2,048-token budget.
The tested skill-serving configuration has its own context and generation
budgets. Its preflight uses an input-size estimate, so it should not be described
as exact tokenizer verification. Neither interface can establish that a caller
has supplied all relevant RTL merely because the input fits.

The adapter's tested loader uses BF16 across three CUDA GPUs. Other placements,
quantization, and CPU-only execution require separate qualification. The skill
path uses the configured Glimmer serving endpoint and pinned skill content.
Changing model weights, skill text, prompt, precision, or serving configuration
can change behavior and requires corresponding capability measurements.

Fresh loading of the frozen adapter reproduced all 144 monitor decisions.
Probability parity did not meet the registered tolerance on 22 rows, with a
maximum absolute score difference of approximately 0.031. The saved adapter
weights matched exactly. This remains a numerical reproducibility question,
not evidence of changed labels on that check.

## Extending coverage

For a new design domain, assemble representative requirements and RTL with
reviewed satisfying, violating, and genuinely missing-context cases. Keep
related designs in the same split, reserve independent families before
inspection, and measure the actual model-plus-skill or adapter deployment.

Report class recall and error direction alongside overall accuracy. Check
simultaneous controls, every relevant writer, and equivalent RTL forms. Use
whole triplets to test consistency and retain the source locations and
assumptions behind each result. These steps turn a plausible new capability
into measured operating scope.

Consequential findings should be checked through engineering review and
appropriate simulation or formal analysis. A neural verdict prioritizes that
work. A bounded simulation establishes what its check exercised, while a
formal result applies under its property and environment assumptions.

## Historical decision-model evidence

The retained research checkpoints named Contract-1 and Contract-2 preceded the
current Glimmer skills. Their results remain useful for the research record,
but must not be presented as measurements of Muse Glimmer or of skill-guided
inference.

| Historical checkpoint / population | Correct decisions | False-safe violations | Whole triplets |
|---|---:|---:|---:|
| Contract-1 counterbalanced development | 257/288 | 0/96 | 66/96 |
| Contract-1 four-family qualification | 249/288 | 0/96 | 59/96 |
| Contract-2 focused temporal development | 203/234 | 17/78 | 51/78 |

The earlier Contract-2 release was owner-selected after its automated promotion
gates failed. That outcome belongs to the earlier checkpoint and is not a
judgment about the current temporal skill or fine-tuned Glimmer adapter.
The [historical model-family card](siliconshield-rtl.md) retains those identities
and publication boundaries.

### Requirement and representation shifts

The historical Contract-1 checkpoint was also evaluated on 756 authored
requirement and representation shifts. It scored 330/756, with 25/252
false-safe violations and 12/252 complete triplets. Its satisfies, violates,
and context recalls were 80/252, 225/252, and 25/252 respectively.

Those measurements concern that earlier model under deliberately changed
inputs. They have not been reproduced as a qualification of the current
Glimmer skills or the fine-tuned Glimmer adapter. The relevant next step for
those systems is their own independent shift evaluation, not inheriting either
the older model's strengths or its errors.
