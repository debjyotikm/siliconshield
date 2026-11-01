# SiliconShield-RTL decision models: Contract-1 and Contract-2

[Documentation index](README.md) · [Router](router.md) ·
[Training recipes](training-recipes.md) · [Capabilities and limitations](LIMITATIONS.md)

SiliconShield-RTL is a family of local decision models for assessing RTL against
explicit requirements. Verification always judges an implementation against an
intended policy. These models take both as input: an RTL region, an English
requirement, and the assumptions that bound it. The same RTL can then receive
different correct labels under different policies, which is exactly how a
protection review works. Contract-1 focuses on visible local guards.
Contract-2 adds a specialist for FIFO, counter, and arbitration behavior.
Both return the same three-way decision and all three model scores.

## Releases and datasets

| Artifact | Purpose |
|---|---|
| [Contract-1 decision model](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1) | Local guard, register-write, clear-enable, and exact-address triage |
| [Contract-2 decision model](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2) | Focused temporal triage for FIFO, counter, and arbitration contracts |
| [RTL training collection](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-training) | 19,050 rows, with direct-training and follow-up-study provenance |
| [RTL benchmark](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-benchmark) | 5,589 development decisions across 11 design areas |

The models run on the compatible LAYA runtime. SiliconShield supplies the
specialized training, input contracts, worker integration, capability routing,
and evaluation tools. Weights and datasets are distributed on Hugging Face,
separately from this source-only repository.

## What a decision means

| Label | Meaning |
|---|---|
| `satisfies_requirement` | The model predicts that the supplied implementation meets the stated requirement under the supplied assumptions. |
| `violates_requirement` | The model predicts an inconsistency between the implementation and that requirement. |
| `insufficient_context` | The model predicts that omitted implementation or policy information can change the answer. |

A model decision prioritizes a requirement for review. It carries neither a
simulation trace nor a proof result. A predicted violation should lead to
inspection of the relevant update path and, where available, a requirement-bound
check. A predicted satisfaction applies only to the supplied region and policy.

Missing context has a specific semantic meaning: an omitted implementation or
policy definition can change the correct answer. A fully specified module with
an unconstrained primary input is a different case. If an allowed input value
violates the requirement, that is a violation under those assumptions.
Low confidence and unfamiliar syntax do not establish missing context. The
worker reports truncated input as an execution error, without assigning a label.

Training labels use registered solver-backed contracts. A satisfies label
requires an established proof over the extracted local model. A violates label
requires a reachable counterexample. A missing-context label requires compatible
completions with opposing established outcomes. Inference does not rerun
those solvers. Use the [investigation workflow](investigation.md)
when independent executable checks are available.

## Contract-1: local guard analysis

Contract-1 is strongest when the target state and update condition are visible
in a small RTL region. Its domain is the combinational guard logic that
verification engineers review first: privilege conjuncts on register writes,
lock-bit checks, clear request paired with clear enable, exact-address decode
windows, sticky bits with declared clear conditions, owner-width comparisons,
and default-deny branches. Six curated local-guard examples cover missing and
enforced privilege guards, clear authorization, and exact-address qualification.
The examples demonstrate measured behavior, not a population accuracy estimate.

| Evaluation | Correct decisions | Violation recall | False-safes | Whole triplets |
|---|---:|---:|---:|---:|
| Counterbalanced development | 257/288 (89.24%) | 95/96 (98.96%) | 0/96 | 66/96 |
| Four-family counterbalanced qualification | 249/288 (86.46%) | 92/96 (95.83%) | 0/96 | 59/96 |
| Shifted requirement/context evaluation | 330/756 (43.65%) | 225/252 (89.29%) | 25/252 | 12/252 |

The development result includes 191/192 correct fully specified decisions:
96 satisfying and 95 violating cases. The full three-class result also includes
insufficient-context cases. Report both when quoting that strength.

The qualification result supports transfer to unseen implementations of familiar
local contracts across four authored families. Broader wording and storage
representations expose a substantial gap. Multi-writer authorization, hidden
policy, helper logic, and unfamiliar encodings remain unreliable. Some correct
lock guards also produce false alarms. See the concrete examples and confusion
matrices in [LIMITATIONS.md](LIMITATIONS.md).

## Contract-2: focused temporal analysis

Contract-2 targets small FIFO, counter, and arbitration fragments under one
requirement. The evaluated population covers occupancy updates, backpressure,
empty/full boundaries, clear behavior, counter saturation, sampled steps,
commit/hold priority, and grant handshakes. Those categories describe evaluation
coverage, not uniform capability: stored-word hold, empty-data blanking,
commit/hold, and arbiter-tree cases remain difficult. The focused development
result was:

| Metric | Result |
|---|---:|
| Correct decisions | 203/234 (86.75%) |
| Satisfies recall | 64/78 (82.05%) |
| Violates recall | 61/78 (78.21%) |
| Insufficient-context recall | 78/78 (100%) |
| False-safes | 17/78 violations |
| Whole triplets correct | 51/78 (65.38%) |

Published area measurements are 97/108 for FIFO, 92/108 for counters, and 14/18
for arbitration. These total the same 234-row development population used for
selection. They are not independent qualification.

Contract-2 is an **owner-selected research release**. The trainer retained its
parent under the frozen gates because this candidate missed violation-recall,
false-safe, and NLL limits. The release preserves that decision history rather
than describing the candidate as an automatic promotion.

Commit/hold, arbiter-tree, empty-data-blanking, and stored-word-hold families
remain difficult. Outside the focused domain, Contract-2 performs poorly on
local safety, transfer, and access-control requirements. It complements
Contract-1 through scoped routing, not through a universal replacement.

## Using the models together

The [router](router.md) consults models whose profiles cover the requested
scope and orders them by measured reliability. A profile must bind its counts
to the same checkpoint, question, runtime, and inference settings. Missing or
mismatched measurements stay unrated. Unsupported or weakly supported scopes
remain visible as thin evidence.

Muse can interpret the resulting evidence and produce a structured conclusion
with dissent and open questions. That conclusion remains a model assessment.
A shared mistake is possible even when experts agree.

Multi-expert consultation and arbitration are available through the experts
CLI and Python API.
The [qualification guide](expert-qualification.md) describes how to add future
models or retain an older release for rollback.

## Run locally

Prepare a compatible Torch/LAYA inference environment and download an immutable
export generation from the model's Hub repository. Record the Hub revision and
verify the export manifest before registering it. Follow the [pinned release setup](releases.md).

Create model entries and attach measured capability profiles using
[expert qualification](expert-qualification.md), then follow the
[router assessment walkthrough](router.md#run-one-assessment). Each release has
its own export, profile, and execution identity. The router consults relevant
specialists and optionally requests Muse arbitration over their evidence.

For executable investigation, configure a selected export and run an authored
case:

```bash
export SILICONSHIELD_SECURITY_PYTHON=/path/to/security-environment/bin/python
export SILICONSHIELD_SECURITY_CHECKPOINT=/path/to/contract-1-export
export LAYA_ROOT=/path/to/pinned-runtime
python -m siliconshield.app --case unauthorized-write \
  --output /tmp/siliconshield-demo-scored
```

This path requires Verilator and a C++ build toolchain. Its checks and default
repair are authored. Add a qualified Muse endpoint for a model-proposed repair.
A custom design receives no inherited testbench.

## Training lineage and reproducibility

Both releases derive from LAYA with residual adaptation and a security decision
head. Their ancestors and optimization histories differ. Contract-1 continues
broad security and OpenTitan training into a compact counterbalanced specialist.
Contract-2 follows a separate fresh-encoder study, focused temporal adaptation,
and triplet continuation. It is not fine-tuned from Contract-1. The
[white-paper ancestry diagram](whitepaper.md#6-model-ancestry-and-reproducible-recipes)
shows the two paths.

The published training collection has two subsets:

| Subset | Rows | Relationship to the release |
|---|---:|---|
| `contract-1-direct` | 16,350 | Includes the 576-row Contract-1 anchor and 15,774 follow-up-study rows. Those follow-up arms did not replace Contract-1. |
| `contract-2-direct` | 2,700 | Training population used in the Contract-2 continuation |

These are not the full ancestor corpora. Registered rows and delivered exposure
at the selected update are also different quantities. Consult
[training recipes](training-recipes.md) for schedules, sampling, selection,
and the boundary between shipped trainer features and versioned study code.

The public benchmark supports reproducible development comparison through
`python -m siliconshield.benchmark`. It does not replace the private,
group-disjoint qualification populations behind every historical result.
The single-row worker path can differ from batch-16 campaign scoring. Match
inference settings before comparing counts.

## Identity

| Field | Contract-1 | Contract-2 |
|---|---|---|
| Weights identity | `f4130223ed19be5cc7e21b3da70cb7aebf944ad6ed4a8a80f11d8f59bcd05ee7` | `f916444ff45cfc67bc5a67c81a36a770a20709fcd7a85fbfc0258fb411144ed8` |
| Question contract | `siliconshield-requirement-context-v1` | `siliconshield-requirement-context-v1` |
| Model computation | BF16 in the reported evaluations | BF16 in the reported evaluations |
| Input budgets | `max_len=1536`, `head_max_len=256` | `max_len=1536`, `head_max_len=256` |

See [training recipes](training-recipes.md) for training lineage.

Contract-1's question-record SHA-256 is
`dd1f38ee223abac1f75b19d9086fd26d90865c9aa43097674fffdd61dbbf8bd3`.
Its export records 28 residual-adapter layers, rank 256, and scale 8.
A shortened fingerprint is a documentation aid, never a substitute for a full
hash check. The worker binds weights, export contents, question bytes, runtime,
precision, label order, and input coverage into the executed assessment.

## Intended use and release terms

Use the family for requirement-conditioned triage, comparisons of local design
variants, and research on specialist routing. Confirm consequential findings
through executable checks or engineering review. General CWE classification,
whole-chip verification, calibrated risk scores, and automatic repair approval
are not established capabilities.

Repository code is [Apache-2.0](../LICENSE). Model weights and datasets have
separate release terms on their Hub pages, with upstream terms retained.
The owner has selected Apache-2.0 for Contract-1, consistent with Contract-2.
The pinned older Contract-1 Hub card still contains pending-license text. Its
correction is prepared separately and must be published before treating Hub
metadata as synchronized. See [release terms](releases.md#release-terms) and
[NOTICE.md](../NOTICE.md).

The [limitations](LIMITATIONS.md) record contains examples of both strengths
and failures. [Evaluation](evaluation.md) distinguishes development selection,
independent qualification, infrastructure checks, and deployment acceptance.
