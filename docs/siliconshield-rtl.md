# SiliconShield-RTL model card

Documentation index

## 1. What SiliconShield-RTL does

SiliconShield-RTL Contract-1 (C1) is a fast and local decision
model for specification conditioned RTL security assessment. It
gives hardware engineers a focused triage signal before simulation
plus formal verification plus engineering review.

Given a self contained RTL region plus an explicit security
requirement plus stated execution assumptions, the model produces
one of three structured answers. It runs locally and keeps design
inputs on user hardware and turns an RTL region plus its
requirement into a compact decision that downstream verification
can act on.

| Label | Description |
|---|---|
| `satisfies_requirement` | The model predicts that the supplied RTL region looks consistent with the stated requirement under the supplied assumptions. |
| `violates_requirement` | The model predicts a possible inconsistency between the RTL and the requirement. |
| `insufficient_context` | The model predicts that omitted implementation detail or assumption or policy information prevents a reliable determination. |

Ground truth for training used solver backed construction with
opposing verified completions for missing context. That method
applies to dataset labels. It does not transfer to predictions at
inference time. A conditional path alone is not missing context.

## 2. Why it matters

SiliconShield-RTL occupies the space between generic code analysis
and full verification. It offers requirement aware triage where
the same implementation can receive different correct answers
under different requirements.

This combination helps teams look in the right place sooner while
keeping consequential findings subject to tool checks.

## 3. Headline results

> SiliconShield-RTL is a fast and locally executable decision
> model for specification conditioned RTL security assessment. It
> analyzes an RTL region together with its security requirement
> and execution assumptions and returns a structured triage
> decision. On unseen counterbalanced development families,
> Contract-1 correctly classified 191 of 192 fully specified
> designs including 95 of 96 violations with no false safe
> predictions observed. It helps hardware engineers prioritize
> simulation plus formal verification plus review while keeping
> sensitive designs on local hardware.

Directly measured headline results are listed below. Full matrices
plus family breakdowns are in capabilities and limitations.

| Metric | Contract-1 |
|---|---:|
| Fully specified decisions | 191 of 192 at 99.48 percent |
| Satisfying cases | 96 of 96 at 100 percent |
| Violating cases | 95 of 96 at 98.96 percent |
| False safe violations | 0 of 96 |
| Full three way accuracy | 257 of 288 at 89.24 percent |

Contract-1 is good at focused local triage where guards are
visible in the supplied region. It shines on privilege gated
writes plus clear with request plus enable plus exact address
match plus lock and authorization checks with explicit reset and
sampling. It keeps identical RTL distinct across requirements and
helps teams prioritize simulation plus formal checks plus review
while keeping sensitive designs on local hardware.

Retained development capability shows strong preservation on
familiar source and contract styles and complements the unseen
qualification story. Full matrices plus family breakdowns are in
capabilities and limitations.

## 4. What it handles well

Contract-1 is strongest when the requirement names a visible state
element or write path and relevant guards are present in the RTL
and assumptions define reset plus sampling precisely.

Representative strengths include lock gated write paths plus
privileged or authorized controls before register updates plus
explicit reset exemption from hold requirements plus distinct
treatment of normal and debug paths plus named targets under
stated control combinations. Counterbalanced construction means a
code only shortcut cannot solve the task. The model must combine
target plus requirement plus assumptions plus visible
implementation on unseen structural families.

## 5. Quickstart and examples

The commands below use only files in this repository plus local
tooling. They run the authored verification path without weights
and then connect a trained export when one is available.

Requires Python 3.11 or newer plus Verilator plus a C++ toolchain.

```bash
python -m pip install -e .
python -m siliconshield.app --list
python -m siliconshield.app --case unauthorized-write \
  --output /tmp/siliconshield-demo
```

Expected conclusion for the authored demo is
`demonstrated_violation_fixed` with `repair_accepted` true. Read
`/tmp/siliconshield-demo/report.md` for the narrative and
`report.json` for inputs plus assessment plus checks plus patch
plus conclusion. Without a configured checkpoint the report marks
the model as unavailable and still exercises the workflow.

Connect a trained export with explicit environment settings.

```bash
export SILICONSHIELD_SECURITY_PYTHON=/path/to/security-env/bin/python
export SILICONSHIELD_SECURITY_CHECKPOINT=/path/to/selected-export
export LAYA_ROOT=/path/to/compatible-runtime
python -m siliconshield.app --case unauthorized-write \
  --output /tmp/siliconshield-demo-scored
```

The assessor resolves the checkpoint path and hashes its full
contents before starting the worker and checks those contents again
after execution. An unreadable or changed checkpoint returns an
error without a judgment. Keep export generations immutable while
assessments run. Inputs that do not fit the configured budgets are
rejected as truncated and never given a verdict.

See investigation for custom RTL inputs and
setup for environment separation. Training plus
controller operation plus inference use separate environments.
Curated GUI demos for privilege plus clear enable plus address
guard illustrate the focused task. They show measured strength and
are not an accuracy estimate.

## 6. How it was trained

SiliconShield-RTL starts from a LAYA checkpoint and adds residual
adapters plus security specific training. The trainer and the
inference worker run on the LAYA runtime as an external dependency
through `LAYA_ROOT` or an installed package. Architecture is a
large encoder with residual bottleneck adapters on all encoder
layers plus a three option decision head. Forward computation uses
BF16 with trainable parameters and optimizer state held at higher
precision.

Training lineage moves from broad security adaptation through
transfer continuation plus joint preservation plus OpenTitan
expansion to a compact counterbalanced specialist. Training data
combines source backed contracts with solver evidence plus authored
semantic scenarios plus guarded mutations. The direct training rows
for Contract-2 and the Contract-1 anchor population plus its
follow-up study mixture are published as the [SiliconShield RTL
training set](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-training).
See [training recipes](training-recipes.md) for which rows each
release actually saw.
Catalogs plus recipes plus full run datasets plus checkpoints live
outside this repository.
`examples/` shows formats only. Source families and their mutations
share a partition. Continuing from a test touched checkpoint needs
a new study manifest that retires old evaluation populations.

Checkpoint selection uses development data only. Calibration and
final evaluation run separately against the exported checkpoint.
Reports keep unweighted and weighted views plus source and mechanism
breakdowns plus source stratified bootstrap intervals over whole
split groups where supported. Exact dataset bytes plus manifests
plus receipts remain outside the public tree per the source only
rule.

Populations are balanced across the three labels and arranged in
linked triplets. Each triplet tests a satisfying plus a violating
plus an insufficient context form of one situation. Solver evidence
backs training labels for local contracts. The model at inference
time does not run that solver. Reported metrics include accuracy
plus class recall plus false safe count with violation denominator
plus whole triplet accuracy plus NLL plus multiclass Brier plus ECE.
Calibration uses raw scores with no temperature scaling.

## 7. Reproducibility and provenance

Four claims are distinguished.

Artifact integrity means published files carry verified checksums.
Inference reproducibility means the published runtime plus
checkpoint reproduce expected predictions under pinned settings.
Metric reproducibility means aggregates can be recomputed from
retained receipts. Independent evaluation reproducibility means
another researcher can rerun the actual cases and checks.

This repository supports the first three through content hashes and
pinned environments and retained evaluation code. It does not by
itself establish the fourth while full rows and receipts stay
private. A practical compromise is the public schema plus metric
code plus split method plus family counts plus summary manifests
plus a small licensed subset without exposing private rows.

| Field | Evaluated value |
|---|---|
| Release | SiliconShield-RTL Contract-1 (C1) |
| Weights SHA-256 | `f4130223ed19be5cc7e21b3da70cb7aebf944ad6ed4a8a80f11d8f59bcd05ee7` |
| Question contract | `siliconshield-requirement-context-v1` |
| Contract file SHA-256 | `dd1f38ee223abac1f75b19d9086fd26d90865c9aa43097674fffdd61dbbf8bd3` |
| Adapter metadata | 28 residual-adapter layers, rank 256, scale 8 |
| Evaluation precision | BF16 model inference, FP32 probabilities and metrics |
| Input budgets | `max_len=1536`, `head_max_len=256` for question and options |
| Weights | [danieltombay/siliconshield-rtl-contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1) |
| Area benchmark | [danieltombay/siliconshield-rtl-benchmark](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-benchmark) |
| Public training rows | [danieltombay/siliconshield-rtl-training](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-training) |

Model bytes plus question contract plus runtime behavior plus label
order are one identity. Loading the same weights with a different
contract is a different system and outside reported evidence.
Marginal decisions can differ across hardware even when identities
match, so reproducible claims must name runtime plus precision plus
contract plus checkpoint and not weights alone.

## 8. Brief operating boundary

Contract-1 brings clear triage value inside its focused envelope
and defines valuable next milestones outside it. Insufficient
context handling invites improved abstention when hidden policy or
helper detail can change the answer. Multi path authorization
rewards structural enumeration of every writer through relevant
statements plus checks. Correct lock precision offers headroom for
fewer false alarms while keeping strong violation recall.
Representation breadth across storage forms plus helper abstractions
plus equivalent rewrites maps where added variation will help most.
Confidence calibration can turn strong raw scores into more useful
thresholds through a new study on a separate population.

The current evidence supports triage use with verification. Signing
off an RTL block without executable verification plus unattended
acceptance of safety critical hardware plus replacement of formal
verification or lint or expert review plus broad discovery across
arbitrary repositories plus CWE classification plus vulnerability
localization plus calibrated risk scores plus automatic patch
approval remain out of scope. Training on sealed evaluation outcomes
is prohibited. Sealed populations stay sealed.

## 9. Detailed limitations

Full confusion matrices plus family breakdowns plus confidence
behavior plus failure examples plus supported and unsupported
claims are openly maintained in capabilities and limitations. That file is the
authoritative boundary record and should be read alongside any
headline figure.

## 10. License and attribution

This repository's code is released under the Apache License,
Version 2.0 (see LICENSE), and the RTL benchmark dataset is
likewise offered under Apache-2.0 (see NOTICE). Model weights,
downloaded datasets, checkpoints, and transcripts stay outside
this repository.

Model weights are licensed separately from the code: each
Hugging Face release page states its own license. The LAYA
foundation weights retain their upstream terms at the pinned
revision recorded in the repository, and upstream source terms
(including OpenTitan and HardSecBench sources where used)
continue to apply to their materials. Review the pinned
upstream licenses and the release-page terms before
redistributing datasets, source excerpts, or model files.

## Evaluation status

Repository gates are unit plus fixture integration plus Verilator
backed checks on authored synthetic cases. GPU execution plus
serving plus model quality evaluation need separate validation with
the pinned environments in setup. State the result type
with every claim.
