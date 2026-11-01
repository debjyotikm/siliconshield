# Decision-model studies

[Documentation index](README.md) · [Training](security-model-training.md) ·
[Campaigns](campaigns.md) · [Release recipes](training-recipes.md)

A reproducible path from a user-supplied dataset to an improved
model or a justified new one. Everything a run depends on is
frozen in one study manifest before training starts, and the
supervisor enforces the frozen plan and budget.

New studies use `siliconshield-decision-study-v2`. Version 1
records are historical evidence and must be re-frozen before
using the enforced launch path. Existing jobs without a study
binding retain their legacy behavior.

## Study lifecycle

The [campaign lifecycle specification](campaigns.md#round-lifecycle) defines
persisted states, transition work, waiting, and recovery. Figure 03 separates
owned execution from between-round agent analysis. This study guide defines
the immutable inputs and qualifications that must hold across that lifecycle.

Selection inside a run and promotion into a campaign are distinct operations.
The initial checkpoint may remain selected. An accepted candidate's reserved
evaluation is a separate release protocol.

## Dataset format

The user supplies JSONL rows. Each row carries an `id`, a `state`
with `REQUIREMENT`, `ASSUMPTIONS`, and `RTL` sections, a
`template_id`, and an `assertion_sha256`. Rows group into
decision triplets (satisfies, violates, insufficient context)
that share one situation. Related rows share a `split_group_id`
so lineage never crosses partitions. Four splits are registered:
`train`, `dev`, plus reserved `qualification` and `test`
populations that training never opens.

A training-only pilot may instead declare an unbuilt reserved split
as `{"status": "not_materialized", "reason": "..."}`. It must not
invent a dataset hash. Such a study cannot claim qualification or
release readiness until independent evaluation is constructed and
registered separately.

Acceptance reuses the dataset pipeline: version pinning in
`dataset_versions`, an independent review ledger covering
every template (`review_ledger`), split-isolation and
overlap audits, and concentration caps. A version trains only
after `accept_version` records a strict audit.

## Two study modes

`improve_existing` continues a released model. Its established
capabilities become retention floors: the study may only
promote a candidate that clears frozen per-view limits on
accuracy, violation recall, false-safes, whole triplets, and
named families.

`create_new` starts from a verified untouched foundation
checkpoint: the manifest records the full export hashes and
refuses any adapter branches found in the safetensors header.
A separately reviewed foundation reference must name the
checkpoint, revision, and exact export file hashes. Absence of
adapters alone does not establish an untouched foundation.
It qualifies a new model domain
with its own dataset and evaluation plan.

## Freezing the plan

`decision_study.freeze_study_manifest` binds model and dataset
hashes, code tree, runtime revision, tokenizer and question
contract, sampling pools, learning rates, seed, hardware,
evaluation views, frozen selection objective, and the exact
launch command into one digest-pinned manifest. Reserved
splits are hash-bound, never read. `verify_study_manifest`
rechecks the bound execution inputs. Reserved split identities are recorded
without opening their data. The supervisor runs verification before each
round commit and refuses on any drift. Freeze a source
snapshot, never the live tree: caches and edits under an
unfrozen checkout can invalidate the code check. Python bytecode
caches are excluded. Source and runtime code are not.

The dataset paths must hash to the accepted version's registered
TRAIN and development splits. The acceptance record, review
ledger, and sampling pools are bound too. Runtime and tokenizer
identities name actual directories, whose full contents are
hashed. `question_contract_path` names a valid export contract
record. `code.python` identifies the executable or immutable
container launcher, and `code.image_digest` identifies the image.

`training.settings` contains the exact effective CLI settings.
`training.recipe` contains the full recipe, including an identical
`training` settings object. The declared trainable groups, batch
geometry, learning rates, schedule, seed, update budget, and
evaluation cadence must agree with those settings. Partial-block
training currently means the last N encoder blocks, matching the
trainer's supported interface.

The launch command is an argument list with optional whole-argument
`{experiment_id}` and `{output}` placeholders. It must match
`security_jobs.command_for` exactly. The job profile records
`decision_study_manifest` and `decision_study_sha256`. Supervisor
commit, operator prepare, operator submit, and the owned worker
all verify the binding. A valid manifest beside an unrelated job
does not authorize that job.

Objectives start from a measured baseline of the starting
checkpoint, then freeze target improvements and retention
limits as exact counts through the shared gate vocabulary
(`constraint_violations`). Trainable parameters are declared
explicitly: full encoder, the last N encoder blocks, or frozen encoder
with adaptation/decision parameters. A small fitting probe must show movement in declared
groups, none in frozen groups, and strict TRAIN-fit gain
before larger runs. A resource check refuses recipes that
exceed offered GPUs, memory, or hours.

## Bounded loop and diagnosis

The supervisor runs the study within `max_rounds` (compute
hours are recorded and enforced by the job operator). Each
round records its hypothesis and delivered exposure. When a
round misses its objectives, `diagnose` classifies the
failure using the evidence checks below. This ordering matches `diagnose()`:

| Check, in order | Diagnosis when it fails |
|---|---|
| Job completion and known SDK parity | Execution defect |
| Required training exposure delivered | Insufficient exposure |
| Registered TRAIN-fit criterion met | Failure to fit |
| Label audit has no reported verification failure | Poor labels |
| Target passes but protected retention fails | Retention trade-off |
| TRAIN fits but target still fails | Poor generalization |

An attempt that meets every objective has no failure to diagnose. This ordering
prevents incomplete execution or exposure from being called a modeling limit.

Every terminal analysis carries a machine-checked diagnosis
block (category, completion/exposure booleans, reasons).
The agent derives it with `assess_round`, which reads
development counts from the trainer report plus the frozen
dev rows and exposure from the run's own tracked draws.
TRAIN fit comes from a fitting probe, since training
reports carry loss, not accuracy. Runs must enable
full-exposure plateau tracking or they cannot qualify.
When a new model is proposed, the supervisor re-derives
completion from receipt phases and exposure from tracked
draws, and refuses any overclaim.

`assess_round` requires the expected owned-job run ID and the
verified export weights hash. It recomputes confusion matrices,
class counts, pairs, and whole triplets from row-aligned predictions.
Supplied summaries must agree. Multiple views must partition the
development rows exactly. Named family gates use `mechanism_id`
within their registered view. Missing family evidence refuses the
assessment. Unknown SDK parity is an execution defect, not a pass.

`evaluation.selection_objective` is the complete trainer objective,
including version, protected views, ranked view, and targets.
Trainer target and retention constraints must agree with the study
gates. Family gates are additionally checked at round assessment.

Selection cannot drift silently: the manifest pins the
campaign policy digest, and every commit refuses when the
live policy differs. Amend the study instead. Before the
first round commits, the operator stages a passing
fitting-probe receipt and a resource-check receipt beside
the manifest. Both name `study_sha256`, `parent_sha256`, and
`train_sha256`, plus an `artifacts` mapping of measured evidence
paths to SHA-256 hashes. The probe measures every parameter group
and strict TRAIN-fit gain. The resource receipt records positive,
finite requirements and available capacity. Both are rechecked
before every submission, including retries.

The operator reserves each attempt's full timeout plus its stop
grace in a locked, durable `budget.json` beside the study. Retries
consume the same cumulative `max_hours` budget. Reservations are
conservative and are not reclaimed after a crash or failed launch.
Repeated submission of the same attempt does not charge twice.

GPU sharing is opt-in. A `gpu_coexistence` profile may name approved
resident PIDs with their start times and command hashes, together
with minimum available system memory. Unknown or reused PIDs fail
closed. This supports a memory-capped trial beside an owned server
on unified-memory hardware. The launcher must enforce the training
memory cap. The default still requires idle GPUs.

## Training path

One pass through the loop covers, in order:

1. The agent proposes a hypothesis and next round within the frozen settings.
   A dataset, recipe, or policy change requires an explicitly reviewed study
   amendment before the changed specification can be committed.
2. The supervisor re-verifies the study manifest, the
   dataset acceptance from live bytes, and the round budget,
   then prepares the round against the frozen spec.
3. The owned job runner trains through `security_train.py`
   with checkpointing, resume, and recovery intact.
4. Development evaluation selects a checkpoint within the run. Round assessment
   then checks the complete promotion policy and either promotes that candidate
   or preserves the incumbent.
5. The agent analyzes the terminal round and its diagnosis
   feeds the next proposal until the budget is spent.

Harness² procedure learning is attached to this loop in
both directions, enabled through the study manifest. A
replay-qualified guidance bundle is appended to every agent
prompt (analysis, preparation, recovery, planning) with its
digest journaled. Each terminal round's trainer report is
queued back as a retrospective review task. Guidance never
changes policy, tools, or authority, and learning failures
degrade to unavailable without stalling owned GPU jobs.
Without a procedure block the loop runs identically minus
the advice.

The study pins the qualified bundle digest. If the worker publishes a new bundle
at the same path, the existing study refuses the changed binding at the next
verification. Adopt the new guidance through an explicit study amendment, or
keep the study's bundle generation immutable while new trials publish elsewhere.
The unbound supervisor path can consume newly qualified bundles at future
sessions without this additional study pin.

## Integration entry points

These are Python APIs, not a `decision_study` launch command:

| API | Input and output |
|---|---|
| `freeze_study_manifest(spec)` | Validates the specification and returns a digest-bearing manifest. It does not write it. |
| `verify_study_manifest(manifest)` | Revalidates execution inputs and returns `study_id` with `verified: true`. |
| `qualify_fitting_probe(report)` | Checks declared parameter movement and TRAIN fitting evidence. |
| `check_resources(requirement, available)` | Checks finite capacity and budget requirements. |
| `assess_round(...)` | Recomputes view counts from aligned development predictions and checks export/run identity and exposure. |
| `diagnose(evidence)` | Returns a failure category and bounded advice. |

Persist `study.json` under the supervisor's `campaign/studies/<study_id>/`
layout, configure its `study_id`, and bind the same manifest and digest into the
owned-job profile. Probe and resource receipts must reference real artifacts.
The [synthetic study tests](../tests/test_decision_study.py) and
[enforcement tests](../tests/test_study_enforcement.py) show complete contract
shapes. They are fixture examples, not substitutes for measured launch evidence.

## Recommending a new model

`recommend_new_model` runs only after the budget is spent,
every terminal round is analyzed with a recorded diagnosis,
and at least one qualified attempt finished with full
exposure, verified SDK parity, and demonstrated TRAIN fit.
Only poor generalization or a retention trade-off supports this
recommendation. Execution defects, poor labels, inadequate
exposure, and failure to fit call for repair or investigation.
A successful promotion also blocks an exhausted-model claim.
Both the analysis and planning paths persist the parked state.
It writes a proposal naming the domain, dataset
plan, and evaluation plan, records the preserved incumbent,
and parks for owner approval. Nothing launches without it.

Selection and release reuse `checkpoint_selection` under the
frozen development gates. The selected export is qualified
separately before its capability profile and reproduction
instructions are published.
