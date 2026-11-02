---
name: study-design
description: Design a reproducible decision-model study with frozen plans, measured objectives, and qualified recipes.
---

# Study Design

Use this skill when starting or reviewing a SiliconShield-RTL
training study, in either mode. Full reference:
`docs/study-walkthrough.md`. Implementation:
`siliconshield/training/laya/decision_study.py`.

## Pick the mode

- `improve_existing` continues a released model. Its current
  behavior becomes retention floors. Promotion needs target
  gains without breaking them.
- `create_new` starts from a verified untouched foundation
  checkpoint and qualifies a new domain with its own dataset
  and evaluation plan. Check actual tensor metadata and full
  export hashes against a separately reviewed foundation
  reference. Adapter absence alone does not prove untouched
  weights.

## Freeze before training

`freeze_study_manifest` binds one digest-pinned manifest:
parent export, accepted dataset version, code snapshot,
runtime, tokenizer, question contract, pools, learning
settings, seed, hardware, views, selection objective,
objectives, budget, and the exact launch command.

- Freeze a source snapshot, never the live tree.
- Reserved qualification/test splits are hash-bound, never
  opened: not by freezing, training, selection, or you.
- An unbuilt reserved split uses `status: not_materialized`
  with a reason, not an invented hash. This permits a
  training-only pilot, not a release-readiness claim.
- Re-run `verify_study_manifest` before every launch. Any
  drift refuses the run.

## Objectives from measurement

Benchmark the starting checkpoint first, then freeze targets
and retention floors as exact counts (accuracy, violation
recall, false-safes, whole triplets, named families).
Count thresholds only, never recall floats, because
populations are fixed. The manifest also pins the campaign
selection digest. A drifted live policy fails the commit
until the study is amended.

## Qualify the recipe

Declare trainables explicitly: full encoder, partial block
list, or adapters only. Before round one, stage two
receipts beside the manifest:

- A fitting probe showing movement in declared groups,
  none in frozen groups, and strict TRAIN-fit gain.
- A resource check proving the recipe fits the offered
  GPUs, memory, and hours.

Both receipts bind the study, parent, and TRAIN hashes and
the measured artifacts. The supervisor checks them at commit
and the operator rechecks them before every submit, including
retries. A recipe change needs a new study.

Use the v2 manifest. Supply actual runtime and tokenizer
directories, a valid question-contract record, and the hashed
Python or container launcher. `training.settings` must equal
the effective CLI settings and `training.recipe.training`.
Bind the job profile to `decision_study_manifest` and
`decision_study_sha256`. A manifest that is not connected to
the submitted job does not qualify the run.

Preflight the actual job through
`siliconshield.research_loop.execution.study_contract.verify_job`. It uses
the trainer's view parser. The development configuration
must name the objective's `selection_view`, enable
`score_triplets`, and declare the required retention or
single-view mode. Views must partition the frozen population
exactly. A schema-valid manifest alone cannot catch a
missing trainer field.

Preflight the study data on CPU through
`siliconshield.training.laya.validate_study_inputs` with the
same pools, recipe, seed, and batch size the job will use.
It runs split loading, pool assignment, sampler coverage,
and reporting-weight checks with the trainer's own gates.
An undeclared source or broken triplet fails here, not on
the GPU. See the glimmer-study skill for the full
pre-submit flow.

A failed fitting probe leaves that recipe unqualified.
Investigate labels, exposure, learning rate, and parameter
movement before changing capacity. Bind a revised recipe to
fresh qualification evidence. Report execution, TRAIN fitting,
and SDK parity as separate checks. Passing them on a tiny
trial does not establish generalization or readiness for
unattended operation.

## Run inside the budget

The supervisor bounds the study to `max_rounds`, verifies
the manifest at every commit, and parks on exhaustion. Each
round records its hypothesis and delivered exposure. Runs
must enable full-exposure plateau tracking or their
attempts cannot qualify. When the budget is spent with no
promotion, `recommend_new_model` proposes a separate
specialist, only after a qualified attempt exists, and the
incumbent is never touched. The owned operator also reserves
attempt time against the cumulative hours budget. Execution
defects and failed fitting probes cannot justify a new model.
Harness² guidance is optional procedure advice, never policy.
