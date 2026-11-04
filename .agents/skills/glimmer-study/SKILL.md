---
name: glimmer-study
description: Assemble and run a Glimmer-operated decision-model study on owned GPUs with validated inputs and readable failure receipts.
---

# Glimmer-Operated Study

Use this skill when running SiliconShield-RTL training through the
Glimmer operator loop (`strategy` campaign plus owned systemd jobs).
It covers assembly, pre-submit validation, operator behavior, and
failure reading. Study design and selection policy live in the
study-design skill. This skill never promotes a checkpoint or opens
sealed evaluation data.

## Assemble outside Git

Keep the profile, dataset, recipe, snapshots, runs, and receipts
outside the repository (for example under `/tmp` or a scratch
volume). Nothing in this flow is committed.

A study needs five pieces:

- A dataset directory with `train.jsonl` and `dev.jsonl`. Every
  `pair_id` must carry all three labels as complete triplets.
  Split groups must not cross splits. Question bytes must match
  the pinned contract exactly. Build questions with the
  repository contract helper, never by hand.
- A pool manifest when sampling by pool. Every train row id must
  appear in exactly one pool.
- A recipe when the study uses source names outside the trainer
  defaults. Reporting weights need declared mass for every
  observed source. The profile `recipe` path is forwarded into
  the job environment.
- A source snapshot holding the full `siliconshield` package.
  Units launch the worker module from the snapshot as their
  working directory, so a trainer-only copy fails on import.
- A job profile: job python, snapshot, parent export, dataset,
  runs directory, resource root, GPU list, runtime root, timeout,
  and training defaults. One GPU takes a single-device list.

## Validate before submit

Run the study-input validator on CPU before any submission:

```bash
python -m siliconshield.training.laya.validate_study_inputs \
  --data /path/to/data --pools /path/to/pools.json \
  --recipe /path/to/recipe.json --seed 7 \
  --global-batch 6 --world 1 --max-updates 9 \
  --output-receipt /path/to/receipt.json
```

It runs split loading, pool assignment, sampler simulation with
coverage horizon, and reporting-weight checks using the
trainer's own gates. A failure here is a data defect. Fix the
data, not the gate. The coverage report tells how many updates
full triplet exposure needs. Compare that horizon with the
planned budget instead of assuming coverage.

## Domains and retries

One resource root plus one runs directory is one reservation
domain. Never move a profile's runs directory inside a live
domain. The owner record binds the original output path, so a
moved tree leaves an unresolvable owner. Start a fresh domain
instead.

Experiment IDs are single-use. A failed attempt keeps its
receipt under its ID, and resubmitting the same ID reattaches
to the failure. Every new attempt takes a new ID.

Preflight also checks free disk on the runs filesystem against
twice the parent export size. Free the disk or move the runs
directory before retrying. Do not delete another run's
receipts to make room.

## Operator behavior

The strategy driver submits one fitting arm per recipe, promotes
recipes by measured TRAIN gain, then runs one development round.
The operator prepares, preflights, submits, observes, and
finalizes. Preflight refuses to disturb live GPU work, probes
the worker entrypoint import from the snapshot, checks disk
headroom, and dry-runs the frozen trainer settings on CPU.

Restate the full development schedule in
`development_overrides`: maximum and minimum updates, warmup,
hold, patience, eval cadence, and learning-rate schedule
shape. Fitting values never carry forward silently. The
driver refuses a development round that would inherit them.
Recipe values such as learning rates and batch size do carry
from the winning arm by design. Keep `development_views` out
of shared defaults and fitting arms. It belongs in
`development_overrides` only, since fitting rejects views.

A gain floor of zero qualifies recipes that do not regress
TRAIN fit. Small fitting gains vary run to run at fixed
seed, so use only the sign. Negative gain stops the
campaign before any development spend.

`intervention_budget` of `None` takes the default learning-rate
budget. An explicit empty object disables interventions. These
are different. State which one the study uses.

With an untrained model driving turns, expect well-formed but
conservative behavior: inspection calls, then `continue` or
`leave_unchanged` on healthy runs and `stop` on dead ones. Stop
requests are rejected without recorded failure evidence or a
running job. That rejection is the guard working, not an error.

A dead unit with no usable receipt is terminalized by abandon,
which writes a failed execution receipt under the attempt
identity and releases the reservation for later experiments.
The driver then reports the failure instead of spinning.

A rejected model call over the Spark transport (an SSH error
in one cycle) is an expected transient. The loop records it,
the owned job keeps running, and the next cycle continues.
Only repeated transport failures across cycles deserve
investigation of the endpoint itself.

## Read failures from receipts

`execution.json` carries the terminal receipt. Its `failure`
field holds the child traceback summary when the child raised,
or a timeout, quota, or abandon note. The full child output
stays in the referenced log file. The strategy error names the
receipt. Start there.

Common causes and their receipt signatures:

- Worker import failure: no receipt at all on the first
  attempt, unit failed within seconds. Rebuild the snapshot
  with the full package.
- Undeclared source mass: `ValueError` naming the source.
  Add the source to the recipe mass.
- Incomplete triplets or pool gaps: the trainer names the
  defect. The pre-submit validator catches these before any
  GPU time is spent.

## Eval parity

Trainer evaluations and SDK replays are separate inference
paths. They can disagree confidently on boundary rows through
precision and input-preparation differences. Before calling a
replay mismatch a weights defect, rerun the differing rows one
per batch to rule out batching, then compare pinned settings.
Reload parity on fitted exports should be exact. Report the
settings alongside every comparison.
