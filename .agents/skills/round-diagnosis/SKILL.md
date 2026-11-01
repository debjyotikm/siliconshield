---
name: round-diagnosis
description: Diagnose a terminal training round with the fixed failure taxonomy and hard-won confound checklist.
---

# Round Diagnosis

Use this skill when a training round ends without promotion.
Classify with `decision_study.diagnose`, in fixed precedence:
execution defect, poor labels, insufficient exposure,
failure to fit, poor generalization, retention trade-off.
Failed execution and incomplete training are never modeling
verdicts. Record the diagnosis block in the analysis. The
supervisor re-derives completion and exposure from run
records and refuses overclaims.

Call `assess_round` with the expected owned-job run ID and
the verified export weights hash. It validates predictions
against confusion matrices and recomputes pairs, triplets,
multiple development views, and registered mechanism gates.
Unknown SDK parity is an execution defect. Only a completed,
fully exposed, fitting-qualified attempt with SDK parity can
support poor-generalization or retention-trade-off advice.

## Evidence to pull first

- `report.json`: stop reason, selected step, dev trajectory,
  train-loss curve. Falling loss with stalled dev is a
  hypothesis to investigate using measured TRAIN accuracy
  and triplets, not proof of memorization by itself.
- `events.jsonl` delivered exposure: unique rows, broken
  triplets, and pool shares at both selected and terminal
  steps. Never call a plateau before the required pools
  complete a pass. Simulate the sampler over the executed
  horizon, not the planned maximum.
- TRAIN-fit probe on the selected weights: per-class and
  whole-triplet counts on TRAIN triplets. Keep a separate
  fitting-probe result distinct from the selected artifact's
  measurements. Poor fit requires checking labels, exposure,
  optimization, and parameter movement before blaming capacity.
- Fresh SDK reload parity on dev: any mismatch stops all
  quality interpretation until execution is fixed.
- Label audit: solver-evidence failures and review
  disagreements on the failing families before blaming the
  model.

## When selection keeps the parent

Record parent, selected, and terminal weights separately,
with each artifact's hash and update. Selection retaining
initialization is a null selection, not evidence that no
optimizer updates happened or that training cannot fit.

Measure terminal TRAIN fit and inspect the trained
checkpoints' development gates. Distinguish failure to learn
from learned changes that fail retention or selection. Keep
the machine diagnosis scoped to the artifact it assessed.
Do not attribute terminal exposure or fitting gains to the
selected initialization, or override a failed selection gate
because TRAIN fit improved.

## Confounds we have paid for

- Trainer ranking vs campaign promotion: confirm `best.json`
  was chosen under the same joint constraints promotion
  enforces, or a better checkpoint may sit unselected.
- Scoring path: batching, precision, and runtime differences
  can change boundary decisions. Compare checkpoints under
  matched inference settings before attributing a change
  to training.
- Distributed coherence: one combined backward per
  microbatch. Cross-rank gradient and weight checks after
  any objective change. Mixed-replica scores are not
  coherent-model results.
- Input-style match: TRAIN-only summaries or annotations
  can teach shortcuts absent at inference. Include probes
  in the intended deployment format before claiming transfer.
- Semantic objectives: option order shuffles per row, so
  margins and pair terms must bind to semantic labels, not
  slots, with a regression test over all six permutations.
- Single-variable rounds: change one thing (data, mixture,
  objective, seed, schedule) per experiment, or the next
  failure stays unexplained.

## Boundaries that never move

Never relax a frozen gate to manufacture success, never
open sealed qualification or test data for any feedback,
and never promote on aggregate gains while a safety gate
fails. A null round with clean evidence is a result:
record it and change the hypothesis.
