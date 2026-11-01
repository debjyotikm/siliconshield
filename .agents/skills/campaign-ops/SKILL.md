---
name: campaign-ops
description: Operate the unattended campaign loop with owned jobs, recovery, budgets, and strict authority boundaries.
---

# Campaign Operations

Use this skill when running or minding a SiliconShield
unattended training campaign. Machinery:
`siliconshield/research_loop/execution/campaign_supervisor.py`
and `laya_campaign.py`.

## Who owns what

The supervisor owns mechanics. Agent sessions own judgment.
The supervisor reconciles unfinished rounds, commits staged
preparations only after re-verifying them, refreshes
status, and never prepares a round without a validated
proposal. Agents analyze rounds, propose hypotheses, and
prepare data. Neither side edits verification outcomes or
relaxes a registered threshold.

## One pass

Each `supervise_once` reconciles every open round, records
newly terminal rounds, runs at most one agent session
(analyze, prepare, recover, or plan), and refreshes status.
All progress lives in durable records. Restart recovers by
rereading state, and sessions continue across context
limits through the persisted handoff.

## Jobs and recovery

One owned training job at a time per campaign, with GPU
lease and idempotency keys. A completed run is never
resubmitted. Reattach to the same owned job only when its
frozen inputs still match. Resume from validated progress
state (weights, optimizer, RNG, data position). The latest
resumable checkpoint and the best selected export serve
different purposes. A fresh continuation from an export
must declare its optimizer reset rather than claim resume.

For an implementation bug, reproduce and test the fix, then
freeze a new source generation and affected configuration.
Register a replacement with a new run identity and study
binding, recording its link to the failed attempt. Never
replace the frozen inputs of an existing job. Preserve its
receipts and charged budget, including across replacement
studies. If the repair exceeds existing authority or budget,
park with the exact blocker. Failed rounds leave the
incumbent intact.

GPU sharing is opt-in through `gpu_coexistence`. Pin approved
resident processes by PID, start time, and command hash,
verify available memory, and enforce the trial's memory cap
in its launcher. Unknown processes refuse the launch. Do not
stop or adopt a foreign workload to make a trial fit.

## Budgets and proposals

Studies run within frozen round and cumulative time budgets.
Each submission reserves its timeout plus stop grace under
a file lock. Retries consume budget and duplicate requests
for an existing attempt do not. Reservations remain charged
after failures. Commits refuse past the round cap and the
loop parks. A spent budget with no
promotion leads to `recommend_new_model`: domain,
dataset plan, evaluation plan, only after a qualified
attempt with demonstrated TRAIN fit and SDK parity. A new
study starts only within the owner's authorized scope.
Harness² guidance may advise prompts and learn from queued
reports, but it changes no policy, tool, or authority.

## Never

Never launch training the owner did not authorize, never
touch sealed qualification or test data, never train on
evaluation outcomes, never rewrite a verification result,
and never present a research-only checkpoint as accepted.
When GPUs sit idle, say exactly why from the status
record. Never guess.
