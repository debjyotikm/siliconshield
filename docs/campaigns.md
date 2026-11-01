# Campaigns and controllers

[Documentation index](README.md)

SiliconShield separates experiment planning, job execution, and checkpoint
promotion. An agent recommends changes. The job runner records what executed,
and the campaign applies the checkpoint acceptance rules.

## Choose a controller

| Path | Scope | Entry point |
|---|---|---|
| Muse coding-agent supervisor | Analyze completed rounds, prepare a proposal, recover, investigate, or plan the next round. | `campaign_supervisor.supervise_once()` Python API. |
| Bounded Glimmer operator | Operate one preauthorized experiment within an intervention budget. | `python -m siliconshield.research_loop.execution.glimmer_operator --config ...` |
| Glimmer training watcher | Follow a training container with evaluation and optional Harness² work. | `python -m siliconshield.research_loop.execution.campaign --help` |
| Fitting/development strategy | Sequence configured fitting and development arms. | `python -m siliconshield.research_loop.execution.strategy --config ...` |

These are separate paths. The campaign supervisor does not require a locally
served Glimmer adapter. The Glimmer operator does not create an unrestricted
follow-on research campaign.

## Prepare an owned security-model job

Copy [the job profile](../configs/glimmer-security.example.json) outside Git.
Set its interpreter, frozen source, runtime checkout, model, dataset, run root,
GPU indices, and defaults. Profiles sharing GPUs must use the same
`resource_root`. Optional `recipe`, `reporting_contract`, `review_ledger`,
and `review_packets` keys name reviewed local files. `defaults` may pin
`question_contract` using the full id `siliconshield-requirement-context-v0`
or `siliconshield-requirement-context-v1`. Training validates every row against it
and records it on export. `defaults` may also set
`plateau_requires_full_exposure` to block plateau interventions until
delivered draws cover a full first pass.

Inspect the prepared command and hashes without launching:

```bash
python -m siliconshield.research_loop.execution.security_jobs \
  --profile /path/to/security.local.json --experiment-id experiment-a
```

This preparation writes local receipts. Add `--launch` only when ready to
submit through the owned-job launcher. Campaigns use the Operator API for their
state and intervention contract. Active GPU processes can cause preflight to refuse.
Resource contention at submission can return a waiting result. Inspect the
returned state before retrying the same experiment.

The service explicitly receives `CUDA_VISIBLE_DEVICES`, `LAYA_ROOT`, and
`SILICONSHIELD_SECURITY_RECIPE` when the profile supplies `recipe`. It does not
inherit arbitrary shell variables. Passing a recipe only in your interactive
shell is not equivalent to registering it in the profile.

For new owned work, freeze the plan as a [decision study](decision-study.md)
before preparing the profile. The sections below cover repeated-round
integration. For status, stop, and recovery, use [operations](operations.md).
Never launch a second driver merely because the first client's connection ended.

## Register a campaign

A deployment calls `laya_campaign.init_campaign(root, config)` once. Required
configuration includes a campaign ID, parent export and weights hash, recipe
path and hash, selection policy, and incumbent export/results/evaluation receipt.
A reporting contract and operator profile can also be supplied. The function
verifies these inputs before writing the registry.

Register accepted dataset versions, then call `prepare_round()` with a version,
recorded hypothesis, and supported training overrides (including
`question_contract`, `review_ledger`, `review_packets`, and
`plateau_requires_full_exposure` where applicable). A repeated
identical specification is idempotent. Changing the specification under the
same round ID is refused. The round freezes the incumbent, dataset, recipe,
policy, question contract, review ledger and packets, and report bindings.
A fresh single-view study freezes an exact-id selection view with no
retention population. Its generated
recipe contains the incumbent-derived selection reference and must be the
recipe used by the training profile. This single-view flow is a
different mechanism from decision-study manifests. New owned work
should use a [decision study](decision-study.md).

Campaign configuration and dataset registration use Python APIs. The regression fixtures in
[test_laya_campaign.py](../tests/test_laya_campaign.py) show the complete contract
using synthetic artifacts. Do not substitute dummy evidence to initialize a real
campaign.

## Decision studies

New owned work freezes its full plan as a [decision
study](decision-study.md) before training: parent export, accepted
dataset version, code and runtime identity, sampling pools, recipe,
evaluation views, frozen selection objective, budget, and the exact
launch command. The job profile names the manifest and its digest.
Supervisor commit, operator prepare and submit, and the owned worker
all verify the binding. An agent proposal cannot silently change settings
already frozen in that study. A changed recipe or policy needs a reviewed
amendment and new identity before a bound launch. Each submission reserves its timeout plus
stop grace in a locked budget beside the study.

Budget limits block further launches. A new-model recommendation additionally
requires every terminal round to be analyzed and at least one qualified
attempt with full exposure, demonstrated TRAIN fit, SDK parity, and a
poor-generalization or retention-trade-off diagnosis. Execution defects or
incomplete training do not justify a new model. A valid recommendation parks
with a domain, dataset plan, and evaluation plan while preserving the incumbent.
A fresh study requires owner authorization.

## Round lifecycle

![Persisted campaign phases and the owned job, promotion, and between-round supervision boundaries.](diagrams/decision-study.svg)

**Figure 03.** The state names match `PHASES` in `laya_campaign.py`.
[Open full-size SVG](diagrams/decision-study.svg) ·
[Editable draw.io source](diagrams/siliconshield-architecture.drawio)

A phase is a persisted boundary. Its handler performs work before advancing:

| Persisted phase | Work on the next driver pass | Successful next state |
|---|---|---|
| `prepare` | Validate the registered dataset and round inputs. | `validate_data` |
| `validate_data` | Prepare/preflight the Operator and check campaign/profile bindings. | `preflight` |
| `preflight` | Wait for GPU availability. | `acquire` |
| `acquire` | Recheck the frozen policy and submit the owned job idempotently. | `submit` |
| `submit` | Enter observation. | `observe` |
| `observe` | Reconcile the unit, finalize a completed job, invoke the evaluator, and attach the verified candidate. | `evaluate` |
| `evaluate` | Apply selection, revalidate evidence, and complete the promotion transaction. | `decide` |
| `decide` | Write and close the round report. | `report`, then `done` |
| `report` | Recover an interrupted report closeout. | `done` |

The live-job wait remains in `observe`. An unavailable resource can keep the
round in `preflight`. Any active phase may explicitly transition to `failed`.
Some validation failures raise before a transition, preserving the last phase
for diagnosis. A refused operation therefore does not necessarily produce a
terminal round receipt.


A restart rereads the receipt. It does not
unconditionally submit a new job. Promotion uses a transaction intent and
revalidates the candidate export, evaluation receipt, frozen policy, and
incumbent before completing the change.

`run_round()` requires two integration functions:

- `operator_factory(profile_path)` returns an operator for the registered job
  profile, normally the existing `Operator` class.
- `evaluator(campaign_root, round_id, reported_job)` returns the candidate export,
  adapted results, and an identity-bound evaluation receipt.

The evaluator must bind real results to the correct round and export. It is not
sufficient to return a metrics dictionary with a successful process exit.
`selection_inputs_from_report()` adapts trainer views and cross-checks rates
against counts. Receipt validation checks which export produced those results.

## Wire the supervisor

`supervise_once(home, config, operator_factory=..., evaluator=...)` reconciles
unfinished rounds, records terminal results, runs at most one agent session, and
writes status. `home/campaign/` contains the registry and round state.
`home/campaign/supervisor/` contains the durable handoff.

The local driver must supply the callbacks, polling schedule, singleton/service
ownership, and deployment paths. There is no `campaign_supervisor --config`
command or generic supervisor systemd unit in this repository. Keep that
machine-specific driver outside Git and launch it against a frozen source tree.

Agent configuration names `muse_bin`, `model`, and `repo`, with optional
per-session timeouts, model-step limits, reasoning effort, extra arguments,
and `agent_env_extra` allowlist entries. Optional `procedure_queue` and
`procedure_bundle` connect the separate Harness² worker. See
[Muse procedure learning](muse-procedures.md).

The agent runs sandboxed by default with web tools disabled. Set
`allow_unsandboxed_agent: true` in the operator-owned config only for a
trusted coding-agent path with workspace access — never by default for
unattended sessions, and never as a substitute for deployment isolation.
Agent and tool subprocesses inherit only an allowlisted environment and run
under bounded CPU, memory, output, and process limits. Filesystem access
itself is unchanged, so untrusted work still belongs on an isolated host.
Deterministic campaign checks protect launch and promotion. They do not turn
the coding agent itself into a sandbox.

## Operate one job with Glimmer

Copy [the operator example](../configs/glimmer-operator.example.json) outside
Git and configure a verified endpoint/adapter, profile, experiment ID, output,
polling budget, and intervention limits. Then use:

```bash
python -m siliconshield.research_loop.execution.glimmer_operator \
  --config /path/to/operator.local.json
```

The six actions are `continue`, `leave_unchanged`, `stop`, `adjust_lr`,
`resume`, and `finalize_retry`. The execution state machine validates authority,
parameters, resume identity, and budget before acting. Inspect the returned
execution state to confirm whether an intervention ran.

Reaching the polling limit does not mean the training job has stopped. The
systemd-owned job can outlive the client. Reconnect to its recorded identity. Do
not resubmit it under a new experiment ID to obtain status.

## Preserve incumbents and policies

Promotion and deployment acceptance are separate outputs. A candidate may
satisfy deployment thresholds but fail to improve the research incumbent, or
improve research metrics while failing deployment requirements. Compare the
recorded reasons and rank vectors to see which rules passed or failed.

`v1-transfer` retains the transfer-oriented tiebreak. `v2-retention` orders
feasible candidates by fewer retention false-safes, more correct decisions, more
correct pairs, retention NLL, and transfer NLL. Changes to ranking semantics are
policy changes. Explicit relaxation requires the update API's authorization and
a recorded reason. The supervisor does not silently loosen gates.

Stopping after failed hypotheses is a supported outcome. See
[operations](operations.md) before resuming a stopped or parked campaign.
