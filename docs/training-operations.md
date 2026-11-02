# Operate and recover training jobs

[Documentation index](README.md)

Keep runtime workspaces separate from source checkouts. A deployed campaign
should use a frozen source snapshot, pinned model/runtime dependencies,
immutable input generations, and a local profile. Service names and polling
drivers depend on the deployment.

## Resource ownership

Security-model jobs use an operator reservation, serialized submission, and a
process-lifetime runtime lock. Configure one shared `resource_root` in profiles
that compete for the same GPUs. Different directories do not coordinate with
one another. `nvidia-smi` is an observation, not an ownership lock.

Idle GPUs are the default preflight requirement. An explicitly configured
`gpu_coexistence` exception can permit named resident processes, bound to PID,
start time, and command hash, with a system-memory floor. The training launcher
must enforce its memory cap. This does not authorize sharing with unknown work.
See [study walkthrough](study-walkthrough.md).

Glimmer serving and in-process evaluation use the host-mounted file named by
`SILICONSHIELD_GPU_LOCK`. All containers and host processes sharing a device
must point at the same underlying file. A busy lease refuses acquisition.
Schedule competing serving and evaluation processes in separate resource
windows. Replay through an existing endpoint uses that server's lease.

Do not delete a lock file to clear contention: a running process may retain the
old inode while a second process locks a newly created file. Determine ownership
and stop the owner through its normal control path.

## Inspect an owned job

Read the job's `run-manifest.json` for identity, phase, unit, attempts, and history.
Use the recorded unit rather than guessing one from a run directory:

```bash
systemctl --user status YOUR_RECORDED_UNIT --no-pager
systemctl --user show YOUR_RECORDED_UNIT \
  -p LoadState,ActiveState,SubState,Result,ExecMainStatus,Description,InvocationID
journalctl --user -u YOUR_RECORDED_UNIT -n 100 --no-pager
```

Then inspect `execution.json`, `job.json`, trainer `status.json`, logs, and the
selected export receipt. Systemd exit success alone does not establish a valid
training report or a promoted checkpoint. A missing garbage-collected unit
requires terminal-receipt reconciliation. It is not permission to duplicate a
job.

For a campaign, read `campaign/incumbent.json`, the relevant round receipt,
leaderboard, and `campaign/supervisor/handoff.json`. Read the reason recorded
for a stopped or parked campaign. A healthy supervisor with idle GPUs can be
waiting for resources, investigating, or stopped after failed hypotheses.

## Stop without accidental follow-on work

1. Stop or pause the deployment's supervisor/watcher so it cannot start another
   round or an automatic evaluator.
2. Observe the current owned job through the Operator and request the permitted
   `stop` intervention. Allow the trainer to reach its checkpoint boundary and
   save terminal state.
3. Reconcile the job and preserve reports, selected export, latest state, and
   execution receipts. Verify that compute processes have exited.
4. Keep follow-on workers disabled until the campaign is intentionally resumed.

Stopping a polling client alone does not stop its systemd-owned training job.
Conversely, stopping a job without stopping its supervisor can leave a live
controller free to schedule follow-up work.

### Request and confirm a stop

First pause the deployment's actual supervisor through its configured service
or driver. This repository does not ship a universal supervisor unit name.
Then load the existing manifest and request a stop through the same job profile.
The following Python runs on the operator host and **changes the owned job's
control state**:

```python
import json
from pathlib import Path

from siliconshield.research_loop.execution.operator import Operator

operator = Operator("/path/to/bound-job-profile.json")
manifest_path = operator.manifest_path("registered-experiment-id")
manifest = json.loads(manifest_path.read_text())
manifest = operator.observe(manifest)
manifest = operator.intervene(manifest, "stop")
print(Path(manifest["output"]) / "operator-control.json")
```

Use a manifest in `observe`, `diagnose`, or `intervene` phase. A manifest already
in `report` is terminal-report reconciliation, not a new stop request. The API
refuses to stop a foreign or non-running job. The returned request is not proof
that training has exited. Poll the recorded unit, inspect the final checkpoint
and `execution.json`, then finalize/report as below. Preserve a failure receipt
if the trainer exits without producing a valid report.

## Recover the same experiment

Reattach with the same profile, experiment ID, and recipe. The Operator
reconciles running, completed, failed, missing, and foreign units with receipts.
A changed training command is a different experiment. It cannot silently adopt a
previous job. Controller-code drift is recorded on a compatible reattachment.

Use `resume` only with valid checkpoint state, compatible identities, and an
eligible lineage. Use `finalize_retry` when training is complete but export/report
completion needs retrying. It invokes the trainer's development-only finalization
path and must not take extra optimizer steps. Latest resume state and the selected
export serve different purposes.

### Reattach, retry, and close out

Load the existing manifest as above instead of creating a second experiment.
These calls have different effects:

```python
# Read current owned-job state and persist an observation.
manifest = operator.observe(manifest)

# Choose at most one action after inspecting the state and checkpoint.
# Both calls can submit an owned job. They are not status queries.
# manifest = operator.intervene(manifest, "resume", checkpoint_dir=manifest["output"])
# manifest = operator.intervene(manifest, "finalize_retry", checkpoint_dir=manifest["output"])

# Once the job has stopped and its report is present:
manifest = operator.finalize(manifest)
manifest = operator.report(manifest)
print(manifest["terminal"])
```

`resume` restores training state and can perform optimizer updates.
`finalize_retry` executes the development-only finalization path. Both validate
that `latest.pt` belongs to the registered run. For a manifest already in
`report`, call `operator.report(manifest)` directly. A `complete` terminal value
establishes report validity under the job contract, not promotion. An
`incomplete` value requires diagnosis before another attempt.

## Troubleshooting

| Symptom | Inspect and respond |
|---|---|
| Prepared input changed | Compare frozen hashes with source, recipe, teacher, dataset, and profile. Prepare a new compatible experiment rather than rewriting evidence. |
| Foreign or stale unit | Check Description/InvocationID and terminal receipt before adoption or cleanup. |
| GPU busy | Find the recorded owner and resource root. Coordinate release instead of bypassing the lock. |
| Round remains unpromoted | Read all selection reasons and per-view results. A successful trainer exit is not promotion. |
| Selected step is zero | All later candidates may have failed protection or ranking. Inspect the curve and gates. |
| Dataset accepted record invalid | Check manifest/pool drift, audit binding, split hashes, lineage/canonical overlap, concentration caps, and release class. Do not replace it with an empty success marker. |
| Model assessment unavailable | Check configured worker Python, runtime import, checkpoint directory, model identity, and recorded question contract. |
| Study binding mismatch | Compare job profile, dataset, recipe, and launch command against the frozen study manifest. Amend the study instead of editing evidence. |
| Study budget exhausted | Read the refusal or parked reason. A new-model proposal exists only if its evidence gate passed. A fresh study needs owner authorization. |
| Model input truncated | Increase a qualified budget or reconstruct complete relevant context. Do not publish its discarded judgment. |
| Glimmer replay fails to parse | Verify model, adapter, native tool parser, response schema, and decoding settings together. |
| Slow Glimmer loading/decoding | Separate load time, prefill, and decode. Inspect the qualified loading/runtime configuration before changing precision. |
| Routing is thin or unrated | Check profile coverage and full observed identity against the measurements. Do not repair it by manually increasing a tier. |
| Study guidance digest changed | Preserve both bundle generations. Explicitly amend the study before adopting new guidance. |
| Procedure worker fails | Preserve its request/attempt receipt and use a fresh output after correction. Don't publish failed guidance. |

See [inference](inference.md) for timing and loading details. After a
performance change, repeat the matched quality checks before deploying the
controller.

## Deploy a code update

1. Record the active driver, source snapshot, owned job and current phase.
2. Validate a separate source snapshot and preserve the previous snapshot.
3. Pause follow-on scheduling and reconcile the current job to a recorded boundary.
4. Point the deployment's driver at the new snapshot, then run its read-only
   reconciliation check before enabling scheduling.
5. Verify the running source identity and that the existing job was reattached,
   not submitted again. Retain the previous deployment for rollback.

The deployment owns the exact service commands. Do not overwrite files underneath a live frozen
process. The Harness² worker runs as a separate service. Updating its code does
not implicitly update the campaign supervisor or vice versa.

Archive evidence before cleanup. Preserve all files referenced by manifests,
selected exports, replay reports, and active processes. The public source tree
should never become the runtime archive.
