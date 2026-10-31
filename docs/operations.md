# Operations and recovery

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

Glimmer training, serving, and evaluation use the host-mounted file named by
`SILICONSHIELD_GPU_LOCK`. All containers and host processes sharing a device
must point at the same underlying file. A busy lease refuses acquisition.
Schedule serving and fine-tuning in separate resource windows.

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

The Glimmer trainer validates current and previous checkpoint generations and
refuses explicit resume when neither is usable. It restores optimizer/progress/RNG
state and checks development-loss parity before continuing. Do not work around
an identity failure by editing its signature.

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
| Model input truncated | Increase a qualified budget or reconstruct complete relevant context. Do not publish its discarded judgment. |
| Glimmer replay fails to parse | Verify model, adapter, native tool parser, response schema, and decoding settings together. |
| Slow Glimmer loading/decoding | Separate load time, prefill, and decode. Inspect the qualified loading/runtime configuration before changing precision. |
| Procedure worker fails | Preserve its request/attempt receipt and use a fresh output after correction. Don't publish failed guidance. |

See [inference](inference.md) for timing and loading details. After a
performance change, repeat the matched quality checks before deploying the
controller.

## Deploy a code update

Validate the new tree independently, freeze a new snapshot, and switch the
driver at a deliberate boundary. Do not overwrite files underneath a live frozen
process. The Harness² worker runs as a separate service. Updating its code does
not implicitly update the campaign supervisor or vice versa.

Archive evidence before cleanup. Preserve all files referenced by manifests,
selected exports, replay reports, and active processes. The public source tree
should never become the runtime archive.
