# Procedure learning with Muse and Harness²

[Documentation index](README.md)

This integration lets the Muse coding agent refine how it reads training
evidence. Harness² proposes and tests prompt advice. The campaign supervisor
can use qualified advice in later sessions. It does not train Muse weights or
change the security model directly.

The backend is `muse-spark-1.3-contributor` through `muse exec`, using the
existing Muse login. It does not require a Glimmer endpoint. The provider
manages this model's weights. The receipts identify the requested model, CLI
binary, settings, and local code, not an independently verified weight revision.

## How the loop works

1. After a completed round has been analyzed, the supervisor copies its
   development observation into a private queue. The adapter rejects reports
   that used final-test results. The copy contains evidence for retrospective
   checkpoint review, not a claim about what caused an improvement.
2. A separate worker runs the pinned Harness² sequential procedure: reference
   replay, proposed revision, probe, and composition. Muse can edit only
   `LongTermMEMORY.md`. It cannot replace the task contract or add executable
   plugins, scripts, or tools.
   Each new trial starts from the last qualified advice, if available, and
   measures the proposed replacement against it on the same replay population.
3. The Python replay dispatcher supplies the observation and executes simulated
   tools. Muse uses an empty native-tool preset with shell and writes disabled.
   Replay cannot submit training jobs. Targets stay inside the external grader.
   The coding agent receives the replay's policy and assessment definitions
   explicitly. It does not need to have learned them through Glimmer training.
   The provider adapter converts nullable optional arguments and comparison
   entries to the existing tool format without changing their values. JSON
   final answers and procedure edits are serialized once by the adapter.
4. Advice must pass a separate, signed development replay population with no
   per-task or per-component regression. Checks cover action, evidence,
   explanation, final-response consistency, and execution completion. Operator
   episodes also check intervention parameters. Only passing advice is published.
5. The supervisor reads the published bundle at its next agent-session boundary
   and records its digest. Invalid or missing guidance is logged and omitted.
   The campaign continues under its existing instructions.

Harness²'s own library selection is not the acceptance test. A lesson that fails
external qualification is removed before the next learning task. Failed trials
leave the previously published bundle unchanged. The bundle is replaced
atomically and checked on read. Its digest detects corruption. The private
directory must still be protected from untrusted writers.

The current checks cover evidence review and replayed operator decisions.
Passing these checks does not establish better experiment design, effective
repairs, or improved RTL accuracy. In particular, the retrospective report
adapter's triage label is not the campaign's research-promotion or deployment
decision. The campaign's registered policy remains authoritative.

## Install separately from a running campaign

Use a separate environment and a frozen source snapshot. Do not install into
the environment used by an active supervisor, edit its source snapshot, or
restart its service to try this integration.

```bash
python3 -m venv /path/to/procedure-env
/path/to/procedure-env/bin/pip install -e /path/to/siliconshield-snapshot
/path/to/procedure-env/bin/pip install \
  -r /path/to/siliconshield-snapshot/requirements/harness2.txt
```

Prepare independently reviewed replay episodes using the existing episode schema
and signed split manifest. Use `train` episodes for proposals and separate `dev`
lineages for qualification. The worker validates the actual file hashes and
split membership. Renaming a test file does not make it eligible. Set
`SILICONSHIELD_DATASET_KEY` to the existing local key file. Never put the key,
episodes, reports, conversations, or generated guidance in Git.

The qualification set should include healthy no-op, regression, trade-off,
invalid-resume, and finalization cases. Arithmetic checks alone cannot qualify
recovery advice. Use authored counterfactuals and unrelated lineages to detect
advice that merely repeats one observed answer. Reused development checks are
not an independent final benchmark.

Run a standalone trial first:

```bash
/path/to/procedure-env/bin/python -m \
  siliconshield.research_loop.procedures.muse_harness2 \
  --episodes /path/to/private-episodes/train.jsonl \
  --checks /path/to/private-episodes/dev.jsonl \
  --muse-bin /path/to/muse \
  --output /path/to/private-procedures/trial-001 \
  --bundle /path/to/private-procedures/current.json
```

This calls the Muse provider but does not use the training GPUs. CLI calls have
separate workspaces and transcripts. A timeout terminates only that call's
owned process group. A failed or interrupted replay is not reused as evidence.

## Connect future supervisor sessions

These optional supervisor configuration keys enable the handoff:

```json
{
  "procedure_queue": "/path/to/private-procedures/queue",
  "procedure_bundle": "/path/to/private-procedures/current.json"
}
```

Start the separate worker using the same qualification set:

```bash
/path/to/procedure-env/bin/python -m \
  siliconshield.research_loop.procedures.muse_harness2 \
  --queue /path/to/private-procedures/queue \
  --checks /path/to/private-episodes/dev.jsonl \
  --muse-bin /path/to/muse \
  --output /path/to/private-procedures/worker-001 \
  --bundle /path/to/private-procedures/current.json
```

Use `--once` to drain the current queue and exit. Without it, the worker polls
every 60 seconds. It locks its output directory and publication target, records
each request's outcome, and skips completed or failed requests on restart. Retry
a failed batch with a fresh output directory after addressing its cause.
Changing the code, model, or qualification population also requires a fresh
output directory.

Keep queue, worker output, and bundle outside both Git and the campaign's frozen
source. A worker can process an exported queue while an older campaign keeps
running. **Installing this code does not activate it in an already running
frozen supervisor.** Deploy the updated supervisor and configuration at a
planned boundary. Do not replace code beneath a live process.

## Inspect and test

- `processed/` records each worker request, including failures.
- `attempts/` contains Harness² proposals, replay traces, external grades,
  and qualification reports.
- `calls/` records Muse CLI requests, transcripts, settings, and terminal state.
- `current.json` contains the qualified advice, evidence grades, and identity.
- Supervisor session receipts record `procedure.status` and the applied digest,
  or the reason optional guidance was unavailable.

```bash
python -m pytest -q tests/test_muse_procedures.py tests/test_campaign_supervisor.py
python -m pytest -q tests/test_muse_harness2.py tests/test_glimmer_harness2.py
bash tools/check.sh
```

The Harness² tests use its real pinned engine with fixture completions. They
skip if the optional dependency is absent. They test the implementation, not
live Muse quality. The existing Glimmer integration remains separate.
