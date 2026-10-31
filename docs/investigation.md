# Investigating RTL and checking repairs

[Documentation index](README.md)

The application evaluates an implementation against an explicit requirement and
assumptions. It preserves inputs, requests a security-model assessment, extracts
relevant statements, executes checks, and verifies a proposed repair against the
same checks. See [workflow.py](../siliconshield/app/workflow.py) for aggregation
and report construction.

## Run the authored demonstration

Use Python 3.11 or newer, Verilator, and a working C++ build toolchain:

```bash
python -m pip install -e .
python -m siliconshield.app --list
python -m siliconshield.app --case unauthorized-write --output /tmp/siliconshield-demo
```

For the unauthorized-write demo, a successful toolchain run should report
`demonstrated_violation_fixed` and `repair_accepted: true`. When no model is
configured, `security_label` is null and the assessment status is `unavailable`.
That is expected: this run checks the authored verification-and-repair path.
Check these report fields after the command finishes:

```python
import json
from pathlib import Path

report = json.loads(Path("/tmp/siliconshield-demo/report.json").read_text())
assert report["conclusion"]["status"] == "demonstrated_violation_fixed"
assert report["conclusion"]["repair_accepted"] is True
```

If the checks do not compile, inspect each `build.log` and verify Verilator,
make, and the C++ compiler are on `PATH`. If execution is inconclusive, inspect
the corresponding `sim.log`. Do not treat a report file's existence as success.

The examples include unauthorized writes, a debug-path lock bypass, legitimate
access, and missing context. They carry authored testbenches and, where needed,
an authored repair. They demonstrate the workflow. They are not unseen model
benchmarks.

The JSON printed to stdout names the conclusion, `security_label`, report path,
and summary path. `laya` is a deprecated alias of the same label for existing
consumers. The output directory contains `report.json`, `report.md`, per-check
build/simulation logs, and `candidate/topmodule.sv` when a repair was proposed.
Use a new directory for each investigation to keep evidence separate.

## Connect a trained security model

```bash
export SILICONSHIELD_SECURITY_PYTHON=/path/to/security-environment/bin/python
export SILICONSHIELD_SECURITY_CHECKPOINT=/path/to/selected-export
export LAYA_ROOT=/path/to/compatible-runtime
```

The assessor launches the worker with that Python interpreter. It uses the
export-record question format and three-way verdict: `laya_worker` resolves
`question_contract.json` from the checkpoint, falling back to the legacy
contract only for unrecorded exports. The newer `SECURITY` environment names
take precedence over the legacy `SILICONSHIELD_LAYA_PYTHON` and
`SILICONSHIELD_LAYA_CHECKPOINT` aliases.

Each assessment distinguishes the product family (`model_id`) from the loaded
artifact (`checkpoint` and its full-content fingerprint) and runtime. The
backend checks checkpoint identity around execution. If checkpoint files change
or cannot be read, the assessor returns an error without a judgment. Keep export
generations immutable.

An unconfigured model is explicitly unavailable. An input that loses state
tokens or question content is reported as truncated without a verdict. Preserving
the full RTL in the evidence report does not mean the model received it all.

## Supply a custom design

```bash
python -m siliconshield.app \
  --rtl /path/to/design.sv \
  --requirement "Only authorized requests may update the configuration register." \
  --assumptions "Two-state sampled-clock RTL under the stated reset conditions." \
  --output /path/to/new-investigation
```

The CLI expects the `topmodule` interface described in its help. Custom input
has no inherited demo checks: its requirement is `requirement_not_verified`
until an integration supplies checks bound to that design and requirement.

For a checked custom case, call `workflow.investigate(case, output, ...)` from
a local integration. Follow the case structure in
[demo_cases.py](../siliconshield/app/demo_cases.py): RTL, requirement, assumptions,
protected/control signals, and named requirement/regression testbenches. The
protected-signal list guides heuristic statement extraction. It is not a learned
localization model. Review testbench coverage independently.

## Request a Muse repair

Add `--muse-endpoint` and `--muse-model` to the authored-case command to connect
a compatible chat endpoint. The proposer receives visible RTL, assumptions,
model assessment, relevant statements, and check outcomes. It does not receive
the authored patch or expected answer.

A demonstrated requirement violation triggers the proposer. Invalid replacement
anchors or endpoint failures become `patch_error`. The workflow does not guess
where to apply a change. Original files remain intact. A candidate must pass all
required checks, including regressions, before its repair is accepted.

The chat client sends no credentials by default, which suits a loopback
service. For a hosted endpoint, supply `api_key` (or an env-var name via
`api_key_env`) so requests carry a Bearer token, always over HTTPS except
to loopback addresses — plaintext HTTP elsewhere is refused. Bound live use
with `max_calls`; the client raises instead of silently exceeding the
budget. Keys never appear in reports or transcripts.

The default demo uses its authored patch when no live proposer is configured.
A report produced that way must not be described as a model-generated repair.

## Read the conclusion

| Status | Interpretation |
|---|---|
| `demonstrated_violation_fixed` | The original violates the requirement. The candidate passes all required checks. |
| `demonstrated_violation_unfixed` | Original violation remains unresolved. |
| `repair_rejected_regression` | Candidate passes requirement checks but fails a regression. |
| `repair_verification_incomplete` | Requirement repair appears successful, but required candidate checks are incomplete. |
| `no_violation_found` | Executed checks passed. This is not a formal proof. |
| `requirement_not_verified` | No requirement check ran. |
| `insufficient_context` | Requirement evaluation lacks necessary context. |
| `regression_failed` | Original regression checks fail without a demonstrated requirement violation. |
| `inconclusive` | Available execution evidence cannot settle the requirement. |

`requirement_verified` means required checks completed with pass/fail outcomes.
It does not mean the design passed. `repair_accepted` is the repair decision.
Inspect `incomplete_checks` and logs before interpreting a summary.

Check classification uses full captured output, not only displayed tails.
Abnormal execution takes precedence over markers: a timeout, missing exit
code, or nonzero exit is an error without counterexample evidence, even when
partial output contains `CHECK_FAIL` lines. On a clean exit,
`CHECK_FAIL` takes precedence over `CHECK_PASS`; no marker is inconclusive.
A demonstrated counterexample remains evidence even if a separate check fails
to execute.

The application currently uses Verilator simulation. Solver-backed label
verification runs in the [data pipeline](data-pipeline.md).
