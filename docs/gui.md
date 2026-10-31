# Desktop GUI

[Documentation index](README.md) · [Setup](setup.md) ·
[Development](development.md)

Native Qt desktop window for RTL security analysis. The window stays
responsive because model work runs in a separate persistent worker
process reached over local pipes.

## Launch

Single command (creates the local registry on first run, then opens
the window):

```bash
python -m siliconshield.gui \
  --checkpoint /path/to/immutable-export \
  --python /path/to/laya-environment/bin/python \
  --laya-root /path/to/pinned-laya-runtime
```

This hashes the checkpoint, validates the interpreter and runtime,
and writes `~/.config/siliconshield/models.local.json` (outside Git).
Flags fall back to `SILICONSHIELD_SECURITY_CHECKPOINT`,
`SILICONSHIELD_SECURITY_PYTHON`, and `LAYA_ROOT`. Use `--device cpu`
for CPU-only or `--device cuda:0` for GPU (default, overridable in
the Device dropdown). Daily use after that is just:

```bash
python -m siliconshield.gui
```

`--models /path/to/models.local.json` overrides the default file and
`SILICONSHIELD_GUI_MODELS` sits between them. `python -m
siliconshield.gui --help` works without Qt, Torch, or CUDA installed.
`--init` writes the registry and exits without opening the window
(advanced; normally omit it).

The registry file lives outside Git. See
`configs/gui-models.example.json` for the illustrative shape with
placeholder paths.

## Workflow

1. Select a decision model and Device GPU or CPU.
2. Tab `Single file`: pick a `Demo:` preset and click `Load Demo`
   (authored synthetics in `examples/gui-demo/`, never training data),
   or choose a `.v`/`.sv` file and enter its requirement.
   `legitimate-access`, `debug-lock-respected`, and
   `write-once-then-locked` are the should-satisfy choices by
   construction (no promise about model output).
3. Click Load and wait for ready confirmation with load time
   (`Ready on cuda:0 in 12.3s` or `Ready on cpu in 45.6s`).
4. Review the read-only preview, requirement, and assumptions.
5. Click Run Analysis and read the three-way assessment, scores, and
   analysis time (`Analysis time: 1.2s Device: … Model: …`).
6. Use Unload to release the worker and its GPU context.

Tab `Large file scan` reuses the already-loaded worker (no second
model load). Point it at a big `.v`/`.sv` file (up to 2 MiB),
enter the requirement, and click Scan:

* With `Use Muse to select excerpts` checked (needs the `muse` CLI
  login), Muse Spark 1.3 Contributor picks the relevant excerpts and
  the decision model assesses each excerpt in turn.
* The scanner first isolates complete `module` through `endmodule`
  regions. This keeps one verdict tied to one coherent RTL scope.
  Modules that exceed the input bound, files without usable module
  boundaries, and malformed boundaries use deterministic overlapping
  line windows instead. Every fallback excerpt is marked incomplete in
  its scan metadata.
* Unchecked (or when Muse is unavailable or fails), all deterministic
  excerpts are assessed directly. The fallback is stated explicitly
  and is never silent.
* The table shows per-excerpt verdict, scores, time, Muse reason, and a
  `Verified` column that always reads `Not verified` (or `Not run` for
  error rows): these are model assessments, never confirmed findings.
  The summary counts `Potentially flagged: X of Y (Z errors)`.
  Counting is deterministic Python on validated verdicts, never an
  LLM tally. A scan stops on the first excerpt error (reload to
  retry); Stop halts between excerpts without unloading the model.
* Click any table row to view its exact design excerpt below the
  table; drag the pane dividers to give the table or the excerpt
  more room.

The public scan preset is `Peripheral configuration bank`, a label-free
twelve-module design under one privilege requirement. Its source contains no
expected verdicts or answer-revealing module names. Authored answers live only
in fixture tests and are not sent to the decision model or Muse selector. The
preset demonstrates a measured strength and is not an accuracy benchmark.

All panes are resizable: drag table column borders (the Muse-reason
column stretches), drag the divider between the tabs and the Details
log, drag the divider between the scan inputs and the scan table, and
resize the window (minimum 900×650). Dropdowns and file fields
already expand with the window; the scan requirement/assumption boxes
are capped so the table keeps the room.

The window shows honest progress (`Checking model files`, `Loading
model`, `Warming up`) without invented percentages. Load is explicit;
changing the selection requires loading before Run. One analysis runs
at a time. Request and session ids prevent a delayed reply from
overwriting a newer run. Cancel terminates only the owned worker.

## Inputs and results

Design reads are bounded (1 MiB), UTF-8, and hashed. The model sees an
immutable snapshot in the shared `REQUIREMENT`/`ASSUMPTIONS`/`RTL`
format with the exact export-record security question, label order, and
`option_order=[0, 1, 2]`. The persistent worker resolves the checkpoint
contract at load and reuses it for warm-up and every analysis. Truncated
inputs produce `Input too long — no verdict produced`, never an
`insufficient_context` label.

Results show the three labels with scores, analysis time, actual
device, and model identity. Details carry artifact hashes, runtime
path, usage, and errors. Save Result is explicit and warns that the
export contains the analyzed RTL and requirement.

Neither authored demos nor custom files receive independent
verification in the window: every result is a model assessment.
A file chosen with Browse (anything outside the `Demo:` presets)
additionally carries the banner “Model assessment only. This custom
input has no requirement-bound check … must not be reported as a
confirmed finding.” To establish a violation, run the CLI
investigation workflow from [investigation](investigation.md) with
requirement-bound checks supplied for that design; without them the
conclusion stays `requirement_not_verified`.

## Troubleshooting a load refusal

`Checkpoint file '<name>' hash mismatch; refusing load` means the
sealed export changed after the registry was created. The worker
never modifies the sealed export (it stages a private copy), so a
mismatch points at a direct `laya.agent.load(sealed_path)` elsewhere:
the pinned runtime's `_fix_tokenizer_config()` rewrites
`tokenizer/tokenizer_config.json` in place on load. Check with
`sha256sum` against the registry's `expected_files` and compare
`stat` times; weights (`model.safetensors`) are the critical file —
if only the tokenizer config differs, re-register explicitly with the
single launch command (it re-hashes and merges by model id), then
Load again. Never hand-edit digests into the registry file.

Scope: `Model assessment of the supplied RTL and requirement. No
verification tool was run.`

## Qualification

Three levels: fixture behavior (synthetic registries, fake workers,
bounded reads, truncation, CPU-fallback, and cleanup tests), real
desktop behavior (offscreen and on-screen inspection), and real GPU
inference (sealed export hashes, load timing, repeated analyses on
one worker, SDK parity within tolerance, oversized-input refusal,
and worker/GPU release). Fixture tests never touch live services,
GPUs, or research files. `gui/qualification.py` adds a fixture-only
product gate over eight cases (locked, bypass, reset-exempt,
reset-inclusive, unresolved policy, helper, relevant/irrelevant
omission) checking state shape and contract bytes without weights.
