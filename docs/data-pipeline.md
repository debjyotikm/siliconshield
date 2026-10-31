# Building and qualifying security data

[Documentation index](README.md)

The training unit is a decision about **RTL, a requirement, and assumptions**. A
file's source repository and CWE tag do not determine that decision. Generated
mutations may preserve behavior, and unresolved solver outcomes are not evidence
of missing context.

## Sources and builders

| Module under `training/laya/` | Role |
|---|---|
| `build_security_data` | Catalog-driven cases, mutations, rendering, compilation, and solver evidence with declared bounds. |
| `source_contracts` | Optional parser-based extraction of local transition contracts. |
| `build_source_security` | Validate a source-backed catalog and write partitions. |
| `expand_security_data` | Add scenarios from a supplied semantic recipe. |
| `augment_security_policies` | Requirement/policy counterfactuals. |
| `augment_security_guards` | Guard-related variants with explicit case budget and seed. |
| `build_security_external` | Separate evaluation construction through a supplied cases provider. |
| `checker_adapter` / `proposer_ingest` | Check agent proposals and publish accepted evidence. |
| `question_contracts` | Versioned model-visible question bytes and export records. |
| `review_ledger` | Independent English-to-assertion review and template coverage. |
| `coverage_audit` | Triplet integrity, quota schedules, and delivered exposure. |
| `inventory` | Eligible origins/groups inventory and broad/narrow-pilot check. |
| `ddp_guard` | Single-backward plus `no_sync` coherence gate. |
| `dataset_versions` | Pin split bytes, bind pools, and record dataset acceptance. |

OpenTitan and HardSecBench are supported source materials, not bundled datasets.
Obtain their source and retain revision, license, and extraction provenance in
local catalogs. The authored fallbacks exercise interfaces. They do not
reproduce private experiment catalogs or establish broad coverage of either
project.

## Catalog workflow

1. Identify an implementation and a precise local requirement. Specify reset,
   clock, and other verification assumptions.
2. Keep satisfying, violating, rewritten, and missing-context variants attached
   to their original lineage. Assign that lineage to one partition before
   expanding variants.
3. Compile complete examples and run the registered assertion. Require proof
   evidence for satisfaction and a reachable counterexample for violation.
4. For missing context, show that filling the declared omission in different
   valid ways produces opposing established answers.
5. Audit overlap, count definitions, label support, and complete model encoding.
6. Pin the finished dataset and its audit before training.

The catalog structure is illustrated in
[security_cases.example.json](../examples/security_cases.example.json).
Source-backed construction uses:

```bash
python -m siliconshield.training.laya.build_source_security \
  --catalog /path/to/reviewed-catalog.json \
  --output /path/to/new-dataset \
  --verification /path/to/evidence \
  --yosys /path/to/yosys --iverilog /path/to/iverilog
```

Install `requirements/data.txt` for parser-based extraction. Solvers and the
compiler are separate executables. See each builder's `--help`. Local providers
and recipes are executable configuration and must be trusted.

## Checking agent proposals

`FixedChecker` loads a registry mapping contract IDs to exact requirement text,
one assertion, scope, reset, and optional `bound_cycles`/`induction_steps`.
A proposal cannot substitute its own assertion or change the registered
assumptions. Both sequential reset-conditioned and combinational scopes have
explicit assumption text. Bounds default to 6 counterexample cycles and 12
induction steps and are recorded in evidence.

A missing-context proposal names an undriven, used wire and completion
expressions. The checker preserves the declaration and uses a continuous
assignment to fill it, retaining width and expression grouping. Completions
must compile and yield opposing established verdicts. A timeout, unsupported
construct, or inconclusive proof is a refusal.

The following is a local integration example, not a dataset recipe:

```python
from pathlib import Path
from siliconshield.training.laya.checker_adapter import FixedChecker
from siliconshield.training.laya.proposer_ingest import ingest_proposals

checker = FixedChecker(
    Path("/path/to/contracts.json"),
    yosys="/path/to/yosys",
    iverilog="/path/to/iverilog",
    workdir=Path("/path/to/evidence"),
)
accounting = ingest_proposals(
    Path("/path/to/proposals"), checker, Path("/path/to/accepted-generations")
)
```

Ingestion checks the evidence identity, echoed contract and assumptions, RTL
hashes, and evidence required for each label. Established lineage, not a new
proposal name, determines family counts. Deduplication includes the contract and
assumptions.

A complete `gen-N/` contains accepted rows, rejections, accounting, and a
receipt. Only an atomic `current.json` switch publishes it. `load_current()`
rechecks output hashes. A malformed file causes ingestion to reject the whole
batch. Failures do not replace the previous good generation. Accepted proposals
still need downstream conversion and dataset qualification before entering the
trainer.

## Dataset versions and acceptance

`write_version(root, version, splits)` records existing split paths, hashes, and
row counts. It does not copy or protect their files. Do not modify those source
files after registration. `verify_version()` detects changes. Register a new
version when the data changes.

`accept_version()` binds an audit to the exact manifest. New releases must
carry the strict audit schema (`siliconshield-dataset-audit-v2`): matching
split hashes, zero reported exact-RTL, retired-RTL, lineage, and canonical
overlap, `max_group_share` at or below 5%, `max_mechanism_share` at or
below 15%, a bound `review_ledger_sha256` digest with a positive
`reviewed_templates` count, and a `release_class` (`broad` or
`narrow-pilot`) consistent with group counts. Audits without the strict
schema are refused for new acceptances; markers written under a legacy
audit stay readable through `is_accepted()` but cannot back a new
release. Acceptance records an audit supplied by the caller. It does not
conduct an independent semantic review of the labels. `is_accepted()`
revalidates the record, its audit, and referenced bytes.

Use the overlap helpers, lineage/canonical checks, concentration audit, and
complete runtime tokenization to produce the audit. The canonical screen maps
renamed identifiers to positional placeholders while keeping keywords,
operators, widths, and literals verbatim (comments stripped), so consistent
renames still collide. Absence of exact byte matches alone cannot establish
independence. Require template reviews from `review_ledger.py` before
admitting English requirements: requirement and assumption hashes are
recomputed from rendered row bytes, disagreements need an explicit
resolved-and-approved status with evidence references, and prose alone never
approves a template. Pin sampling pools before acceptance because pool
bindings change the version manifest.

## Sampling without changing provenance

Pools assign every TRAIN row exactly once. Weights determine grouped batch
quotas, while source and family identity remain on the rows for reporting.
Each pool declares a `mode`: `cycle` stratifies mechanism then design,
`flat` draws triplets uniformly. Triplet members must share a pool. Do not
rename a source to express a sampling priority. The trainer carries
per-rank draw queues across epochs and rebuilds them deterministically on
resume, so `simulate_coverage()` must be called with the same carried
queue state to predict delivered exposure honestly.

A correct pool quota does not guarantee a desired source mix inside that pool.
Run the real sampler over the intended update horizon and inspect source/family
visits, unique rows seen, repeat counts, and hard-example coverage. Report
delivered exposure at selected and terminal checkpoints with
`coverage_audit.delivered_exposure()`: an early selected checkpoint must not
be described as trained on rows only the terminal horizon reaches.
`reconstruct_seen()` replays draws exactly as the trainer consumed them for
resume-safe exposure accounting, and `exposure_gate_passed()` refuses
plateau interventions until a full first pass closes every triplet.
Parameter variants and repeated samples do not add independent families. Use
`inventory.summarize()` and `feasibility()` to classify a short inventory as
a narrower pilot with disclosed gaps.

## What coverage means

Distinguish decision rows, source files, catalog records, extracted design IDs,
split groups, and distinct RTL bodies. The manifest's `counts` fields and the
reporting functions use these separately. Mechanism coverage should also name
what was excluded: a checked local register transition is not whole-IP security
coverage or proof of CDC, timing, analog, or physical behavior.

Keep calibration and test material out of generation feedback. A diagnostic
view used to decide the next training change is spent as independent evidence.
See [evaluation](evaluation.md) and [training](security-model-training.md) for
continued studies and retired populations.
