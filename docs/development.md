# Development and testing

[Documentation index](README.md)

Start with [CONTRIBUTING.md](../CONTRIBUTING.md), [AGENTS.md](../AGENTS.md), and
the [Python style standard](python_style.md). Keep source changes separate from
private experiments and machine configuration.

## Local gate

```bash
python3.11 -m venv .venv-style
source .venv-style/bin/activate
python -m pip install -r requirements/style.txt
bash tools/check.sh
```

The gate checks Ruff format, narrow lint including import order, compilation,
mypy, CPU fixtures, and wheel installation outside the checkout. CI runs it on
Python 3.11 and 3.12. Keep the code compatible with Python 3.11.

The lightweight gate excludes `test_security_training.py` and
`test_fresh_selection.py`, which import Torch and the external runtime; run
those two suites with the security-model environment instead (see the
runtime-specific fixtures below). Optional dependencies and toolchains cause
explicit skips in other suites. A skipped integration is not a passing live
qualification. Pylint is a review signal rather than a blanket score gate.

## Test map

| Area | Tests |
|---|---|
| Investigations, classification, patch checks | `test_app_workflow.py` |
| Installed package and CLI imports | `test_installation.py`, `test_laya_public_examples.py` |
| Proposal labels and solver adapter | `test_checker_adapter.py`, `test_proposer_ingest.py`, `test_verification_bounds.py` |
| Dataset identity and lineage | `test_dataset_versions.py`, `test_independence_gates.py`, `test_laya_new_study.py` |
| Question contracts and review ledger | `test_question_contracts.py`, `test_review_ledger.py` |
| Trainer, sampler, checkpoint export | `test_security_training.py`, `test_coverage_audit.py`, `test_ddp_guard.py`, `test_inventory.py`, `test_fresh_selection.py` |
| Protected objectives and teacher loss | `test_checkpoint_selection.py`, `test_development_views.py`, `test_replay_preservation.py` |
| Reports and uncertainty | `test_security_reporting.py` |
| Campaign transactions and supervision | `test_laya_campaign.py`, `test_campaign_supervisor.py` |
| Job authority, reattachment, recovery | `test_operator_contract.py`, `test_glimmer_jobs.py`, `test_glimmer_operator.py` |
| Glimmer episodes, identity, training | `test_operator_curriculum.py`, `test_identity_and_splits.py`, `test_glimmer_training.py` |
| Action-class selection and checkpoint review | `test_glimmer_selection.py`, `test_checkpoint_assessment.py` |
| Inference interfaces and timing | `test_glimmer_serving.py`, `test_inference_metrics.py` |
| GUI fixtures and qualification | `test_gui_contracts.py`, `test_gui_controller.py`, `test_gui_demos.py`, `test_gui_entry.py`, `test_gui_lifecycle.py`, `test_gui_qualification.py`, `test_gui_registry.py`, `test_gui_scan.py`, `test_gui_setup.py`, `test_gui_assessment_clearing.py` |
| Harness² integration | `test_glimmer_harness2.py`, `test_muse_harness2.py`, `test_muse_procedures.py` |
| Export privacy/allowlist | `test_export_check.py` |

Run runtime-specific fixtures with the appropriate environment:

```bash
CUDA_VISIBLE_DEVICES='' LAYA_ROOT=/path/to/compatible-runtime \
  /path/to/security-environment/bin/python -m pytest -q \
  tests/test_security_training.py tests/test_replay_preservation.py
```

Put Yosys and Icarus on `PATH` to exercise the checker integration tests. Some
older builder tests additionally use `LAYA_TEST_YOSYS` or `LAYA_TEST_SOURCES`.
Inspect their skip reasons. Source fixtures must stay outside Git. Harness²
integration fixtures require the pinned optional dependency.

## Changing interfaces

Add a focused regression reproducing changed behavior, then run the affected
suite and the required lightweight gate. Use synthetic fixtures. Keep expensive
training, serving, and external-agent trials separate from unit checks.

A new command must work from an installed wheel outside the checkout as well as
from the repository. Update installation/CLI tests. A new serialized contract
needs explicit version ownership and compatibility expectations. Changes to
selection, resume, or promotion require tests for refusal and recovery paths,
not only the successful path.

Trainer and campaign selection should share predicates where semantics match.
Test both sides against the same count-derived counterexamples. Identity changes
need tampering/drift tests. Writes spanning multiple artifacts need interrupted
publication or recovery tests.

## Release review

Review staged bytes, not just the working copy. Keep
`tools/release_files.txt` synchronized with intended portable files. For incremental
commits, use:

```bash
python tools/export_check.py --tracked
```

This inspects the entire Git index against the release list. The default mode
expects the listed files in the staged changed set and is useful for a first
full-tree release. Neither mode audits ignored private directories. Add private
markers with repeated `--private-word` options when reviewing a release.

Keep datasets, reports, model files, transcripts, credentials, local recipes,
source-derived experiment catalogs, and workstation identities out of commits.
Review links, example paths, test names, and documentation for accidental private
references as well as obvious binary artifacts.

For documentation, verify local links, CLI flags, environment names, and schema
claims against the implementation. Label proposed integrations as planned. Do
not present a Python API as a shipped command. Published accuracy and speed
claims require separately reviewed evidence and a pinned scope.

## Documentation acceptance

Review documentation changes as interface changes:

1. Identify the reader, prerequisites, inputs, side effects, expected output,
   and next step for each runnable procedure.
2. Check arguments and environment forwarding against the parser and launcher,
   not another guide. Check callbacks and return contracts against their owners.
3. Resolve file links and section anchors. Check fenced shell/Python syntax.
   Show code without credentials, workstation paths, or experiment values.
4. Run the smallest documented workflow in a clean environment. For commands
   requiring GPUs, private data, or live agents, state the qualification still
   needed instead of treating `--help` as an execution test.
5. Update the canonical guide and link to it from setup/reference pages. Avoid
   independent copies of recipes or acceptance requirements that can diverge.
6. Check the export allowlist and private-content scan before publishing.

The current CI gate checks source, fixtures, and installed entry points. It does
not automatically validate Markdown links, render Mermaid, or execute every
documentation example. Include those review results explicitly when relevant.
