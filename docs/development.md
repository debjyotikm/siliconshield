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

The submission tree carries a curated submission gate covering installation,
routing, benchmark, studies, operators, selection, the Glimmer bridge, and
browser request boundaries.
Torch, solver, campaign-history, and optional-dependency suites live
outside the submission tree as local-only fixtures. Pylint is a review
signal rather than a blanket score gate.

## Test map

| Area | Tests |
|---|---|
| Investigations, classification, patch checks | `test_app_workflow.py` |
| Browser HTTP, CPU dispatch, routing, and toy-job boundaries | `test_web_demo.py` |
| Installed package and CLI imports | `test_installation.py` |
| Router assessment and arbitration binding | `test_multi_expert.py`, `test_expert_safety.py` |
| Area benchmark rerun and metric math | `test_benchmark.py` |
| Study freeze, manifest binding, override refusal | `test_decision_study.py`, `test_study_enforcement.py` |
| Job authority, preflight, recipe binding | `test_operator.py`, `test_glimmer.py` |
| Promotion gates and pair preservation | `test_selection.py` |
| Router bridge sessions | `test_glimmer_router.py` |
| Export privacy/allowlist | `test_export_check.py` |

The broader fixture suites are local-only and not part of the submission
tree. Run them from a preserved checkout with the matching environment:
the security-model interpreter with `LAYA_ROOT` for training suites,
Yosys and Icarus on `PATH` for checker integration, and the pinned
optional dependency for Harness² fixtures. Source fixtures must stay
outside Git.

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
6. Review the technical claims as an RTL and ML interface: distinguish requests
   from accepted transactions, declared assumptions from missing logic, and
   predictions from executed checks. Name the measured population and denominator.
7. Keep capability, implementation, and qualification status distinct. A registered
   expert, a completed job, and a successful release evaluation are different results.
8. Check the export allowlist and private-content scan before publishing.

Diagrams have a multipage draw.io source and checked-in SVG previews under
[docs/diagrams](diagrams/README.md). The [white-paper figures](whitepaper/figures/README.md)
add research plots and their editable sources. Update source and preview
in the same change and inspect the rendered output. Keep one canonical diagram per flow, link it from
other pages, and describe its meaning in adjacent text for accessibility.

The current CI gate checks source, fixtures, and installed entry points. It does
not automatically validate Markdown links, render draw.io/SVG figures, or
execute every documentation example. Include those review results explicitly when relevant.
