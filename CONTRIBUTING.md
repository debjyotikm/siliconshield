# Contributing to SiliconShield

## Source-only rule

This repository is source-only. Do not add research reports, datasets,
model weights, checkpoints, transcripts, populated indexes, local
configuration, machine identities, or credentials. Use authored synthetic
tests. Datasets, weights, and runtime artifacts live outside Git.
See `AGENTS.md`.

LAYA core is an external dependency (`LAYA_ROOT` or installed package);
do not change LAYA core APIs here. Trainers belong under
`siliconshield/training/`; the research loop calls them only through
`siliconshield/research_loop/adapters/`.

## Style

Derived from the Google Python Style Guide, adapted to this repository. The
guide's 80-column target has exceptions, and its advice to reconsider functions
over roughly 40 lines is guidance, not a rule to split every function
mechanically. `ruff format` follows Black-compatible formatting. It is not a
complete Google-style compliance checker. The written policy and review checks
below cover what automation cannot.

### Line length and formatting

- Target 80 columns. `ruff format` is authoritative for layout. Do not
  hand-format against it.
- Exceptions, with a short comment explaining why: embedded RTL, prompt
  templates, regular expressions, and serialized contract strings where
  splitting hurts readability or risks behavior. Long URLs are also exempt.
- Never reflow string contents to satisfy a width complaint.

### Imports

- Use absolute package imports (`from siliconshield... import ...`). Do not use
  relative imports. New modules use package imports unconditionally.
- Order imports as standard library, third-party, then first-party
  (`siliconshield`). The `I` lint selection checks this order.
- Module scope by default. Function-local imports are allowed only for
  optional heavy dependencies (`torch`, `transformers`, `peft`) so that
  lightweight CLI commands keep working without GPU packages. Each local
  import carries a one-line comment stating the reason.
- Never use `import *`. Never shadow an import with a local redefinition
  (`F811` is gated).
- Do not add new `sys.path` manipulation. Tests import the package.
  Existing compatibility paths are exceptions: the standalone trainer
  anchors its own package and external runtime, and the legacy training tests
  support by-path loading. Preserve their launch tests. Do not copy that pattern
  into ordinary package modules.
- File-based module loading (`importlib.spec_from_file_location`) is reserved
  for runtime-chosen experiment definitions (external cases modules, semantic
  recipe modules) and for test harnesses that must load a file by path.
  Ordinary package modules use package imports.
- Every CLI entry point must run as `python -m ...` from the repository
  root and from an installed wheel outside the checkout.

### Naming

- `CapWords` for classes, `snake_case` for functions/variables,
  `UPPER_SNAKE` for module constants. Test functions describe behavior:
  `test_<unit>_<condition>_<expectation>`.

### Documentation

- Google-style docstrings required for public functions, classes, methods
  with logic, and every function over roughly 40 lines. Trivial private
  helpers (a few lines, self-evident) are exempt.
- Document inputs, outputs, side effects, failure conditions, and units.
- Comments explain decisions and assumptions, not what the code says.
  User-facing errors state what failed and how to resolve it.

### Typing

- Annotate parameters and returns on new and changed public interfaces.
- Use `TypedDict` for existing dictionary contracts (run manifests,
  observations, decisions, grades), `dataclasses` where a configuration
  object helps, and `typing.Protocol` for interchangeable clients
  (chat clients, runners). Do not satisfy the checker with `Any`.
- Strict `mypy` checking applies to the modules listed under
  `[[tool.mypy.overrides]]` in `pyproject.toml`. New code in those modules
  must pass without adding untyped definitions. In other modules, annotate
  new and changed public interfaces to prepare for stricter checking.

### Exceptions and validation

- Raise explicit exceptions (`ValueError`, `RuntimeError`, ...) with
  actionable messages. Never use bare `except:`.
- Keep `try` blocks narrow. Review every broad `except Exception`
  individually. Endpoint isolation and terminal-receipt handling are
  legitimate broad catches. Most other code is not.
- Raise an exception when an application-critical check fails. Do not use
  `assert` for these checks because it disappears under `python -O`.
  `assert` is for internal invariants only.
- Preserve machine-readable `stdout` in trainers (JSON lines for receipts
  and events). Human diagnostics go to logs, not the receipt stream.

### Resource handling

- Context managers for files, locks, and subprocesses. Atomic writes for
  receipts and checkpoints (`atomic_json`, `atomic_bytes`). Never leave a
  half-written manifest behind.

### Function structure

- One responsibility per function. Separate CLI parsing, configuration
  validation, data preparation, model construction, training, evaluation,
  checkpointing, and export into identifiable components.
- Mutable module-level state is configuration that has not been made
  explicit yet: replace it with explicit configuration objects passed
  to the functions that need it.

### Compatibility

- Minimum Python is 3.11 (`requires-python = ">=3.11"`, Ruff
  `target-version = "py311"`). No 3.12-only syntax (`type` statements,
  etc.). Verify with the 3.11 environment, not just the default 3.12.

### Local gate

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

Apply the [source-only contribution policy](#source-only-rule)
to staged source, documentation, and examples. Review links and example paths
for accidental private references as well as obvious binary artifacts.

## Documentation and figure checks

Before accepting a documentation change:

1. Check commands, environment variables, schema fields, and callbacks against
   their implementation. State prerequisites, side effects, and expected output.
2. Resolve local links and heading anchors. Syntax-check fenced shell and Python
   examples, including the [study walkthrough](docs/study-walkthrough.md) and
   [qualification example](examples/qualify_experts.py).
3. Keep one maintained procedure for each workflow and link to it from other
   guides. Distinguish model predictions, executed checks, and measured quality
   using the [evaluation protocol](docs/evaluation.md).
4. Update editable diagrams and rendered previews together. Inspect them at
   normal page width. Rendering instructions belong to the
   [architecture figure index](docs/diagrams/README.md) and
   [whitepaper figure index](docs/whitepaper/figures/README.md).
5. State which examples were actually executed. Documentation checks do not
   authorize GPU launches, stop/resume actions, or calls to live agents.

The lightweight gate does not validate Markdown links, render figures, or run
every documented workflow. Report those checks separately. A clean link or
syntax check does not establish runnable onboarding or scientific accuracy.

## Commits

- Keep mechanical cleanup (formatting, import order, straightforward
  renames) in separate commits from behavior changes, one package at a
  time.
- Verify formatting preserves executable structure and embedded strings
  (prompts, generated RTL, regular expressions, serialized contracts).
- Verify CPU-only import behavior still works after import changes.
- New CLI entry points must work as `python -m ...` from the repository
  root and from an installed wheel outside the checkout; extend the
  entry-point tests in `tests/test_installation.py`.
- Before any commit: stage the intended files and run
  `python tools/export_check.py --tracked`; keep `tools/release_files.txt`
  accurate for new source/configuration files.

## Versioning

- The package version lives only in `pyproject.toml`; the package
  reports it via installed metadata with a `0.0.0+source` fallback for
  uninstalled checkouts. Tests read the expectation from `pyproject.toml`
  (`read_project_version`); never hardcode the version in tests.
- Serialized contract identifiers live in `siliconshield/schemas.py`.
  Never edit an existing value; add a new identifier for a new contract
  version, with a compatibility test pinning the value.

Do not launch training, create remotes, or push without explicit
instruction.
