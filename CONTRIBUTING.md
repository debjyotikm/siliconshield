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

Follow `docs/python_style.md` (Google-derived: imports, documentation,
typing, exceptions, function structure; 80-column target with documented
exceptions). Tooling:

```bash
ruff format --check siliconshield tests tools
ruff check siliconshield tests tools --select=E9,F63,F7,F82,F401,F811,I --line-length=80
python -m compileall -q siliconshield tests tools
python -m mypy siliconshield
python -m pylint siliconshield --rcfile=.pylintrc
```

`bash tools/check.sh` runs the full lightweight gate (formatting, lint,
compilation, types, CPU fixture tests, wheel install) in one step; CI runs
it on Python 3.11 and 3.12 (`.github/workflows/checks.yml`). GPU training,
serving, and model-quality evaluation are out of scope for the gate.

Development tools are pinned in `requirements/style.txt`, separately from
GPU training requirements. The supported minimum is Python 3.11; verify
with a 3.11 interpreter, not just the default 3.12.

## Tests

Run the focused tests in `docs/setup.md`. Fixture tests never touch live
services, GPUs, or research files. GPU execution and model quality need
separate validation; state clearly whether a result is a unit,
integration, serving, GPU execution, or model-quality result.

## Commits

- Keep mechanical cleanup (formatting, import order, straightforward
  renames) in separate commits from behavior changes, one package at a
  time.
- Verify formatting preserves executable structure and embedded strings
  (prompts, generated RTL, regular expressions, serialized contracts).
- Verify CPU-only import behavior still works after import changes.
- New CLI entry points must work as `python -m ...` from the repository
  root and from an installed wheel outside the checkout; extend the
  entry-point tests in `tests/test_laya_public_examples.py` and
  `tests/test_installation.py`.
- Before any commit: stage the intended files and run
  `python tools/export_check.py`; keep `tools/release_files.txt`
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
