# Working on SiliconShield

Keep this repository source-only. Do not add research reports, datasets,
model weights, checkpoints, transcripts, populated indexes, local
configuration, machine identities, or credentials. Use authored synthetic
tests. Datasets, weights, and runtime artifacts live outside Git.

LAYA core is an external dependency (`LAYA_ROOT` or installed package); do
not change LAYA core APIs in this repository. Experiment-specific dataset
catalogs, evaluation cases, and recipes must stay in local manifests outside
the public tree (see `examples/` for illustrative shapes); never embed them
in source, tests, or docs.

Preserve pinned versions in `requirements/` and
`deploy/spark/glimmer/Dockerfile`. Trainers belong under
`siliconshield/training/`; the research loop calls them only through
`siliconshield/research_loop/adapters/`.

Follow the repository standard in `CONTRIBUTING.md` and
`docs/python_style.md` (Google-derived: imports, documentation, typing,
exceptions, function structure; 80-column target with documented
exceptions). Development tools are pinned in `requirements/style.txt`.

Before declaring a change done, run the full lightweight gate (or the
equivalent commands): `bash tools/check.sh` (formatting, narrow lint plus
import order, compilation, `mypy`, CPU fixture tests, wheel install).
CI runs it on Python 3.11 and 3.12.

- Run the focused tests in `docs/setup.md` (fixture tests never touch live
  services, GPUs, or research files).
- Run Ruff: `--select=E9,F63,F7,F82,F401,F811,I --line-length=80`.
- Check formatting: `ruff format --check siliconshield tests tools`.
- Compile `siliconshield/`, `tests/` and `tools/`.
- `python -m mypy siliconshield` must pass; strict modules are listed in
  `pyproject.toml`. `python -m pylint --rcfile=.pylintrc` is a review
  signal, not a zero gate.
- Before any commit: stage the intended files and run
  `python tools/export_check.py` (add `--private-word` markers as needed);
  it inspects staged bytes for credentials, GPU identities, and artifacts.
- State clearly whether a result is a unit, integration, serving, GPU
  execution, or model-quality result.

Do not launch training, create remotes, or push without explicit instruction.
