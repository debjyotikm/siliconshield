# Python style standard

[Documentation index](README.md)

Derived from the Google Python Style Guide, adapted to this repository. The
guide's 80-column target has exceptions, and its advice to reconsider functions
over roughly 40 lines is guidance, not a rule to split every function
mechanically. `ruff format` follows Black-compatible formatting. It is not a
complete Google-style compliance checker. The written policy and review checks
below cover what automation cannot.

## Line length and formatting

- Target 80 columns. `ruff format` is authoritative for layout. Do not
  hand-format against it.
- Exceptions, with a short comment explaining why: embedded RTL, prompt
  templates, regular expressions, and serialized contract strings where
  splitting hurts readability or risks behavior. Long URLs are also exempt.
- Never reflow string contents to satisfy a width complaint.

## Imports

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

## Naming

- `CapWords` for classes, `snake_case` for functions/variables,
  `UPPER_SNAKE` for module constants. Test functions describe behavior:
  `test_<unit>_<condition>_<expectation>`.

## Documentation

- Google-style docstrings required for public functions, classes, methods
  with logic, and every function over roughly 40 lines. Trivial private
  helpers (a few lines, self-evident) are exempt.
- Document inputs, outputs, side effects, failure conditions, and units.
- Comments explain decisions and assumptions, not what the code says.
  User-facing errors state what failed and how to resolve it.

## Typing

- Annotate parameters and returns on new and changed public interfaces.
- Use `TypedDict` for existing dictionary contracts (run manifests,
  observations, decisions, grades), `dataclasses` where a configuration
  object helps, and `typing.Protocol` for interchangeable clients
  (chat clients, runners). Do not satisfy the checker with `Any`.
- Strict `mypy` checking applies to the modules listed under
  `[[tool.mypy.overrides]]` in `pyproject.toml`. New code in those modules
  must pass without adding untyped definitions. In other modules, annotate
  new and changed public interfaces to prepare for stricter checking.

## Exceptions and validation

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

## Resource handling

- Context managers for files, locks, and subprocesses. Atomic writes for
  receipts and checkpoints (`atomic_json`, `atomic_bytes`). Never leave a
  half-written manifest behind.

## Function structure

- One responsibility per function. Separate CLI parsing, configuration
  validation, data preparation, model construction, training, evaluation,
  checkpointing, and export into identifiable components.
- Mutable module-level state is configuration that has not been made
  explicit yet: replace it with explicit configuration objects passed
  to the functions that need it.

## Compatibility

- Minimum Python is 3.11 (`requires-python = ">=3.11"`, Ruff
  `target-version = "py311"`). No 3.12-only syntax (`type` statements,
  etc.). Verify with the 3.11 environment, not just the default 3.12.

## Tests

- Authored synthetic fixtures only. Fixture tests never touch live
  services, GPUs, or research files. Datasets, weights, checkpoints,
  transcripts, and credentials live outside Git (see `AGENTS.md`).

## Tooling

Development tools are pinned separately from GPU training requirements
(`requirements/style.txt`):

```bash
ruff format --check siliconshield tests tools
ruff check siliconshield tests tools --select=E9,F63,F7,F82,F401,F811,I --line-length=80
python -m compileall -q siliconshield tests tools
python -m mypy siliconshield
python -m pylint siliconshield --rcfile=.pylintrc
```

`ruff check` with the narrow gate above must pass. `pylint` uses a
Google-derived configuration to flag code for review. Review its findings
individually. Some rules conflict with numerical code, optional imports, or test
conventions, so a warning is not automatically a bug.
