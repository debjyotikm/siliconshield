#!/usr/bin/env bash
# Repeatable lightweight gate: formatting, linting, types, compilation,
# fixture tests, and wheel installation.
#
# GPU execution, serving, and model-quality evaluation are explicitly out
# of scope here, as is the SiliconShield-RTL training suite (needs torch,
# a compatible local-model-runtime checkout, and the Verilator/Yosys toolchain); see docs/setup.md.
# The tree carries a curated submission gate. The broader
# fixture suite is a local-only artifact and is not part of this gate.
set -euo pipefail
cd "$(dirname "$0")/.."

ruff format --check siliconshield tests tools
ruff check siliconshield tests tools --select=E9,F63,F7,F82,F401,F811,I --line-length=80
python -m compileall -q siliconshield tests tools
python -m mypy siliconshield
python -m pytest tests -q
python -m pytest tests/test_installation.py -q
