#!/usr/bin/env bash
# Repeatable lightweight gate: formatting, linting, types, compilation,
# fixture tests, and wheel installation.
#
# GPU execution, serving, and model-quality evaluation are explicitly out
# of scope here, as is the SiliconShield-RTL training suite (needs torch,
# a LAYA checkout, and the Verilator/Yosys toolchain); see docs/setup.md.
# Torch- and solver-dependent tests skip cleanly inside this gate; the two
# torch-dependent training suites are excluded outright.
set -euo pipefail
cd "$(dirname "$0")/.."

ruff format --check siliconshield tests tools
ruff check siliconshield tests tools --select=E9,F63,F7,F82,F401,F811,I --line-length=80
python -m compileall -q siliconshield tests tools
python -m mypy siliconshield
python -m pytest tests -q --ignore=tests/test_security_training.py --ignore=tests/test_fresh_selection.py
python -m pytest tests/test_installation.py -q
