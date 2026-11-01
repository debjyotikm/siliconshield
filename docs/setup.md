# Setup

[Documentation index](README.md) · [Investigation](investigation.md) ·
[Training](security-model-training.md) · [Operations](operations.md)

Choose one environment below. Installing the Python package does not install
model weights, GPU runtimes, EDA executables, or a campaign service. Run commands
from the repository root and replace `/path/to/` placeholders before use.

- **Assess RTL:** prepare the controller, follow [pinned releases](releases.md),
  and register measured profiles. Specialist assessment needs no EDA toolchain.
- **Check a repair:** add Verilator and the C++ toolchain for the investigation
  application.
- **Train a model:** use a separate CUDA/runtime environment and an accepted
  dataset under a [decision study](decision-study.md).

## Environment requirements

| Workflow | Requirements | First check |
|---|---|---|
| Authored investigation | Python 3.11+, Verilator supporting `--binary --timing`, C++ compiler, make | `python -m siliconshield.app --list` |
| Controller tools and fixtures | Python 3.11 or 3.12 with pinned style dependencies | `bash tools/check.sh` |
| Harness² procedures | Controller environment plus `requirements/harness2.txt` and Muse CLI/login for live agent calls | Worker `--help`, then fixture qualification |
| Security-model training | Torch/CUDA environment, `requirements/core.txt`, compatible LAYA runtime and local export | Trainer `--help`, then registered preflight |
| Glimmer training/reference serving | NVIDIA container runtime, pinned image, local base/adapter, shared GPU lease | Identity and dataset preflight |
| Native Glimmer inference | Separately qualified vLLM build with native Glimmer parsers | Launcher `--dry-run`, then serving qualification |
| Expert router and benchmark | Lightweight controller plus local model-worker interpreters and checkpoints. `requirements/benchmark.txt` for Hub downloads. Muse CLI/login only for arbitration. | `python -m siliconshield.experts --help`, `python -m siliconshield.benchmark --help` |

Linux with systemd user services is required for the shipped owned-job path. The
portable Python fixtures do not require systemd or GPUs. Check that the selected
CUDA image supports your device. This repository does not install a compatible
GPU driver.

## Smallest installation: investigation

```bash
python3.11 -m venv .venv-app
source .venv-app/bin/activate
python -m pip install -e .
python -m siliconshield.app --list
verilator --version
make --version
c++ --version
```

The case listing should include `unauthorized-write`, `lock-bypass-second-path`,
`legitimate-access`, and `missing-context`. If the compiler or Verilator command
is unavailable, install the host build toolchain before running executable checks.
Continue with the [investigation walkthrough](investigation.md), including its
expected report fields. No model download is needed for the authored demo.

## Controller and development environment

```bash
python3.11 -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install -e . -r requirements/style.txt
bash tools/check.sh
```

For Harness², add its pinned dependency to this environment or a separate worker
environment:

```bash
python -m pip install -r requirements/harness2.txt
python -m siliconshield.research_loop.procedures.muse_harness2 --help
```

The controller can build episodes and call an existing endpoint without loading
Glimmer weights. Optional Torch, solver, and Harness² tests skip when
dependencies are absent. Inspect the skip reasons. See
[development](development.md).

## Security-model environment

For inference with the published Contract-1 and Contract-2 decision models, use the
[release revisions and loading commands](releases.md). The templates below
also support custom exports and training environments.

Use a CUDA-compatible environment for the pins in `requirements/core.txt`.
Choose the appropriate platform wheel/index for Torch rather than assuming the
host's default installation is suitable. Install SiliconShield into that same
interpreter and identify a compatible residual-adapter runtime checkout:

```bash
/path/to/security-environment/bin/python -m pip install -e . -r requirements/core.txt
export LAYA_ROOT=/path/to/compatible-runtime
/path/to/security-environment/bin/python \
  siliconshield/training/laya/security_train.py --help
```

`LAYA_ROOT` must contain a `laya/` package. An installed compatible package is
also supported. Pin the revision. Residual adapters, checkpoint loading, input
encoding, and calibration are dependency contracts: a newer runtime that imports
successfully may still differ in behavior. Use the focused runtime tests in
[development](development.md) before training or finalization.

For dataset extraction, install `requirements/data.txt` and provide Yosys/Icarus
executables separately. Follow the [data pipeline](data-pipeline.md), then
[security-model training](security-model-training.md).

## Decision-model releases and routing

The model exports are available from
[Contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1)
and [Contract-2](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2).
Choose a release revision, preserve its files, and record the resolved commit
and full export hashes. Install the compatible runtime in the worker environment.
The Python package does not bundle weights or supply a standard Transformers
pipeline for them.

Register immutable model exports with interpreter, runtime, device, and full
file hashes using [expert qualification](expert-qualification.md). Attach a
capability profile for every model that should participate in routing. An
unprofiled entry can be benchmarked individually, but is not a routable expert.
Use the [router walkthrough](router.md#run-one-assessment) to supply a scoped
requirement and inspect the resulting evidence.

For benchmark downloads, install `requirements/benchmark.txt` in the controller
environment. The training and benchmark collections are separate Hub datasets.
See [training recipes](training-recipes.md) for lineage and [benchmark](benchmark.md)
for the pinned population and output contracts.

## Continue with the selected workflow

| Task | Procedure |
|---|---|
| Load the published models | [Pinned releases](releases.md) gives exact exports, runtime, and loading commands. |
| Build a Glimmer training workspace | [Image and workspace](glimmer.md#image-and-workspace) covers the container, signed data, source identity, and shared lease. |
| Qualify Glimmer training and recovery | [Training and resume qualification](glimmer.md#qualify-training-and-resume) provides the bounded smoke run. |
| Serve Glimmer | [Inference](inference.md) covers the separate native server and timing protocol. |
| Submit an owned decision-model job | [Campaign job preparation](campaigns.md#prepare-an-owned-security-model-job) covers profiles, resource ownership, and submission. |

For new training work, register a [decision study](decision-study.md) before
launch. Environment installation alone does not qualify its data or recipe.
