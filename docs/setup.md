# Setup

[Documentation index](README.md) · [Investigation](investigation.md) ·
[Training](security-model-training.md) · [Operations](operations.md)

Choose one environment below. Installing the Python package does not install
model weights, GPU runtimes, EDA executables, or a campaign service. Run commands
from the repository root and replace `/path/to/` placeholders before use.

## Environment requirements

| Workflow | Requirements | First check |
|---|---|---|
| Authored investigation | Python 3.11+, Verilator supporting `--binary --timing`, C++ compiler, make | `python -m siliconshield.app --list` |
| Controller tools and fixtures | Python 3.11 or 3.12 with pinned style dependencies | `bash tools/check.sh` |
| Harness² procedures | Controller environment plus `requirements/harness2.txt` and Muse CLI/login for live agent calls | Worker `--help`, then fixture qualification |
| Security-model training | Torch/CUDA environment, `requirements/core.txt`, compatible LAYA runtime and local export | Trainer `--help`, then registered preflight |
| Glimmer training/reference serving | NVIDIA container runtime, pinned image, local base/adapter, shared GPU lease | Identity and dataset preflight |
| Native Glimmer inference | Separately qualified vLLM build with native Glimmer parsers | Launcher `--dry-run`, then serving qualification |

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

## Glimmer image and workspace

Build the checked-in image recipe:

```bash
docker build -t siliconshield-glimmer deploy/spark/glimmer
```

The Dockerfile pins Python packages on an NVIDIA image tag. Capture the resolved
image ID for each execution. The tag alone is not an immutable image identity.
The image supplies Transformers training/reference serving, not the separately
qualified vLLM inference build.

Prepare a private workspace containing local model snapshots, signed episode
partitions, run output directories, and a runtime lock directory. Use the base
revision declared by `MODEL_REVISION` in
[identity.py](../siliconshield/training/glimmer/identity.py). Preserve the snapshot's
`.cache/huggingface/download/` metadata because revision verification uses it.

For a new dataset trust domain only, create a key in an existing private directory.
This example refuses to overwrite an existing key and creates it with owner-only
permissions:

```python
import os
import secrets

fd = os.open("/path/to/private/dataset-key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "wb") as stream:
    stream.write(secrets.token_bytes(32))
```

Set `SILICONSHIELD_DATASET_KEY` to the existing private signing-key file before
building episodes, and use that same key for training/evaluation. Do not replace
the key to bypass a signature failure. Dataset generation is covered in
[Glimmer training](glimmer.md). Report conversion alone does not guarantee the
operator curriculum's required action coverage.

The following launch template mounts the key read-only. It assumes the source
snapshot and workspace already exist and the dataset was signed with this key:

```bash
export SILICONSHIELD_EXPECTED_CODE_IDENTITY="$(PYTHONPATH=/path/to/source-snapshot \
  python -c 'from siliconshield.training.glimmer.identity import code_identity; print(code_identity())')"
export SILICONSHIELD_IMAGE_DIGEST="$(docker image inspect --format '{{.Id}}' siliconshield-glimmer)"
docker run --rm -it --gpus all --shm-size=8g --network host \
  -e SILICONSHIELD_EXPECTED_CODE_IDENTITY \
  -e SILICONSHIELD_IMAGE_DIGEST \
  -e SILICONSHIELD_DATASET_KEY=/run/secrets/dataset-key \
  -e SILICONSHIELD_GPU_LOCK=/workspace/runtime/glimmer-gpu.lock \
  --mount type=bind,src=/path/to/dataset-key,dst=/run/secrets/dataset-key,readonly \
  --mount type=bind,src=/path/to/glimmer-workspace,dst=/workspace \
  --mount type=bind,src=/path/to/source-snapshot,dst=/workspace/source,readonly \
  --workdir /workspace/source --entrypoint bash siliconshield-glimmer
```

Capture source identity from the exact snapshot mounted in the container. A
source change requires new preflight, not replacement of the old receipt's hash.
`--gpus all` exposes every GPU on the chosen host: use this template only for the
intended dedicated Glimmer resource window. The shared lease coordinates
cooperating training, serving, and evaluation processes, not arbitrary GPU users.

Host networking lets a controller on this Linux host reach a loopback-bound
server in the container. Remote clients need a separately configured transport.
The example does not expose a public endpoint.

## Qualify training and resume

Inside the prepared Glimmer container, use a signed qualification dataset:

```bash
python -m siliconshield.training.glimmer.train \
  --model /workspace/models/muse-glimmer-30b \
  --data /workspace/data/pilot --output /workspace/runs/pilot \
  --max-length 2048 --task-eval-every 100 --max-steps 2
```

Inspect identity, coverage, checkpoint, and finite-loss receipts. Then resume the
same run with the same settings, omitting the temporary step limit:

```bash
python -m siliconshield.training.glimmer.train \
  --model /workspace/models/muse-glimmer-30b \
  --data /workspace/data/pilot --output /workspace/runs/pilot \
  --max-length 2048 --task-eval-every 100 --resume
```

Require a successful resume-parity event before treating recovery as qualified.
These commands use the default `qualification` purpose. For operator training,
use `--purpose operator` and meet the implemented action/lineage coverage
requirements and task-selection contract. A two-update smoke run is not
controller-quality evidence.

See [Glimmer training](glimmer.md) for continuation, DPO, external selection,
and evaluation. Release the training GPU lease before starting
[serving](inference.md) or evaluation on the same device.

## Prepare an owned security-model job

Copy [the job profile](../configs/glimmer-security.example.json) outside Git.
Set its interpreter, frozen source, runtime checkout, model, dataset, run root,
three GPU indices, and defaults. Profiles sharing GPUs must use the same
`resource_root`. Optional `recipe`, `reporting_contract`, `review_ledger`,
and `review_packets` keys name reviewed local files. `defaults` may pin
`question_contract` (`v0` or `v1`); training validates every row against it
and records it on export. `defaults` may also set
`plateau_requires_full_exposure` to block plateau interventions until
delivered draws cover a full first pass.

Inspect the prepared command and hashes without launching:

```bash
python -m siliconshield.research_loop.execution.security_jobs \
  --profile /path/to/security.local.json --experiment-id experiment-a
```

This preparation writes local receipts. Add `--launch` only when ready to submit
through the operator. Active GPU processes can cause preflight to refuse.
Resource contention at submission can return a waiting result. Inspect the
returned state before retrying the same experiment.

The service explicitly receives `CUDA_VISIBLE_DEVICES`, `LAYA_ROOT`, and
`SILICONSHIELD_SECURITY_RECIPE` when the profile supplies `recipe`. It does not
inherit arbitrary shell variables. Passing a recipe only in your interactive
shell is not equivalent to registering it in the profile.

For repeated rounds, follow [campaign integration](campaigns.md). For status,
stop, and recovery procedures, use [operations](operations.md). Never launch a
second driver merely because the first client's connection ended.
