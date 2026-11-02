# Pinned decision-model releases

[Documentation](README.md) · [Setup](setup.md) · [Demo](web-demo.md) · [Model card](siliconshield-rtl.md)

This is the reproducible inference path for the Contract-1 and Contract-2
RTL decision models. It is separate from reproducing their full training
histories. The source-controlled [release manifest](../configs/model-releases.toml) pins every inference payload
file and the compatible runtime. Downloads do not change a Hub repository.

## Tested combination

| Component | Pinned value |
|---|---|
| Worker Python | 3.11.15 |
| Torch / Transformers | 2.14.0+cu130 / 5.5.0 |
| Safetensors / NumPy | 0.8.0 / 2.4.6 |
| Inference | BF16, one row per worker request, sorting disabled, zero CPU fallbacks |
| Tested device | NVIDIA RTX PRO 4000 Blackwell, CUDA |
| Contract | `siliconshield-requirement-context-v1` |
| Label order | satisfies, violates, insufficient context |
| Budgets | 1,536 input tokens, 256 question/options tokens |

The release manifest owns the exact model and runtime identities.
This is a historical inference rehearsal, not a claim that every later source
revision was GPU-qualified. Record `git rev-parse HEAD` beside each new
measurement. The payload and runtime pins remain the reproducible artifact tuple. The rehearsal used fresh public downloads and a fresh runtime checkout
with an existing pinned worker environment. It did not rebuild the GPU driver
or install every dependency into an empty operating system.

## Download immutable exports

Run from the SiliconShield checkout with Python 3.11 or later. Choose a workspace
outside the checkout:

```bash
export SILICONSHIELD_WORKSPACE="$HOME/siliconshield-workspace"
mkdir -p "$SILICONSHIELD_WORKSPACE"
python tools/download_models.py --model all \
  --output "$SILICONSHIELD_WORKSPACE/models"
git clone https://github.com/debjyotikm/laya-next.git \
  "$SILICONSHIELD_WORKSPACE/laya-next"
SILICONSHIELD_RUNTIME_REVISION="$(python3 -c 'import tomllib; from pathlib import Path; print(tomllib.loads(Path("configs/model-releases.toml").read_text())["runtime"]["revision"])')"
git -C "$SILICONSHIELD_WORKSPACE/laya-next" checkout "$SILICONSHIELD_RUNTIME_REVISION"
```

Each export contains exactly six inference files. The downloader verifies full
SHA-256 values, publishes each file only after checking its bytes, and refuses
existing mismatched or extra files. Retrying reuses verified files. It prints the verified payload identities recorded in the release manifest.

The payload identities match the worker's `execution_identity.export_sha256`: SHA-256
of the sorted file-hash mapping encoded as compact JSON, with separators
`(',', ':')`. The application checkpoint-manifest helper uses spaced JSON and
therefore has a different aggregate digest over the same files. It excludes Hub presentation files such as
README and benchmark summaries. It must not be compared to a historical export
that contains a different file set. The manifest also contains the full weights
hash and question-record hash for each model.

Use the manifest revisions, not an assumed `v1.0.0` tag. Contract-1's older public card
referenced that nonexistent tag. The download commands above use the immutable revision directly.

## Worker environment and first assessment

Use a Python 3.11 environment compatible with the tested CUDA build. The training
and worker dependency pins remain in [requirements/core.txt](../requirements/core.txt).
The GPU driver must support that build. Do not replace the separate lightweight
controller environment with it.

```bash
python3.11 -m venv "$SILICONSHIELD_WORKSPACE/worker-env"
"$SILICONSHIELD_WORKSPACE/worker-env/bin/python" -m pip install \
  -e . -r requirements/core.txt
export SILICONSHIELD_SECURITY_PYTHON="$SILICONSHIELD_WORKSPACE/worker-env/bin/python"
export SILICONSHIELD_SECURITY_CHECKPOINT="$SILICONSHIELD_WORKSPACE/models/contract-1"
export LAYA_ROOT="$SILICONSHIELD_WORKSPACE/laya-next"
python -m siliconshield.app --case unauthorized-write \
  --output "$SILICONSHIELD_WORKSPACE/model-demo"
```

This last command uses a real model and the authored repair. It requires
Verilator, a C++ compiler, and make. Inspect `security_assessment`, including
coverage and actual status, as well as the executable checks. A missing model
assessment must not be presented as a successful model demonstration.

For both-model registration and measured routing, follow
[decision-model qualification](expert-qualification.md). A model without matching
capability measurements can run individually but must not inherit an unrelated
expert's routing authority.

### Verilator build failures

Use Verilator 5.024 or newer for the executable-repair workflow. Version 5.020
can abort intermittently with `attempted to destroy locked Thread Pool` during
compiler shutdown. The upstream fix is recorded in the
[5.024 release notes](https://github.com/verilator/verilator-announce/issues/69).
Check the installed version with `verilator --version`.

This is a tool execution failure, not an RTL verdict. SiliconShield records a
compile error and leaves the investigation inconclusive. Preserve the failed
build log and rerun into a new output directory with a fixed toolchain. A
successful retry alone does not establish that an older toolchain is reliable.

## Real Muse repair

The simplest endpoint path is:

```bash
python -m siliconshield.app --case unauthorized-write \
  --muse-endpoint "$SILICONSHIELD_MUSE_ENDPOINT" \
  --muse-model "$SILICONSHIELD_MUSE_MODEL" \
  --output "$SILICONSHIELD_WORKSPACE/live-demo"
```

Set those variables to your configured endpoint and its actual served model.
Credentials, if needed, belong in the configured client, not in the repository.
Remote endpoints require HTTPS. For an existing loopback service on a remote
machine, the supported Python client can use SSH without TCP forwarding:

```python
import os
from pathlib import Path

from siliconshield.agents.muse_proposer import make_muse_proposer
from siliconshield.app.demo_cases import CASES
from siliconshield.app.workflow import investigate
from siliconshield.research_loop.controller.client import ChatClient

client = ChatClient(
    os.environ["SILICONSHIELD_MUSE_ENDPOINT"],
    os.environ["SILICONSHIELD_MUSE_MODEL"],
    ssh_host=os.environ["SILICONSHIELD_MUSE_SSH_HOST"],
    max_calls=1,
    max_tokens=4096,
)
proposer = make_muse_proposer(client.endpoint, client.model, client=client)
investigate(
    CASES["unauthorized-write"],
    Path(os.environ["SILICONSHIELD_WORKSPACE"]) / "live-demo",
    proposer=proposer,
)
```

The assessor uses the environment variables above. The proposer receives the
original RTL and check evidence, with neither the authored patch nor expected
verdicts. The same requirement and regression checks run after generation.
The endpoint alias identifies the requested serving configuration. Capture its
base/adapter deployment identity separately when making stronger provenance
claims. One successful repair is an integration result, not a repair benchmark.

## Release terms

The owner selected **Apache-2.0 for Contract-1** on November 1, 2026, consistent
with Contract-2. The LAYA foundation card declares Apache-2.0 in the
[foundation model card](https://huggingface.co/convaiinnovations/laya).
The tested LAYA Next runtime includes its
[Apache-2.0 license](https://github.com/debjyotikm/laya-next/blob/main/LICENSE).
These links point to the current upstream pages. The inference runtime remains
pinned by the release manifest. This license check does not reconstruct an
unspecified historical foundation revision or relicense source datasets.
Retain upstream attribution and applicable source terms.

### Publication status checked on 3 November 2026

Both public model cards now declare Apache-2.0, resolving the previously
recorded Contract-1 metadata discrepancy. Neither model repository's file
listing contains a standalone LICENSE file at the checked revision.

| Model | Public model card | Declared license |
|---|---|---|
| Contract-1 | [Model card](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-1) | Apache-2.0 |
| Contract-2 | [Model card](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-2) | Apache-2.0 |

The Hub revisions checked on this date differ from the pinned inference
revisions in the release manifest. This metadata check did not re-download or qualify their current model payloads.
The release manifest retains the tested inference pins. Review payload identity
and terms before adopting a newer revision.

For a release handoff, retain the measured source revision, environment,
worker identity, and execution receipts. Live inference, fixture checks,
and model-quality evaluation remain separate qualifications.
