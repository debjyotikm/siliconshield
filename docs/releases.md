# Pinned decision-model releases

[Documentation](README.md) · [Setup](setup.md) · [Demo](demo.md) · [Model card](siliconshield-rtl.md)

This is the reproducible inference path for the Contract-1 and Contract-2
RTL decision models. It is separate from reproducing their full training
histories. The source-controlled [release manifest](../configs/model-releases.toml) pins every inference payload
file and the compatible runtime. Downloads do not change a Hub repository.

## Tested combination

| Component | Pinned value |
|---|---|
| SiliconShield inference source | `23eea64b281088fe69919b3153ee37f73d4044e6` |
| LAYA Next | `55862a438a0f2e2ffec7a8db138bab537ac8a6d9` |
| Contract-1 Hub revision | `2f63a4a87d0d6010fca033b1637ff4142ff7cc60` |
| Contract-2 Hub revision | `b96211872d573ae48b6dabe128c0d96aad291156` |
| Worker Python | 3.11.15 |
| Torch / Transformers | 2.14.0+cu130 / 5.5.0 |
| Safetensors / NumPy | 0.8.0 / 2.4.6 |
| Inference | BF16, one row per worker request, sorting disabled, zero CPU fallbacks |
| Tested device | NVIDIA RTX PRO 4000 Blackwell, CUDA |
| Contract | `siliconshield-requirement-context-v1` |
| Label order | satisfies, violates, insufficient context |
| Budgets | 1,536 input tokens, 256 question/options tokens |

The release-preparation changes add the downloader and documentation on top of
that source commit. The model worker and inference contracts are unchanged.
Before publishing the final source revision, record its commit alongside this
manifest. The rehearsal used fresh public downloads and a fresh runtime checkout
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
git -C "$SILICONSHIELD_WORKSPACE/laya-next" checkout \
  55862a438a0f2e2ffec7a8db138bab537ac8a6d9
```

Each export contains exactly six inference files. The downloader verifies full
SHA-256 values, publishes each file only after checking its bytes, and refuses
existing mismatched or extra files. Retrying reuses verified files. It prints
these complete payload fingerprints:

| Model | Inference payload fingerprint |
|---|---|
| Contract-1 | `bd18becfd65f99b5e04f9150da460b158970d0fb5cf4c7a077c30635f6f6b9ef` |
| Contract-2 | `9c3080509bba9b2973f384a342fb801a7b6861d05f07a8c937235ac45658c5eb` |

These hashes match the worker's `execution_identity.export_sha256`: SHA-256
of the sorted file-hash mapping encoded as compact JSON, with separators
`(',', ':')`. The application checkpoint-manifest helper uses spaced JSON and
therefore has a different aggregate digest over the same files. It excludes Hub presentation files such as
README and benchmark summaries. It must not be compared to a historical export
that contains a different file set. The manifest also contains the full weights
hash and question-record hash for each model.

Use these commits, not an assumed `v1.0.0` tag. Contract-1's older public card
referenced that nonexistent tag. The prepared card correction uses the immutable
revision above.

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
[expert qualification](expert-qualification.md). A model without matching
capability measurements can run individually but must not inherit an unrelated
expert's routing authority.

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
with Contract-2. The LAYA foundation card declares Apache-2.0 at
[revision 7b928d8](https://huggingface.co/convaiinnovations/laya/blob/7b928d828b7b0e022f929d9bd2e44165aa270148/README.md).
The tested LAYA Next runtime includes its
[Apache-2.0 license](https://github.com/debjyotikm/laya-next/blob/55862a438a0f2e2ffec7a8db138bab537ac8a6d9/LICENSE).
This check records published terms at those revisions. It does not reconstruct
an unspecified historical foundation revision or relicense source datasets.
Retain upstream attribution and applicable source terms.

The pinned Contract-1 Hub card predates this decision and still says the license
is pending. Corrected card and LICENSE files are prepared outside Git for a
separate Hub publication. Until that publication, repository intent and the old
Hub metadata are not synchronized. Model payload hashes are unaffected by a
card-only update.

## Submission closeout

Before declaring the submission frozen:

1. Publish the approved Contract-1 license/card correction and verify its links.
2. Record the final source commit and retain the release manifest.
3. Publish the genuine CLI/report demonstration assets and add their URLs to
   the README. No placeholder media URL should be presented as a recording.
4. Run the documented commands under the selected environment and retain their
   receipts. Code gates, live inference, and model-quality results stay separate.

The CLI demonstration uses complete authored inputs and explicit backend
configuration. Assessments apply only to the supplied RTL and assumptions.
The router and repair application have separate invocation and output contracts.
