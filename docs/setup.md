# Setup

[Documentation index](README.md) · [Pinned releases](releases.md) ·
[Router](router.md) · [Study walkthrough](study-walkthrough.md)

This guide connects the three SiliconShield workflows to their prerequisites,
entry points, and output checks. Start with the shared installation, then choose
one path:

| Workflow | Start here | What runs |
|---|---|---|
| Investigate RTL with Muse Glimmer and specialist decision models | [Agent-assisted investigation](#1-investigate-rtl-with-muse-glimmer-and-specialists) | Local Glimmer requests scoped assessments and interprets specialist evidence. |
| Use the decision models directly | [Direct inference](#2-use-the-decision-models-directly) | Contract-1 or Contract-2 returns a structured prediction, without an agent. |
| Build or improve a decision model | [Training setup](#3-build-and-improve-decision-models) | A frozen study governs data, optimization, evaluation, and recoverable GPU jobs. |

Decision-model inference can use CPU or CUDA. Glimmer serving has its own GPU
runtime and can run on another machine. The shipped training job runner requires
Linux and systemd user services. If working from a Mac, run these commands in an
SSH session on the Linux host. A local browser or terminal does not need the
training dependencies.

## Shared installation

### Controller and workspace

Use Python 3.11 or 3.12 for the controller. Run subsequent commands from the
checkout root with this environment active:

```bash
git clone https://github.com/debjyotikm/siliconshield.git
cd siliconshield
python3.11 -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install -e .

export SILICONSHIELD_WORKSPACE="$HOME/siliconshield-workspace"
mkdir -p "$SILICONSHIELD_WORKSPACE"
python -m siliconshield.experts --help
```

The controller dispatches work and validates results. It does not need Torch
or load model weights into its own process. Keep models, registries, datasets,
traces, and run outputs in the workspace, outside Git. Record
`git rev-parse HEAD` with each measurement.

### Decision-model worker environment

Download both immutable exports and check out the compatible LAYA Next runtime
using [release setup](releases.md#download-immutable-exports). That procedure
reads the pinned payloads from [model-releases.toml](../configs/model-releases.toml)
and produces `models/contract-1`, `models/contract-2`, and `laya-next` beneath
`SILICONSHIELD_WORKSPACE`. Do not substitute a moving runtime branch.

Create the worker environment separately. The commands below use the pinned
runtime dependencies. For CUDA, select the platform wheel/index compatible with
your GPU and driver before installation. The tested versions and hardware are
recorded in [releases](releases.md#tested-combination).

```bash
python3.11 -m venv "$SILICONSHIELD_WORKSPACE/worker-env"
"$SILICONSHIELD_WORKSPACE/worker-env/bin/python" -m pip install \
  -e . -r requirements/core.txt

export SILICONSHIELD_SECURITY_PYTHON="$SILICONSHIELD_WORKSPACE/worker-env/bin/python"
export LAYA_ROOT="$SILICONSHIELD_WORKSPACE/laya-next"
export SILICONSHIELD_DEVICE=cpu

PYTHONPATH="$LAYA_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$SILICONSHIELD_SECURITY_PYTHON" -c \
  'import laya, torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

`cpu` is an explicit supported inference device. CUDA availability may be false
for that choice. To use a GPU, check `nvidia-smi` and set
`SILICONSHIELD_DEVICE=cuda:0` before creating the registry. Select an available
card. The worker does not silently fall back from CUDA to CPU.

The LAYA runtime must contain `laya/__init__.py`. Installing SiliconShield alone
does not install that runtime, decision-model exports, or a GPU driver. CPU and
CUDA inference have different execution identities and can differ on marginal
predictions. Capability measurements must match the chosen execution settings.

### Register both decision models

Create a new registry from the release manifest. This checks paths and schema
on CPU. It does not load weights or assign capability tiers:

```bash
python - <<'PY'
import json
import os
import tomllib
from pathlib import Path

from siliconshield.experts.registry import load_registry
from siliconshield.schemas import GUI_MODEL_REGISTRY_SCHEMA

workspace = Path(os.environ["SILICONSHIELD_WORKSPACE"]).resolve()
release = tomllib.loads(Path("configs/model-releases.toml").read_text())
models = []
for name, model in release["models"].items():
    models.append({
        "id": name,
        "display_name": name + " decision model",
        "checkpoint": str(workspace / "models" / name),
        "python": os.environ["SILICONSHIELD_SECURITY_PYTHON"],
        "laya_root": os.environ["LAYA_ROOT"],
        "device": os.environ["SILICONSHIELD_DEVICE"],
        "expected_files": model["files"],
        "release": model["revision"],
        "question_contract": release["runtime"]["question_contract"],
    })
path = workspace / "models-unprofiled.json"
with path.open("x") as stream:
    json.dump({"schema": GUI_MODEL_REGISTRY_SCHEMA, "models": models}, stream, indent=2)
print([entry["id"] for entry in load_registry(path)])
PY
```

Expected output is `['contract-1', 'contract-2']`. An existing registry is not
overwritten. Full export hashes and runtime identity are checked when a worker
loads. The schema constant retains its historical name for compatibility. It
has no desktop-interface dependency.

This registry is sufficient for direct Python inference. Routing additionally
needs measured capability profiles, prepared in the next section. Changing an
environment variable later does not change an existing registry's device.

### Prepare one complete assessment

See [Writing requirements for RTL analysis](rtl-requirements.md) for examples
of protection policies, accepted transfers, reset scope, and missing context.

Save the requirement, assumptions, and relevant RTL together. The section
headers are literal and have no trailing colons:

```bash
cat > "$SILICONSHIELD_WORKSPACE/assessment.txt" <<'ASSESSMENT'
REQUIREMENT
After reset, a push while full must leave occupancy unchanged.

ASSUMPTIONS
Two-state logic sampled at the rising clock edge. Reset is exempt.

RTL
module topmodule(input logic clk, rst_n, push, full,
                 output logic [3:0] occupancy);
  always_ff @(posedge clk) begin
    if (!rst_n) occupancy <= 4'd0;
    else if (push && !full) occupancy <= occupancy + 1'b1;
  end
endmodule
ASSESSMENT
```

Here `full` is a supplied input. The question concerns occupancy hold, not the
correctness of full detection or FIFO data ordering. For another requirement,
include all relevant writers, priorities, helper definitions, and signal widths.
A model prediction applies to the supplied region and assumptions. It is not a
proof of the surrounding design.

## 1. Investigate RTL with Muse Glimmer and specialists

### Measure routing support

Install the benchmark download dependency in the controller:

```bash
python -m pip install -r requirements/benchmark.txt
```

Follow [the reviewed-population procedure](benchmark.md#reproduce-the-reviewed-population)
to prepare the reviewed dataset and its provenance record. Then measure the
registered models and publish their profiles into a new output directory:

```bash
python examples/qualify_experts.py \
  --registry "$SILICONSHIELD_WORKSPACE/models-unprofiled.json" \
  --dataset "$SILICONSHIELD_WORKSPACE/reviewed-benchmark/benchmark-reviewed.jsonl" \
  --areas fifo,counter,arbitration,ot-access-control \
  --output "$SILICONSHIELD_WORKSPACE/routing-measurement"

export SILICONSHIELD_REGISTRY="$SILICONSHIELD_WORKSPACE/routing-measurement/models.local.json"
```

This runs model inference on the registry's device and creates ready receipts,
measurement reports, capability profiles, and the routable registry. It makes
no agent calls and does not train. The example refuses to publish profiles if
inference errors occur. See [decision-model qualification](expert-qualification.md) for
the output contract and additional scopes.

These are development measurements for the selected areas. They do not establish
independent model qualification or a combined-router accuracy advantage. Reuse
an existing profile only when its checkpoint, runtime, question contract, and
inference settings match the current worker. CPU inference needs CPU measurements.

### Connect local Glimmer

Use a running Glimmer service with the tool-call interface supported by the
[router bridge](router.md#driving-assessments-from-glimmer). For a new deployment,
follow [reference serving](inference.md#image-and-workspace) or the separately
qualified [native inference path](inference.md). Keep that environment separate
from the decision-model worker. The repository supplies a reference Dockerfile,
not a ready-built native vLLM image.

Set the actual endpoint and served model name. These values match the reference
server's defaults:

```bash
export SILICONSHIELD_MUSE_ENDPOINT=http://127.0.0.1:18320/v1/chat/completions
export SILICONSHIELD_MUSE_MODEL=glimmer-base
```

No Muse CLI login is required for this endpoint path. If Glimmer runs on another
machine, add `--ssh-host YOUR_SSH_ALIAS` to the session command below. The
endpoint is then resolved on that host, and the transport uses SSH without TCP
forwarding. Configure and verify that SSH connection first. Direct non-loopback
HTTP is refused. A directly reachable remote service must use HTTPS.

Create the task from the same assessment file:

```bash
cat > "$SILICONSHIELD_WORKSPACE/router-task.txt" <<'TASK'
Investigate the requirement below using the available specialists.
Read the roster, assess the relevant scope, and preserve the supplied RTL,
requirement, and assumptions. Explain the predictions, disagreements, and
missing context. Distinguish a model assessment from verified behavior.

TASK
cat "$SILICONSHIELD_WORKSPACE/assessment.txt" >> "$SILICONSHIELD_WORKSPACE/router-task.txt"

python -m siliconshield.research_loop.procedures.glimmer_router \
  --registry "$SILICONSHIELD_REGISTRY" \
  --endpoint "$SILICONSHIELD_MUSE_ENDPOINT" \
  --model "$SILICONSHIELD_MUSE_MODEL" \
  --skill .agents/skills/glimmer-router/SKILL.md \
  --task-file "$SILICONSHIELD_WORKSPACE/router-task.txt" \
  --output "$SILICONSHIELD_WORKSPACE/router-session" \
  --max-turns 6 --max-calls 6
```

**Check the result:** the session directory should contain a trace of the
roster request, assessment tool call, specialist responses, and final agent
message. Confirm that the assessed input matches the task. Inspect expert
errors, effective scope coverage, and disagreements rather than checking only
the process exit code. The final conversational message is not passed through
the structured arbitration validator.

The experts CLI's separate `--arbitrate` option uses the configured **Muse CLI**
and its login. It does not read the Glimmer endpoint variables above. See
[router assessments](router.md#run-one-assessment) for that structured path.
In either deployment, RTL and evidence go to the configured agent backend.
Local specialist execution alone does not make a remote agent call local.

## 2. Use the decision models directly

### Python integration without an agent

The worker API accepts an explicit model entry, so direct use needs no capability
profile, Muse endpoint, or agent login. This example loads Contract-2, assesses
the file above, prints its structured result, and closes its worker:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

from siliconshield.experts.registry import load_registry
from siliconshield.experts.worker import WorkerPool

workspace = Path(os.environ["SILICONSHIELD_WORKSPACE"])
entries = load_registry(workspace / "models-unprofiled.json")
entry = next(model for model in entries if model["id"] == "contract-2")
state = (workspace / "assessment.txt").read_text()
with WorkerPool() as workers:
    workers.load(entry)
    result = workers.analyze(entry["id"], state)
    print(json.dumps(result, indent=2))
PY
```

Select `contract-1` and supply a local guard or register-update requirement to
use the other decision model. In a Python application, retain the `WorkerPool`
context across requests to avoid repeated loading. `workers.ready(model_id)`
returns the validated load receipt.

**Check the result:** `label`, `probabilities`, `model_id`, `execution_identity`,
and `coverage` describe the prediction and the system that produced it.
Truncated inputs are refused without a verdict. A valid response establishes
successful inference, not that the prediction is correct. Scores are not
calibrated correctness probabilities.

### CLI without an agent

With the measured registry from workflow 1, run the same input through the
specialists without arbitration:

```bash
python -m siliconshield.experts assess \
  --registry "$SILICONSHIELD_REGISTRY" \
  --scope fifo \
  --state-file "$SILICONSHIELD_WORKSPACE/assessment.txt" \
  > "$SILICONSHIELD_WORKSPACE/assessment.json"
```

The CLI uses scope-based routing but makes no agent call unless `--arbitrate`
is supplied. Read `assessment.responses`, `assessment.errors`, and effective
`assessment.routing`. An unprofiled registry is valid for the Python example
but provides no routable experts for this command. The CLI reports specialist
scores as `scores`, while the direct worker result uses `probabilities`.

## 3. Build and improve decision models

### Training environment and study mode

Use a separate CUDA-capable training interpreter with SiliconShield,
[core dependencies](../requirements/core.txt), and a pinned compatible LAYA
runtime. It may use the same dependency versions as the inference worker, but
its source snapshot, runtime, devices, and settings are frozen for the study.
Check imports and systemd before preparing a job:

```bash
LAYA_ROOT="$LAYA_ROOT" "$SILICONSHIELD_SECURITY_PYTHON" \
  -m siliconshield.training.laya.security_train --help
systemctl --user show-environment >/dev/null
nvidia-smi
```

If using a separate training environment, set `SILICONSHIELD_SECURITY_PYTHON`
and `LAYA_ROOT` to it before these commands. Existing model registries retain
their own interpreter and runtime paths.

| Study mode | Starting artifact | Evaluation obligation |
|---|---|---|
| `improve_existing` | An immutable decision-model export, such as Contract-1 or Contract-2 | Improve the target while satisfying the registered retention limits. |
| `create_new` | An untouched, adapter-free foundation with a verified foundation reference | Establish the new domain's baseline, development gates, and reserved evaluation plan. |

Removing adapters from an already trained export does not make it an untouched
foundation. Full encoder training, adapters, or partial-block training are
explicit study choices, not automatic consequences of the mode.

### Qualify the inputs and bind the job

Complete these steps before launching:

1. **Accept the data.** Follow the [data pipeline](data-pipeline.md) for solver
   evidence, English-to-assertion review, lineage-disjoint splits, and immutable
   acceptance. Public training rows are reusable source material, not an
   automatic acceptance record. Keep benchmark/development rows out of TRAIN.
2. **Measure the recipe.** Run a bounded TRAIN-only fitting probe and resource
   check. Verify that the intended parameter groups move, the examples fit the
   input budget, and the sampler delivers required exposure. The
   [input validator](decision-model-training-with-glimmer.md#validate-the-data-on-cpu)
   checks the planned data and sampling configuration before GPU submission.
3. **Freeze the study.** Follow the [study walkthrough](study-walkthrough.md)
   to bind parent, data, source, runtime, recipe, views, selection policy,
   receipts, and the exact launch command. Its output is `study.json` and a
   `bound-job-profile.json`. The study API verifies them without launching.
4. **Assign resources.** The profile selects GPU IDs, run storage, timeout,
   retries, and a shared resource root. Use the same resource root for jobs
   competing on one host. Resolve a busy-device refusal through ownership
   checks rather than changing lock paths.

Keep all study artifacts outside Git. A source snapshot must contain the full
`siliconshield` package. Use the [trainer guide](security-model-training.md)
for optimization controls and [training operations](training-operations.md) for ownership and
recovery. Installing dependencies alone does not make a study train-ready.

### Launch with a local Glimmer operator

Prepare `operator.local.json` in the study workspace using the complete
[one-job configuration](decision-model-training-with-glimmer.md#run-one-frozen-study-job).
It names the bound profile, the frozen experiment ID, the actual Glimmer
endpoint/model, output directory, polling budget, and intervention budget.
Use the same endpoint preparation as workflow 1. On another serving host,
set `ssh_host` in this configuration.

The following command **submits or reattaches to a GPU training job**:

```bash
export SILICONSHIELD_STUDY=/path/to/reviewed-study
python -m siliconshield.research_loop.execution.glimmer_operator \
  --config "$SILICONSHIELD_STUDY/operator.local.json"
```

Glimmer reads evidence and proposes bounded actions. SiliconShield enforces the
study and owns execution, checkpoints, and recovery. Only the decision model
receives optimizer updates. Glimmer's weights remain unchanged.

**Check the result:** require completed execution, valid checkpoints, delivered
exposure, per-view development results, an explicit promote-or-preserve
assessment, and a fresh SDK re-score of the selected export. Calibration and
final test evaluation follow their separate reserved protocols.

Reattach with the same configuration after an interruption. Reaching
`max_cycles` ends that client invocation, not necessarily the owned job.
Inspect execution and operator receipts before another submission. The
[bounded strategy driver](decision-model-training-with-glimmer.md#fitting-followed-by-development)
can compare fitting recipes, but does not create a study manifest for each job.
Use the bound one-job path for manifest-enforced studies. Multi-round diagnosis
and supervision are configured separately in [campaigns](campaigns.md).

## Optional tools and development checks

Verilator, a C++ compiler, and make are needed for the separate
[executable repair workflow](releases.md#real-muse-repair), not ordinary specialist
assessment. Dataset construction may need Yosys, Icarus, and
`requirements/data.txt`. Harness² guidance is optional. Its dependency and
worker setup are described in [Muse procedures](muse-procedures.md).

For repository development, add the pinned tools to the controller and run:

```bash
python -m pip install -r requirements/style.txt
bash tools/check.sh
```

Focused CPU checks for these paths are:

```bash
python -m pytest -q tests/test_multi_expert.py tests/test_expert_safety.py \
  tests/test_glimmer_router.py tests/test_decision_study.py \
  tests/test_study_enforcement.py tests/test_operator.py
```

These fixtures use synthetic inputs. They do not start live agent services or
GPU jobs. Passing them does not establish model accuracy or autonomous operator
quality. See [contributor checks](../CONTRIBUTING.md#local-gate) for the curated gate's scope.

## Troubleshooting

| Symptom | Check |
|---|---|
| Worker cannot load | Verify its interpreter can import SiliconShield and the pinned LAYA runtime. Check export files, hashes, device, and memory. |
| Router finds no experts | Use the generated profiled registry and a scope present in its measurements. An unprofiled registry is insufficient. |
| Evidence becomes unrated | Compare the measured and observed execution identities. Remeasure after changing weights, runtime, question contract, device type, or precision. |
| Glimmer is reachable but makes no valid tool calls | Check the served model name, tool-call/reasoning parser, response budget, and recorded trace. Reachability alone is insufficient. |
| `--arbitrate` asks for Muse login | That flag uses the Muse CLI. Use the Glimmer router bridge for an explicit local endpoint. |
| Input is refused as truncated | Supply a smaller, semantically complete region. Do not silently clip the RTL or turn a transport refusal into a context label. |
| Study refuses launch | Inspect the reported dataset, recipe, command, receipt, or resource mismatch. Review and rebind a changed plan rather than bypassing the check. |
| Controller exits while training continues | Inspect the owned unit and receipts, then reattach with the same configuration. Do not duplicate the job. |
