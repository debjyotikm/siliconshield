# Train decision models with local Muse Glimmer

[Documentation](README.md) · [Study walkthrough](study-walkthrough.md) ·
[Training operations](training-operations.md) · [Measured case study](case-study-decision-model-training-with-glimmer.md)

Muse Glimmer can operate an owned decision-model training job from a local
endpoint. SiliconShield prepares and launches the job, presents recorded
training evidence to Glimmer, checks proposed interventions, and finalizes the
result. The decision model receives optimizer updates. Glimmer supplies bounded
operator actions.

The case study used Glimmer to operate decision-model fitting and development
jobs, including a completed round with no qualifying improvement. It tested mostly
conservative actions. It did not qualify every intervention or demonstrate an
autonomous model-quality gain.

## Prepare the two environments

| Component | Required preparation |
|---|---|
| Decision-model worker | CUDA-capable training environment, compatible LAYA runtime, immutable parent export, accepted TRAIN/dev data |
| Glimmer endpoint | Locally served model with the tool-call interface used by the operator, reachable from the controller |
| Controller | SiliconShield package, systemd user services, writable private run and resource directories |
| Study | Frozen recipe, measured fitting/resource receipts, exact development views, selection rules and budget |

Use [setup](setup.md) for environment separation and [inference](inference.md)
for Glimmer serving. That guide distinguishes the reference Transformers server
from a separately qualified native inference build. Record the actual serving
model, runtime, precision and parser in the execution record. A healthy endpoint
alone does not establish operator quality.

Decision-model and Glimmer serving devices may be different machines on the
local network. The measured case study used an RTX PRO 4000 for training and DGX
Spark for Glimmer serving. This is a measured deployment, not a required topology.

## Validate the data on CPU

Start from an accepted dataset version. This command checks split loading,
question bytes, pools, reporting mass and planned sampling exposure. Use the
training interpreter because it imports the trainer, even with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' LAYA_ROOT="$LAYA_ROOT" \
  "$SILICONSHIELD_SECURITY_PYTHON" \
  -m siliconshield.training.laya.validate_study_inputs \
  --data /path/to/accepted-data --pools /path/to/pools.json \
  --recipe /path/to/recipe.json \
  --question-contract siliconshield-requirement-context-v1 \
  --seed 7 --global-batch 48 --world 1 --max-updates 300 \
  --output-receipt /path/to/input-validation.json
```

Replace the illustrative schedule with the registered study settings. Inspect
both the result and the coverage horizon. A successful input parse does not
mean the update budget delivers full exposure. Require the reported horizon to
fit the budget before launch. An optional token-length screen flags definite
oversize states but does not certify the full assembled model input.

## Run one frozen study job

The [study walkthrough](study-walkthrough.md#freeze-and-bind-the-study) constructs
the manifest and binds its digest into the job profile. The profile owns the
worker interpreter, source snapshot, runtime, parent, data, device list, recipe,
resource root and run directory. Use one shared resource root for competing jobs.
Do not move a live run directory or edit a frozen source snapshot.

Create a local operator configuration outside Git:

```json
{
  "profile": "/path/to/bound-job-profile.json",
  "experiment_id": "local-study-round-01",
  "endpoint": "http://127.0.0.1:18320/v1/chat/completions",
  "model": "glimmer-base",
  "output": "/path/to/operator-receipts/round-01",
  "max_cycles": 8,
  "poll_seconds": 60,
  "overrides": {},
  "intervention_budget": {}
}
```

Use the endpoint's actual model name. For an adapter, also bind `adapter_path`
and `checkpoint_fingerprint` in the same configuration object.
For a local-network SSH transport, set the supported `ssh_host` field rather
than assuming TCP forwarding is available. Endpoint and transport details stay
outside Git.

An empty intervention budget disables learning-rate adjustments. It does not
disable every operator action. `stop`, `resume`, and `finalize_retry` retain
their state and identity checks. New jobs have single-use experiment IDs.

**Launch:** this command calls Glimmer and can submit GPU work:

```bash
python -m siliconshield.research_loop.execution.glimmer_operator \
  --config /path/to/operator.local.json
```

The operator submits or reattaches to the registered job, observes its state,
validates Glimmer's actions and records the outcome. Reaching `max_cycles` ends
that client invocation, not necessarily the owned training job. Reattach with
the same configuration. Inspect `run-manifest.json`, `execution.json` and the
operator receipt before scheduling anything else.

## Fitting followed by development

The existing `strategy` driver automates a bounded recipe comparison: run
TRAIN-only fitting arms, qualify a recipe by fitting gain, then run one
development job from the profile's parent. It carries the winning **settings**
forward, not the fitting checkpoint. This is the path exercised in the case
study. Its local configuration has this complete outer shape:

```json
{
  "output": "/path/to/strategy-receipts",
  "experiment_prefix": "local-fit-study",
  "profile": "/path/to/job-profile.json",
  "timeout_seconds": 7200,
  "min_fit_gain": 0.01,
  "operator": {
    "endpoint": "http://127.0.0.1:18320/v1/chat/completions",
    "model": "glimmer-base",
    "max_cycles": 8,
    "poll_seconds": 60,
    "intervention_budget": {}
  },
  "fitting_common": {
    "limit_train_rows": 144,
    "no_development_in_fitting": true,
    "max_updates": 24,
    "min_updates": 24,
    "warmup": 2,
    "hold_updates": 0,
    "patience": 24,
    "eval_every_updates": 6,
    "lr_schedule": "constant",
    "lr_endpoint": 1.0
  },
  "arms": [
    {"name": "adapter", "overrides": {"freeze_encoder": true}}
  ],
  "development_overrides": {
    "max_updates": 300,
    "min_updates": 150,
    "warmup": 20,
    "hold_updates": 0,
    "patience": 6,
    "eval_every_updates": 10,
    "lr_schedule": "cosine",
    "lr_endpoint": 0.1,
    "development_views": "/path/to/development-views.json"
  }
}
```

These numbers demonstrate the interface, not a qualified recipe. The profile
must supply the accepted dataset, compatible sampler/pools and measured learning
rates. The fitting subset needs its own valid pool coverage when using pools.
Keep development views out of the profile's shared defaults and fitting arms.
Restate the full development schedule to prevent a short fitting schedule from
being inherited accidentally.

A single decision-study binding freezes one exact command and effective
settings. It cannot cover all the differently configured jobs above. The
strategy driver does not manufacture per-job study manifests. For new work
requiring manifest enforcement, qualify fitting separately, freeze the chosen
development job, and use the one-job operator path above. Do not remove a study
binding just to make a multi-job strategy configuration pass.

After reviewing that distinction, the bounded strategy entry point is:

```bash
python -m siliconshield.research_loop.execution.strategy \
  --config /path/to/strategy.local.json
```

## Read the closeout before another round

| Artifact or field | What it establishes |
|---|---|
| `execution.json` and owned-unit state | Execution completed, failed, or still needs reconciliation |
| Fitting report, `evaluation_role: train_fitting` | The recipe's measured TRAIN gain, not generalization |
| Strategy status receipt in the configured output directory | `no_qualified_recipe`, development progress, or bounded strategy completion |
| Development report and selection record | Which checkpoint was selected and which frozen gates passed |
| Fresh SDK re-score | Export behavior under the recorded inference protocol |
| Decision-study assessment | Whether the candidate qualifies for research promotion |

Selection may retain the parent. A null result is a completed experiment when
its exposure, outputs and failed gates are recorded. Resume, finalization retry,
and deliberate stopping are covered in [training operations](training-operations.md). Multi-round
Muse diagnosis belongs to the [campaign supervisor](campaigns.md), which is a
separate integration from this bounded strategy driver.
