# Training the Glimmer controller

[Documentation index](README.md)

Muse Glimmer learns to read training evidence and make bounded operator
recommendations. The implementation supports completed-checkpoint review,
six-action operator exercises, multi-constraint checkpoint assessment,
supervised LoRA training, and offline DPO. The Muse coding-agent campaign
supervisor is a separate path. See [campaigns](campaigns.md).

Local Glimmer currently has an implemented router-driver role: it can inspect
the roster and request specialist assessments through the router skill.
This guide records existing research infrastructure for training and
evaluating that backend. Adoption for other agent roles requires matched
task qualification, including per-action behavior, evidence use, and
recovery. An operator adapter is not automatically an RTL analyst or
router arbitrator.

Contract-1 and Contract-2 are RTL decision models. Glimmer is trained to read
operational evidence and choose bounded actions. A successful controller
adapter does not change either decision model until a separately authorized
job trains it.

Training prompts historically name the LAYA runtime. That wording is part of
the model input contract, not the product name. Do not silently rewrite a frozen
training prompt during an evaluation comparison.

## Router assessments

Glimmer can request specialist assessments through `router.roster` and
`router.assess`. The [router-session guide](router.md#driving-assessments-from-glimmer)
owns the command, tool-evidence contract, and current qualification status.
That role is separate from the operator training described here.

## Image and workspace

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
the key to bypass a signature failure. Use the [training-example workflow](#training-examples) below. Report conversion
alone does not guarantee the operator curriculum's required action coverage.

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

## Training examples

Set `SILICONSHIELD_DATASET_KEY` before building data as described in
[workspace preparation](#image-and-workspace). To convert eligible completed
reports, choose a fresh output directory:

```bash
python -m siliconshield.research_loop.episodes.dataset \
  --runs /path/to/completed-development-runs \
  --output /path/to/controller-data
```

The builder emits TRAIN, development, and test JSONL partitions with a manifest.
It checks eligibility and groups related receipts. A broader curriculum still
needs examples of operator decisions and independently reviewed targets.

`research_loop/episodes/dataset.py` reads completed development reports and
produces tool-use conversations. Each conversation follows three steps:

1. `research.read_run` reads the report.
2. `research.compare_metrics` calculates agreement and negative log-likelihood
   (NLL) changes.
3. `research.record_decision` records an action, both deltas, and an explanation.

A fixed policy supplies the training targets. For example, when agreement falls
and NLL rises, the target recommends investigating a regression. Related runs
stay in the same dataset split.

These examples teach report review and arithmetic. Training a controller to
design experiments or repair code will require additional tasks and evaluation.

## Build the right supervision

The episode package contains report conversion, operator scenarios, a curriculum
builder, independent development cases, transfer-evidence cases, and replay
baselines. Consult [the command reference](reference.md) for the separate entry
points. Authored scenario generation tests known situations. Paraphrasing one
situation does not produce independent operational experience.

Keep only information available at the decision point in model-visible messages.
Store later intervention outcomes separately. Include healthy no-op, execution
failure, quality regression, failed recovery, and incomplete-evidence cases.
Preserve lineage across arms, seeds, counterfactuals, and related schedules when
assigning splits. Remove irrelevant candidate/control identifiers that reveal
an authored target without requiring evidence analysis.

`--purpose operator` requires the trainer's action/lineage coverage checks and
task-based selection. Inspect the emitted coverage report before training. The
coverage report shows which situations the dataset contains. Replay evaluation
measures how well the model handles them. Keep final-test episodes out of SFT,
preference pairs, intervention planning, and Harness² retention.

## LoRA training

`training/glimmer/train.py` uses AdamW to train FP32 LoRA parameters on a frozen
BF16 language-model base. Attention and MLP layers are the LoRA targets. Vision
components remain frozen. Rank defaults to 16, alpha to twice the rank, and
dropout to 0.05. These are implementation defaults, not a qualified recipe.

Only assistant spans enter the supervision objective. Complete tool traces
must fit the registered token budget. Overlength examples are rejected before
training. Trainable parameters and optimizer state use FP32.

Execution checks bind the code, resolved image, base snapshot, adapter,
tokenizer/template, signed data, and resume state. Keep the source snapshot
fixed from preflight through execution. Use the identity variables and shared
GPU lease from [workspace preparation](#image-and-workspace).

The trainer saves the adapter, optimizer, progress, and random-number state.
Resume verifies configuration and dataset hashes, restores state, and compares
development loss before another update. It retains the previous checkpoint
generation for recovery after an interrupted save. Choose the task-selection
mode described in [evaluation](#evaluation) before launch.

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

Release the training GPU lease before starting [serving](inference.md) or
evaluation on the same device. Continue with the replay-selection and DPO
procedures below once execution and recovery are qualified.

## Evaluation

`research_loop/controller/evaluate.py` runs the same development tasks with the
base model and the adapter. It reports whether each run finished and whether
the decision was supported by the calculated metric changes.

The grader checks the action and structured deltas separately from the prose.
Explanation checks are reported on their own. Full tool calls and responses
are saved locally for inspection.

Final responses use a small JSON report that must match the recorded action,
metric deltas, and evidence scope. The model supplies its explanation in the
tool call. Unchecked free-form final claims do not pass verification.

Use `--task-eval-every` when training to select checkpoints by verified
development replay, then action/evidence correctness, with token loss as a
tie-breaker. A value of zero retains the older loss-only selection mode.

For a separate serving-based evaluator, use `--external-task-selection` with
`--task-eval-every 0`. The trainer preserves an adapter at each development
evaluation and marks task selection as pending. An external comparison must
select the checkpoint. Lower token loss cannot select it automatically. Training
and serving use separate GPU windows.

Operator observations can include checkpoint candidates and explicit limits
for each development population. Their structured assessment identifies
which candidates meet every limit, which fail, and which lack evidence.
This makes checkpoint review testable without granting job-control authority.

Operator qualification episodes use separate inspection and decision tools.
Replay cannot submit jobs. Authored exercises are interface tests, not verified
outcomes from live training interventions. Repeated copies do not add
independent situations. The job operator has a separate execution path with
owned systemd units, a process-lifetime GPU lock, and checkpoint-boundary
control requests.

Development replay measures this review task. It does not measure general RTL
security skill or the quality of a proposed research program.

## Selecting an operator adapter

Use `siliconshield.training.glimmer.selection.select_replay_candidate` for
matched development replay reports. It checks the episode population, evaluator
and decoding identities, recomputes totals, and requires coverage of all six
operator actions. Selection must preserve action correctness, evidence,
explanation checks, final consistency, and verified outcomes for each action,
as well as for checkpoint-review and other tasks.

The selector accepts multiple references, such as the serving adapter and a
newer research adapter. A candidate must preserve both references' strengths and
improve the best aggregate verified count. A gain on finalization cannot hide
worse stop decisions. Equal results do not qualify as an improvement. Reports
involving final-test evaluation are ineligible for adapter selection. The result
identifies the selected adapter. Deployment and job authorization remain
separate steps.

## Harness²

`research_loop/procedures/harness2_adapter.py` connects the local Glimmer endpoint
to Harness²'s Propose-Probe-Compose procedure. It proposes prompt edits, replays
tasks, and retains lessons only when the external grader accepts the analysis.
The selector chooses before receiving the external grade.

Harness² edits instructions. LoRA training changes model weights. Evaluate the
two separately so the source of any change is clear.

## Running the workflow

The Transformers reference endpoint
handles one request at a time and uses Muse's native ATEM tool-call format. The
separate native vLLM launcher has configurable concurrency. See
[inference](inference.md).

The campaign command can wait for a Glimmer training container, run base and
adapter evaluations, and run Harness² trials. It saves the results for review.
Deploying the updated controller remains a separate step. Submitting the next
SiliconShield-RTL experiment is handled by the job runner or the bounded
training operator.

## Offline DPO

DPO starts from a reviewed operator adapter and a validated preference file.
The reference includes the same starting adapter. Reference log-probabilities
are computed once, allowing training to load one base plus a trainable adapter
instead of retaining a second live reference model.

After [preparing the workspace](#image-and-workspace), run:

```bash
python -m siliconshield.training.glimmer.dpo --mode reference \
  --model /path/to/base --init-adapter /path/to/starting-adapter \
  --pairs /path/to/reviewed-pairs.jsonl --output /path/to/reference

python -m siliconshield.training.glimmer.dpo --mode train \
  --model /path/to/base --init-adapter /path/to/starting-adapter \
  --pairs /path/to/reviewed-pairs.jsonl \
  --reference-logprobs /path/to/reference/reference_logprobs.json \
  --output /path/to/preference-run --beta 0.1 --lr 5e-7
```

The numbers illustrate the CLI, not a validated training recipe. Pair validation
excludes regression-only examples from optimization. Reference bindings include
pair content, tokenizer, snapshot, adapter, base revision, encoding budget,
implementation, and image identity. The trainer rejects missing or incompatible
resume state. Preserve the starting adapter unchanged.

Preference loss is not the acceptance metric. Run matched development replay
against the starting adapter and other required references, checking every
action class and final-response consistency. A corrected explanation authored by
an experimenter must retain that provenance. It is not evidence that the model
independently discovered the correction.

## Qualification and deployment

Use [evaluation](evaluation.md) for component metrics and split boundaries.
Freeze a candidate and its serving parser/configuration together before final
qualification. Review the selected adapter before deployment. Training does not
replace the active service or change its permitted actions.
