# Training the Glimmer controller

[Documentation index](README.md)

Muse Glimmer learns to read training evidence and make bounded operator
recommendations. The implementation supports completed-checkpoint review,
six-action operator exercises, multi-constraint checkpoint assessment,
supervised LoRA training, and offline DPO. The Muse coding-agent campaign
supervisor is a separate path. See [campaigns](campaigns.md).

Training prompts historically name the LAYA runtime. That wording is part of
the model input contract, not the product name. Do not silently rewrite a frozen
training prompt during an evaluation comparison.

## Training examples

Set `SILICONSHIELD_DATASET_KEY` before building data as described in
[setup](setup.md#glimmer-image-and-workspace). To convert eligible completed
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

## LoRA training

`training/glimmer/train.py` loads the local BF16 checkpoint and trains FP32 LoRA
parameters in the language model. It uses AdamW and computes loss on assistant
responses. Examples that exceed the token limit are rejected before training.

The trainer saves the adapter, optimizer, progress, and random-number state.
Resume checks the configuration and dataset hashes, restores the state, and
compares development loss before taking another optimizer step. The previous
checkpoint generation is retained for recovery after an interrupted save.

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

## Harness²

`research_loop/procedures/harness2_adapter.py` connects the local Glimmer endpoint
to Harness²'s Propose-Probe-Compose procedure. It proposes prompt edits, replays
tasks, and retains lessons only when the external grader accepts the analysis.
The selector chooses before receiving the external grade.

Harness² edits instructions. LoRA training changes model weights. Evaluate the
two separately so the source of any change is clear.

## Running the workflow

See [setup.md](setup.md) for commands. The Transformers reference endpoint
handles one request at a time and uses Muse's native ATEM tool-call format. The
separate native vLLM launcher has configurable concurrency. See
[inference](inference.md).

The campaign command can wait for a Glimmer training container, run base and
adapter evaluations, and run Harness² trials. It saves the results for review.
Deploying the updated controller remains a separate step. Submitting the next
SiliconShield-RTL experiment is handled by the job runner or the bounded
training operator.

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

## Supervised training configuration

The trainer uses language-model attention and MLP LoRA targets with the base and
vision components frozen. Rank defaults to 16, alpha is twice rank, and dropout
is 0.05. These are implementation defaults, not universally optimal settings.
Only assistant spans enter the supervision objective. Complete tool traces must
fit the registered token budget. LoRA parameters and optimizer state use FP32
with BF16 base computation.

Execution checks cover the expected code identity, resolved image identity,
base snapshot, adapter, tokenizer/template, data signatures, and resume state.
Keep the source snapshot fixed from preflight through execution. Setup instructions
for the identity variables and shared GPU lease are in [setup](setup.md).

Choose one task-selection mode: in-process replay with `--task-eval-every`, or
`--external-task-selection --task-eval-every 0` with separate serving/evaluation
windows. External mode preserves observations as checkpoints and leaves selection
pending. A lower token loss alone cannot complete that selection.

## Offline DPO

DPO starts from a reviewed operator adapter and a validated preference file.
The reference includes the same starting adapter. Reference log-probabilities
are computed once, allowing training to load one base plus a trainable adapter
instead of retaining a second live reference model.

After preparing the identities and lock described in setup:

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
