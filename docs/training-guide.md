# Training Muse Glimmer for RTL requirement decisions

Implementation record | 4 November 2026

[Documentation index](README.md)

This document records how the frozen SiliconShield Muse Glimmer decision adapter was built. It describes the executed experiment, including unsuccessful attempts and verification failures. It is not a claim that an upstream cookbook configuration was run unchanged, or that the reusable trainer below has reproduced the released experiment on a fresh machine.

For measured capabilities and scientific interpretation, see the [research white paper](whitepaper.md).

## 1. Official references and the method we used

Meta describes supervised fine-tuning as the appropriate starting point when target responses are available. Its post-training documentation covers adapter and full-parameter methods. We chose supervised fine-tuning because each RTL example already had a reviewed target decision. We did not use reinforcement learning, preference optimization, or a reward model. See [Muse Glimmer post-training](https://dev.meta.ai/docs/muse-glimmer/post-training).

The [official post-training cookbook](https://github.com/meta-models/meta-oss-cookbook/tree/main/muse-glimmer/post-training) supplies framework-specific examples. Its [Axolotl SFT recipe](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/post-training/sft/axolotl/README.md) distinguishes quantized adapter training, BF16 adapter training, and full-parameter training. Our successful run used a custom PyTorch loop with Transformers, PEFT, and Accelerate device hooks. We did not execute the Axolotl or FSDP2 recipe.

| Aspect | Executed SiliconShield method |
|---|---|
| Foundation | Original Muse Glimmer 30B |
| Task | English requirement and assumptions plus RTL to one decision token |
| Training | Supervised next-token cross-entropy |
| Adaptation | LoRA on language attention and MLP projections |
| Frozen parameters | Base model and original vocabulary head |
| Hardware | Three NVIDIA RTX PRO 4000 Blackwell GPUs, approximately 24 GiB each |
| Parallelism | One model partitioned across three GPUs |
| Data | Reused, accepted RTL decisions from two earlier-model training populations |
| Selected artifact | Owner-frozen optimizer update 199 |

The official recipe and this run have different hardware, runtime versions, data, losses, and parallelism. Upstream measurements should not be used as throughput or memory estimates for this run.

## 2. What we trained the model to do

Each model input contains three sections:

```text
REQUIREMENT
An English statement of the behavior being assessed.
ASSUMPTIONS
The clock, reset, input, and omitted-context interpretation.
RTL
The implementation region to assess.
```

The target is one of three labels:

| Answer | Label | Intended meaning |
|---|---|---|
| A | `satisfies_requirement` | The stated requirement holds for allowed behavior under the assumptions. |
| B | `violates_requirement` | An allowed counterexample contradicts the requirement. |
| C | `insufficient_context` | Omitted information can change the correct answer. |

The model learns these classifications. It does not prove them during inference. We did not train a separate vulnerability-localization head, CWE classifier, repair generator, or explanation objective.

A useful local example is a clocked register whose only non-reset write is guarded by `wr_en && privileged`. For the requirement “Outside reset, the register may change only when privileged is asserted,” the complete guarded design satisfies. Removing the privilege conjunct creates a violating design. Replacing the authorization expression with omitted logic supports an insufficient-context label only when compatible completions produce opposing outcomes.

For a counter, the requirement “Outside reset, simultaneous push and pop must leave count unchanged” distinguishes a hold branch from an increment branch. Reset exemptions and edge semantics belong in the assumptions. A correct answer to this one requirement does not establish overflow protection or full FIFO correctness.

## 3. How we assembled the dataset

### 3.1 Reuse rather than new generation

We did not generate a new broad RTL corpus for this run. We selected existing accepted examples from the final local-guard specialization dataset and the focused FIFO, counter, and arbitration dataset previously used with other open-source models.

The local-guard training subset was the explicit anchor-row list in its source pool manifest. We filtered the source TRAIN file by that list. Its development subset was the source counterbalanced family population. The temporal source supplied its focused TRAIN and DEV files directly, including nine verified sibling examples in TRAIN.

| Domain | Available TRAIN | TRAIN triplets | DEV | DEV triplets |
|---|---:|---:|---:|---:|
| Local-guard specialization | 576 | 192 | 288 | 96 |
| Focused temporal contracts | 2,700 | 900 | 234 | 78 |
| Total | 3,276 | 1,092 | 522 | 174 |

Each split is balanced across the three labels. The final specialization subsets do not reproduce the complete historical training curriculum of the earlier models. We did not use the entire public research training collection.

### 3.2 Semantic construction inherited from the source datasets

The source data uses linked examples with accepted evidence for their labels. Local-guard counterbalancing can keep the same RTL while changing which state element the requirement addresses. One write path can be protected, another unprotected, and a third dependent on an omitted expression. The requirement selects the relevant path.

Temporal examples connect satisfying behavior, a violating variant, and an omission whose resolution can change the answer. The required relationship is semantic. An unfamiliar signal name or a low-confidence prediction is not sufficient to label an example insufficient context.

Accepted proof records and earlier semantic review were inherited from these datasets. This focused preparation checked the accepted-record fields. It did not independently rerun all solvers or repeat the entire historical semantic review. That distinction matters when describing what was verified during this run.

### 3.3 Preparation checks actually executed

The preparation script checked:

1. Expected source counts and the registered TRAIN/DEV assignment.
2. The accepted validation flag on every selected row.
3. Exactly three members per triplet, with one of each label.
4. Unique row identifiers within each output split.
5. Zero TRAIN/DEV intersection in row identifiers, exact input-state strings, and registered split-group identifiers.
6. Encoded input length below the configured rejection limit.

A bounded scan also checked for label-revealing comments. That scan is not a proof that every semantic shortcut is absent. Exact split checks do not establish independence from Glimmer's original pretraining data.

The source row records retained fields such as `id`, `pair_id`, `split_group_id`, `expected.security`, and `validation.accepted`. Those fields support auditing and evaluation. Only the `state` text was inserted into the model's user message. Expected labels and provenance metadata were not included in that text by the converter.

### 3.4 Output of preparation

The converter wrote four source-preserving JSONL files, encoded TRAIN and DEV files, and a dataset receipt. Each encoded record retained:

```text
id, domain, pair_id, group, label, input_ids, state
```

`label` is the training target, stored separately from `input_ids`. The receipt records population counts, label balance, token lengths, split checks, prompt configuration, answer-token mapping, and monitor membership. Exact content identities remain in private evidence, not in this guide.

TRAIN contains 1,874,886 input tokens. DEV contains 276,892. The longest training input is 766 tokens. The configured limit was 2,048 tokens. Inputs exceeding the limit were rejected rather than truncated.

## 4. Native prompt and answer-token construction

The system instruction defined A, B, and C, required assessment using only supplied RTL and assumptions, and distinguished missing information from a conditional path. It explicitly discouraged deriving behavior from names or comments.

The fixed user prefix was:

```text
Assess this RTL requirement. Respect the reset scope and execution assumptions.
Answer with A, B, or C only.
```

The converter used the base tokenizer's native chat template with system, user, and assistant roles. It rendered an assistant message containing a sentinel, then cut the rendered text immediately before that sentinel. This produced the native final-answer prefix without appending the correct answer to the input.

Template settings fixed the date field and low reasoning-strength setting. The date value was a template input, not a time-dependent label feature. The three answer letters each encoded as one token. Their token identifiers were checked and stored in the receipt rather than assumed to be universal across tokenizers.

No reasoning trace or end-of-turn token was supervised. This is a single-answer-token decision adaptation, not ordinary long-response instruction tuning.

## 5. Runtime and GPU placement

The successful training runtime used:

| Component | Recorded version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.14.0 with CUDA 13.0 |
| Transformers | 5.21.0 |
| PEFT | 0.21.2 |
| Accelerate | 1.15.0 |

A private environment reused the workstation's installed Torch through an environment overlay. It was not a clean-room replay of an independently published lockfile. The initial Python 3.12 and newer-Torch attempt was retained as failed evidence. Numerical failure also occurred outside that version combination, so we did not identify the newer Torch version as the root cause.

The successful configuration loaded the frozen language model in BF16. It placed embeddings and decoder layers 0–19 on GPU 0, layers 20–39 on GPU 1, and layers 40–51 plus the final normalization and vocabulary head on GPU 2. Unused vision components remained on CPU. The text task never supplied images.

This was model partitioning in one training process. It was not three data-parallel replicas. DGX Spark did not perform optimizer updates in this experiment.

The base weights were staged on SSD after slow loading from rotational storage. Observed load time fell from roughly five minutes forty seconds to eighteen to twenty-two seconds. Full integrity checks matched the upstream model files. The SSD move changed loading performance, not the dataset or training objective.

## 6. LoRA and the training update

The adapter targets language-layer attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`, together with MLP `gate_proj`, `up_proj`, and `down_proj` modules.

| Setting | Value |
|---|---|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0 |
| Trainable parameters | 47,923,200 |
| Optimizer | AdamW |
| Peak learning rate | 0.0001 |
| Weight decay | 0.01 |
| Gradient clipping | Global norm 1 |
| Effective batch | Six rows |
| Warmup | Ten optimizer updates |
| Decay | Wall-time cosine schedule toward 20% of peak |
| Activation checkpointing | Enabled, non-reentrant |

Only adapter parameters received optimizer updates. The base vocabulary head supplied the logits and remained frozen.

For each row, the model produced the logits at the final prompt position. Training used FP32 cross-entropy over the complete vocabulary with the correct answer token as target. The loss did not normalize over only A/B/C during training.

The implemented update was equivalent to:

```python
optimizer.zero_grad(set_to_none=True)
for microbatch in effective_batch:
    logits = model(**inputs(microbatch)).logits[:, -1, :].float()
    targets = answer_token_ids[labels(microbatch)]
    loss = cross_entropy(logits, targets)
    (loss * len(microbatch) / len(effective_batch)).backward()
clip_global_gradients_on_cpu(parameters, max_norm=1.0)
optimizer.step()
```

This snippet describes the recorded implementation. It omits the checkpoint, device-hook, finite-value, and recovery code and is not a standalone runnable trainer.

## 7. Sampling, exposure, and microbatches

Triplets were shuffled without replacement within each domain using seed 44. The sampler interleaved one local-guard triplet and four temporal triplets while both pools remained. Six-row optimizer batches contained two complete triplets.

The first successful segment used one-row microbatches. Training stopped safely at update 39 for a resume with three-row microbatches. Effective batch size remained six. Batched inputs were left-padded with explicit attention masks and position identifiers. An early configuration note describing all inputs as unpadded was superseded by this batched implementation.

The frozen update-199 checkpoint had received exactly:

| Domain | Distinct rows delivered | Available rows |
|---|---:|---:|
| Local guards | 240 | 576 |
| Focused temporal | 954 | 2,700 |
| Total | 1,194 | 3,276 |

There were no repeated draws within that delivered first pass. The candidate therefore represents approximately 36.4% of the available TRAIN population, not a completed epoch.

The learning-rate schedule used elapsed wall time. Resuming after an interruption could therefore use a different learning rate than an uninterrupted run at the same update number. This execution should not be described as a controlled microbatch experiment or exact trajectory replay.

## 8. Failures and corrections

### Numerical corruption before valid learning

The initial quantized attempt and early BF16 probes produced nonfinite outputs. Those attempts did not supply valid optimizer updates. Subsequent direct-copy probes showed corrupt transfers from lower-numbered CUDA devices to higher-numbered devices in this environment. Tested reverse directions and same-device transfers behaved correctly.

We did not isolate the lower-level driver or interconnect cause. The finding does not prove physical hardware failure and does not invalidate unrelated historical jobs automatically.

### Host-staged transfers

The private transfer helper patched Accelerate's tensor-transfer hooks. A tensor moving between distinct CUDA devices was copied to CPU and then to the destination GPU. Autograd traversed the corresponding host-staged path during backward computation.

The gradient-clipping helper also avoided cross-device reduction. It copied each parameter's gradient norm to CPU, computed the global norm there, and applied a Python scalar clipping coefficient to each gradient on its own device.

Forward and backward probes passed across all nine source/destination combinations. Full-model qualification then checked finite logits, finite gradients, and a nonzero adapter parameter update. This workaround belongs to the recorded execution contract. It is not presented as a universal requirement for Glimmer training.

### Memory failure after the first evaluated checkpoint

The three-row microbatch passed the implemented longest-input probe, including backward computation, but the later run still encountered a CUDA memory allocation failure after update 217. Passing a bounded memory probe did not guarantee adequate headroom throughout training.

The latest saved checkpoint was update 199. We resumed its adapter and optimizer state and reduced the microbatch to one. The unsaved intervening updates were lost. The resume verified exact adapter tensor equality. Subsequent training stopped at the owner's request and completed its closeout at update 217.

### Frozen candidate versus terminal checkpoint

The owner requested freezing the model reported at update 199. We copied that checkpoint into a separate candidate directory and bound verification to it. The terminal update later ranked slightly better on the small monitor, but it did not replace the owner-frozen candidate. All final candidate metrics in the companion paper refer to update 199.

## 9. Checkpoint contents and recovery

Each saved checkpoint included:

- The adapter weights in safetensors format and PEFT adapter configuration.
- Optimizer state, optimizer update number, and per-row exposure counters.
- Python, CPU Torch, and CUDA random-number states.
- The training configuration and decision contract.

The decision contract records the prompt template, label order, answer tokens, base identity, runtime information, and relevant training settings. The adapter is approximately 183 MiB. It is not a standalone 30B model export and still requires the matching original model and tokenizer.

Checkpoint writes used a temporary directory followed by a rename. Latest and selected pointers were separate records. Recovery loaded weights, verified exact tensor equality, restored optimizer and random states, and used exposure counters to avoid restarting the entire first pass.

## 10. Evaluation and verification actually performed

A seeded monitor selected 24 complete triplets per domain before inference. Its 144 decisions were used for selection. Ranking prioritized worst-domain false-safe rate, worst-domain triplet accuracy, weakest class recall, mean domain accuracy, and conditional negative log likelihood.

After freezing, a fresh process evaluated all 522 development decisions and six authored demonstrations. At inference, the A/B/C logits were normalized over those three tokens. The combined full-vocabulary probability mass was also recorded. Decision scores are not calibrated probabilities of engineering correctness.

| Frozen model population | Correct | False-safes | Whole triplets |
|---|---:|---:|---:|
| Local-guard full DEV | 280/288 | 0/96 violations | 88/96 |
| Focused temporal full DEV | 223/234 | 5/78 violations | 67/78 |

All six authored demonstrations were correct. They covered protected and unprotected register writes, correct and incorrect simultaneous push/pop behavior, and omitted helpers. They are demonstrations, not independent qualification.

Fresh loading needed two implementation corrections. A missing `psutil` dependency was installed in the private environment. PEFT's convenience loader attempted automatic redispatch, so the verifier instead constructed the adapter on the already qualified placement and loaded its saved state explicitly.

The saved adapter tensors matched exactly. All 144 monitor decisions matched the fresh process. However, 22 probability vectors exceeded the registered absolute tolerance of 0.00001, with a maximum difference of approximately 0.03083. Exact probability parity failed and its cause remains unresolved. The record supports decision agreement, not numerically identical inference.

No sealed final test was opened. Neither development results nor successful checkpoint loading constitute automatic production qualification.

## 11. Where the implementation evidence lives

The experiment retained the following private artifacts outside the source repository:

| Artifact | Purpose |
|---|---|
| `prepare_data.py` | Select source subsets, validate structure and splits, render native inputs |
| `data/DATA_RECEIPT.json` | Counts, token limits, split checks, prompt settings, monitor IDs |
| `train_glimmer_rtl.py` | Initial successful host-staged training implementation |
| `train_glimmer_rtl_v2.py` | Resume with larger microbatches |
| `train_glimmer_rtl_v3.py` | Smaller-microbatch recovery after memory failure |
| `safe_gpu_transfer.py` | Host-staged cross-device transfer and CPU norm clipping |
| `loading.py` | CPU staging during model loading |
| `run/events.jsonl` | Update, evaluation, checkpoint, and failure events |
| `FROZEN_CANDIDATE.json` | Explicit update-199 freeze record |
| `frozen-update-199/` | Preserved adapter, optimizer state, and decision contract |
| `verify_frozen.py` | Full-development fresh-process scoring and strict parity check |
| `verify_frozen_finish.py` | Retain measured parity failure and complete authored demonstrations |
| `verification/VERIFICATION.json` | Full results, parity status, and demonstration outputs |

These scripts include experiment-specific locations and time limits. They must be reviewed and parameterized before reuse for a new study. Running the original preparation script against the completed directory could replace configuration files, so a new study should use a new output directory. This document does not claim a tested one-command installation or portable end-to-end training package.

## 12. What to carry into a new training run

The established procedure is to define the decision semantics, preserve the source split boundaries, validate complete triplets, render the native prompt, verify the output-token mapping, qualify real optimizer updates, and test saved-artifact inference. Keep measured exposure separate from requested epochs. Preserve failure receipts and report per-class behavior alongside total accuracy.

For this particular implementation, any reproduction also needs to account for the host-transfer workaround, runtime overlay, wall-time schedule, and probability-parity discrepancy. A new portable training workflow should resolve those details rather than copy these study scripts without review.

The official [post-training guide](https://dev.meta.ai/docs/muse-glimmer/post-training) and [cookbook](https://github.com/meta-models/meta-oss-cookbook/tree/main/muse-glimmer/post-training) remain the reference for supported framework recipes. The procedure documented here is the specific, measured SiliconShield adaptation built on those model and training concepts.


## Reusable RTL training entry point

The repository now provides `siliconshield.training.glimmer.rtl.train` for
answer-token LoRA training. This is separate from the existing conversation
trainer in `siliconshield.training.glimmer.train`. Training Glimmer and using
Glimmer as an agent to operate a study are separate capabilities.

Start with [the training configuration](../configs/glimmer-rtl-training.example.json)
and [the answer contract](../configs/glimmer-rtl-contract.example.json).
Copy the configuration outside the repository and supply TRAIN, development,
base-model and output locations. Paths resolve from the current working directory.
The output directory must not exist for a new run.

Each JSONL row requires `id`, `domain`, `pair_id`, `split_group`, `state`, and
an integer `label`: 0 for satisfies, 1 for violates, 2 for insufficient context.
`state` contains the English requirement, assumptions and RTL in the format
shown above. Each domain/pair combination must contain all three labels and
one coarse lineage group. Use the same lineage group for related designs.
Label review and solver evidence remain prerequisites. Schema validation does
not establish semantic correctness or detect paraphrased leakage.

```bash
python -m siliconshield.training.glimmer.rtl.train \
  --config /path/to/training-config.json --validate-only

CUDA_VISIBLE_DEVICES=0,1,2 python -m siliconshield.training.glimmer.rtl.train \
  --config /path/to/training-config.json

CUDA_VISIBLE_DEVICES=0,1,2 python -m siliconshield.training.glimmer.rtl.train \
  --config /path/to/training-config.json \
  --resume /path/to/run/checkpoint-000025
```

Use the GPU environment versions recorded in this guide. The validation-only
command checks data structure and split separation without loading weights.
Tokenization and memory suitability are checked only on the GPU execution path.
The example map partitions the language model across three GPUs and keeps
unused vision components on CPU. It is model parallelism, not three independent
training replicas. Microbatch size is explicit and is not silently increased.

The trainer freezes the base, trains attention and MLP LoRA parameters, rejects
over-budget inputs, and records per-domain confusion matrices, class recalls,
false-safes, whole-triplet accuracy, conditional answer probabilities and answer
mass. Domain quotas interleave complete triplets without replacement until each
epoch is exhausted. They do not oversample a depleted domain.

Every evaluated checkpoint is saved before scoring. Checkpoints contain the
adapter, optimizer, sampler position, exposure counts and random states.
Resume requires the latest checkpoint, unchanged configuration, dataset bytes,
runtime versions and trainer source. Local base files are fingerprinted before
loading, which adds disk I/O. Only load trusted local optimizer checkpoints.
An exclusive run lock prevents duplicate writers. SIGINT and SIGTERM request a
stop after the current update, followed by a checkpoint and development score.
An abrupt failure recovers from the last committed checkpoint.

CPU-staged weight loading and GPU transfers are explicit options. They preserve
the workaround used in the original experiment and restore patched hooks when
the trainer exits. They are not a universal GPU performance recommendation.

The reusable schedule uses an update budget for warmup and cosine decay. The
original experiment used a wall-clock decay schedule. Consequently, this entry
point preserves the training method but is not a byte-for-byte replay of that
run. It does not automatically select or promote a model, open reserved test
sets, or launch another round. Development results must be reviewed against a
study's acceptance criteria before release.

Validation of this integration consists of synthetic CPU fixtures and repository
checks. A fresh multi-GPU training and resume qualification remains separate
from the already completed experiment described in this document.
