# Local Muse Glimmer deployment and qualification

[Documentation index](README.md)

Muse Glimmer runs locally as an agent for RTL assessment and decision-model
training. It reads evidence and requests actions through controlled tools.
The decision models receive optimizer updates during authorized training.
Glimmer's weights remain unchanged.

For Contract-1 and Contract-2 inference, use the [router guide](router.md) and
[decision-model qualification](expert-qualification.md). For Glimmer-operated training,
use the [local training guide](decision-model-training-with-glimmer.md).
The [case study](case-study-decision-model-training-with-glimmer.md) records the
measured operator scope. Serving health, successful tool execution, and
qualification of an agent role are separate results.

Local Glimmer can request router assessments through the
[router skill](router.md#driving-assessments-from-glimmer). CLI arbitration uses
the configured Muse CLI. Configure and qualify the intended integration rather
than treating a healthy endpoint as sufficient evidence.

## Choose a serving implementation

| Implementation | Environment | Use |
|---|---|---|
| Transformers reference server | Repository Dockerfile with pinned dependencies | Reference execution, one request at a time |
| Native vLLM server | Separately qualified build supporting `muse_glimmer` tool-call and reasoning parsers | Concurrent requests and optional speculative decoding |

Keep serving separate from the decision-model training environment. The
repository image installs the reference stack, not vLLM.

## Image and workspace

The reference server uses a pinned Transformers environment. Build its image
from the repository root:

```bash
docker build -t siliconshield-glimmer -f deploy/spark/glimmer/Dockerfile .
```

Keep the source snapshot, downloaded model, runtime locks, and replay outputs
in a workspace outside Git. Preserve model download metadata. The
[identity helpers](../siliconshield/training/glimmer/identity.py)
verify the pinned base revision and the full local snapshot before serving.
Record the image digest and source revision with deployment results.

For the reference server, install the package into that environment and run:

```bash
export SILICONSHIELD_GPU_LOCK=/path/to/shared-runtime/glimmer-gpu.lock
python -m siliconshield.research_loop.controller.serve \
  --model /path/to/base-checkpoint \
  --name glimmer-base --host 127.0.0.1 --port 18320
```

The endpoint is `http://127.0.0.1:18320/v1/chat/completions`. Requests use
`model: "glimmer-base"`. The reference server handles one request at a time.
For native kernels, concurrency, and optional speculative decoding, use the
[native server](#start-a-native-server). The reference image does not install vLLM.

All processes competing for the Glimmer device must use the same underlying
lease file. A lease refusal requires resolving resource ownership. It is not a
reason to change the lock path. A replay client calling a running endpoint uses
the server's existing ownership.

## Prerequisites and dry run

Install the package into the qualified inference environment. Confirm that its
vLLM build supports `muse_glimmer` tool-call/reasoning parsers and the
chosen speculation mode. The checked-in Glimmer Dockerfile installs the
Transformers reference-serving stack, not vLLM. This repository does not pin or
build a ready-to-run native inference image. Record the exact image and runtime
revision used for a deployment.

Inspect generated arguments before loading a model:

```bash
python -m siliconshield.research_loop.controller.serve_fast \
  --model /path/to/base-checkpoint --dry-run
```

Expected output is a JSON object containing the command argv. Dry run does not
validate checkpoint files, runtime availability, memory fit, or GPU ownership.
Check those during a real launch.

## Start a native server

Run this inside the inference environment, with local checkpoints mounted:

```bash
export SILICONSHIELD_GPU_LOCK=/path/to/shared-runtime/glimmer-gpu.lock
python -m siliconshield.research_loop.controller.serve_fast \
  --model /path/to/base-checkpoint
```

The server listens on loopback port 18320. Send requests to
`http://127.0.0.1:18320/v1/chat/completions` with `model: "glimmer-base"`.
This serves the base model without an adapter. The profile supports
text input.

The launcher verifies the full base snapshot. It holds the shared GPU lease
for the server's lifetime. Use the same host-mounted lock as other Glimmer
serving and in-process evaluation jobs. `--dry-run` prints the command without
loading models or acquiring a GPU.

Defaults are BF16, an 8,192-token context, up to eight concurrent requests, and
an 8,192-token scheduler budget. The memory fraction defaults to 0.65. Check
memory use under the intended workload. Unified-memory machines also need room
for the operating system and temporary loading buffers. Lazy shard iteration
avoids buffering an entire large shard in Python. A runtime with per-tensor CPU
staging can further reduce transfer overhead.

Keep both arms of a model comparison on the same runtime, precision,
prompt template, stop conditions, and decoding settings. Do not combine
results from different inference configurations into one comparison.

## Speculative decoding

An optional DFlash assistant can propose several tokens per target pass:

```bash
python -m siliconshield.research_loop.controller.serve_fast \
  --model /path/to/base-checkpoint \
  --draft /path/to/glimmer-assistant \
  --speculative-tokens 15
```

The target model verifies the proposals. Speculation changes the decoding
process, not the target weights. Qualify the runtime and decoding configuration
before using it for a comparison.
Measure acceptance and latency on the actual operator prompts. A low acceptance
rate can erase the speed benefit.

The default is fifteen proposed tokens. Compare draft lengths such as 4, 8, and
15 on the intended prompts. The fastest setting depends on the hardware, model,
and workload. Set `--max-batched-tokens` to at least `max_sequences * (1 + speculative_tokens)` so every draft position fits. The launcher checks this
capacity before starting the server.

Quantization is a separate change. This launcher retains BF16 target
weights and does not convert or overwrite any checkpoint.

## Demonstrated deployment

The validated router sessions ran against one Glimmer server with this
configuration:

- Host: DGX Spark (NVIDIA GB10), Ubuntu 24.04, driver 580.173.02.
- Server: vLLM in a container, loopback port 18400, served model
  `glimmer-base`. See [Served model identity](router.md#served-model-identity)
  for the weight revision and snapshot fingerprint.
- Decoding: BF16, 8,192-token context, up to eight concurrent requests,
  8,192-token scheduler budget, 0.65 GPU memory fraction.
- Parsers: `muse_glimmer` tool-call and reasoning parsers.
- Speculation: DFlash assistant, 15 proposed tokens.

Measured at concurrency one, after one discarded warm-up request, with a
short fixed prompt (73 input tokens, 128 generated tokens per request):

| Run | Time to first token | Total time | Overall | Decode |
| --- | --- | --- | --- | --- |
| 1 | 0.61 s | 10.0 s | 12.8 tok/s | 13.6 tok/s |
| 2 | 0.64 s | 10.0 s | 12.8 tok/s | 13.7 tok/s |
| 3 | 0.64 s | 10.0 s | 12.8 tok/s | 13.7 tok/s |

Decode throughput is generated tokens divided by total time minus time
to first token. Lifetime speculation acceptance on that server is
23.3% (107,252 accepted of 459,675 proposed draft tokens), measured
across all requests since the server started rather than on this
probe alone. These figures describe one short-prompt probe at server
defaults, not operator-prompt throughput; remeasure on the intended
workload before quoting them elsewhere.

## Timing

Report time to first token, generated-token count, total generation time,
and decode throughput separately. Also state concurrency, input length,
output length, warm-up policy, and whether prefix caching was active.
For streamed blocks, exclude every token in the first block when dividing
subsequent tokens by the time between first and last blocks.

The Transformers endpoint accepts `record_timing: true` for diagnostic requests.
The local evaluator exposes the same option as `--record-timing`. It returns
timing alongside `usage`. Per-token synchronization can itself add overhead.
Ordinary requests leave this instrumentation off. Replay trajectories retain
usage and timing metadata outside the messages shown to the model. Missing
server timing stays missing rather than being estimated from visible prose.

Measure speed at concurrency one before quoting single-response latency.
Aggregate throughput across several requests answers a different question.
Keep runtime measurements and generated responses outside Git.

## Replay and qualification

Controller episodes exercise evidence retrieval, action selection, explanations,
and final-response consistency. They do not train Glimmer. Existing episode
manifests retain their `train`, `dev`, and `test` partition names for compatibility
with historical records. Reused populations remain development or regression
material, not new independent qualification.

To rehearse the authored regression cases without a model:

```bash
python -m siliconshield.research_loop.episodes.operator_baseline \
  --build /path/to/private/regression-cases
python -m siliconshield.research_loop.episodes.operator_baseline \
  --episodes /path/to/private/regression-cases/baseline.jsonl \
  --output /path/to/private/fixture-replay --fixture
```

This checks the tool and grading contracts. For a live replay, replace
`--fixture` with `--endpoint http://127.0.0.1:18320/v1/chat/completions
--model glimmer-base`. Report it as model execution on authored regression
cases, separately from fixture validation or independent qualification.

For manifest-bound development episodes, use
`siliconshield.research_loop.controller.replay`. Keep the episode population,
prompt, tool schema, runtime, decoding settings, and model identity fixed when
comparing configurations. Inspect each action class and each grading component.
The [replay selection helper](../siliconshield/training/glimmer/selection.py)
checks action-level non-regression on matched reports. A gain in one class does
not excuse a failure in another.

Harness² can propose and replay procedural guidance without changing model
weights or authorizing jobs. See [Muse procedures](muse-procedures.md) for
bundle provenance and adoption rules. Decision-model jobs remain subject to
the [study requirements](study-walkthrough.md#study-requirements) and
[campaign controls](campaigns.md).

## Controller evaluation

The controller has different targets from the RTL model. It must read evidence,
choose an allowed action, calculate changes, and report consistently.

| Field or dimension | What it establishes |
|---|---|
| `action_correct` | Agreement with the episode's declared action policy. |
| `evidence_correct` | Structured numbers and evidence references satisfy the contract. |
| `explanation_correct` | Deterministic explanation checks did not identify unsupported claims. |
| `final_consistent` | Final structured response agrees with the recorded action/evidence. |
| Execution completion | Tool protocol completed without errors. |
| `verified` | The replay's combined checks passed. |

A deterministic grader is not an independent assessment of arbitrary scientific
reasoning. Report each component, per-action support, and lineage distribution.
Synthetic fixture replay tests interfaces. Live generation measures the model
under the chosen prompt, parser, runtime, and decoding settings.

`select_replay_candidate()` requires matched population, evaluator, source,
manifest, and decoding identities, coverage of all six operator actions, and
per-action/cohort non-regression against every supplied reference. It requires a
net verified gain. Selection records a replay comparison. It does not deploy a configuration or
update model weights.

## Sealed controller evaluation

Keep reserved controller evaluation separate from development replay and any
proposer or lesson-retention loop. Freeze the episode population, prompt, tool
schema, parser, runtime, and decoding configuration before evaluating it.
For the optional existing-adapter evaluator's authenticated `--final` mode,
see the [compatibility reference](reference.md#existing-adapter-compatibility).

Repeated development replay is useful for selection but should not be presented
as a fresh final benchmark. Neither correct arithmetic nor a completed job proves
that an intervention caused a better security model.

## References

- [Meta deployment guide](https://dev.meta.ai/docs/muse-glimmer/deploy)
- [Meta vLLM recipe](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/inference-server/vllm.md)
- [Meta SGLang and DFlash recipe](https://github.com/meta-models/meta-oss-cookbook/blob/main/muse-glimmer/inference-server/sglang.md)
