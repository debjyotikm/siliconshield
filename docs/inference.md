# Glimmer inference

[Documentation index](README.md)

The Transformers server is the reference implementation. For repeated
operator replays, use a vLLM build with native Muse Glimmer support and the
`muse_glimmer` tool-call and reasoning parsers. Keep the inference environment
separate from the pinned training environment.

## Prerequisites and dry run

Install the package into the qualified inference environment. Confirm that its
vLLM build supports `muse_glimmer` tool-call/reasoning parsers, LoRA, and the
chosen speculation mode. The checked-in Glimmer Dockerfile installs the
Transformers training/reference stack, not vLLM. This repository does not pin or
build a ready-to-run native inference image. Record the exact image and runtime
revision used for a deployment.

Inspect generated arguments before loading a model:

```bash
python -m siliconshield.research_loop.controller.serve_fast \
  --model /path/to/base-checkpoint \
  --adapter trained=/path/to/adapter-checkpoint --dry-run
```

Expected output is a JSON object containing the command argv. Dry run does not
validate checkpoint files, runtime availability, memory fit, or GPU ownership.
Check those during a real launch.

## Start a native server

Run this inside the inference environment, with local checkpoints mounted:

```bash
export SILICONSHIELD_GPU_LOCK=/path/to/shared-runtime/glimmer-gpu.lock
python -m siliconshield.research_loop.controller.serve_fast \
  --model /path/to/base-checkpoint \
  --adapter trained=/path/to/adapter-checkpoint
```

The server listens on loopback port 18320. Send requests to
`http://127.0.0.1:18320/v1/chat/completions` with `model: "trained"`.
`glimmer-base` names the base without an adapter. Each adapter needs a unique
name. The profile supports rank-16 and smaller LoRA adapters and text input.

The launcher verifies the base snapshot and hashes adapter files. It holds
the shared GPU lease for the server's lifetime. Use the same host-mounted
lock as training and evaluation. `--dry-run` prints the command without
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
  --adapter trained=/path/to/adapter-checkpoint \
  --draft /path/to/glimmer-assistant \
  --speculative-tokens 15
```

The target model verifies the proposals. The assistant does not replace the
trained adapter. Support for speculation together with LoRA depends on the
runtime build. Qualify that combination before using it for a comparison.
Measure acceptance and latency on the actual operator prompts. A low acceptance
rate can erase the speed benefit.

The default is fifteen proposed tokens. Compare draft lengths such as 4, 8, and
15 on the intended prompts. The fastest setting depends on the hardware, model,
and workload. Set `--max-batched-tokens` to at least `max_sequences * (1 + speculative_tokens)` so every draft position fits. The launcher checks this
capacity before starting the server.

Quantization is a separate change. This launcher retains BF16 target
weights and does not convert or overwrite any checkpoint.

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

## References

- [Meta deployment guide](https://dev.meta.ai/docs/muse-glimmer/deploy)
- [Meta vLLM recipe](https://github.com/meta-models/meta-oss-cookbook/blob/main/inference-server/vllm.md)
- [Meta SGLang and DFlash recipe](https://github.com/meta-models/meta-oss-cookbook/blob/main/inference-server/sglang.md)
