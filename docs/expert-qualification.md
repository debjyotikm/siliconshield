# Register and qualify decision models

[Documentation index](README.md) · [Router](router.md) · [Benchmark](benchmark.md)

A specialist compatible with the current task and worker protocol joins through
a registry entry, capability profile, question contract, and qualification
reference. A different output task or runtime protocol needs an integration
change. Registration, measured capability, and end-to-end router qualification
are separate stages. Loading an entry does not establish all three.

**Prerequisites:** a working CPU or CUDA decision-model worker, immutable
exports, and an unprofiled registry from
[setup](setup.md#register-both-decision-models). CUDA requires a compatible
GPU driver. Network access is needed for public downloads, not inference on
existing local artifacts.

## Register and measure the published models

Complete [model registration](setup.md#register-both-decision-models) first,
then install `requirements/benchmark.txt` in the controller environment.
The example below uses that setup's workspace and `models-unprofiled.json`.
Measurements run on each entry's registered CPU or CUDA device. A CPU run
cannot inherit CUDA capability measurements.

Use the [reviewed benchmark procedure](benchmark.md#reproduce-the-reviewed-population)
to prepare `reviewed-benchmark/benchmark-reviewed.jsonl` and its provenance
record. Then measure both decision models on FIFO. This command
**loads models and runs inference** on the selected device, one model
at a time. It makes no Muse calls and does not train:

```bash
python examples/qualify_experts.py \
  --registry "$SILICONSHIELD_WORKSPACE/models-unprofiled.json" \
  --dataset "$SILICONSHIELD_WORKSPACE/reviewed-benchmark/benchmark-reviewed.jsonl" \
  --areas fifo --output "$SILICONSHIELD_WORKSPACE/fifo-measurement"
```

The new output directory contains each worker's ready receipt, its measurement
report, its validated capability profile, and `models.local.json`. The example
refuses an existing output directory and refuses to publish a profile when
inference errors occur. Keep partial receipts, fix the cause, and retry into a
new directory. Use the resulting registry in the [router example](router.md#run-one-assessment).

This measures **99 FIFO development decisions per model**. It establishes
coverage only for `fifo`, not every capability of either release. Add reviewed
areas deliberately and remeasure to extend coverage. The router computes tiers
from measured counts and observed execution identity. The example assigns none
by hand and does not claim independent qualification.

## Admitting a new expert

1. Add a registry entry with a unique id, checkpoint path,
   `profile`, `release`, `question_contract`, and `qualification`
   fields. Profile paths resolve against the registry file.
2. Author a capability profile matching
   `EXPERT_CAPABILITY_SCHEMA`. Bind every strength claim to a
   checkpoint SHA-256 and a measured evaluation. Record known
   failure cases explicitly.
3. Name the exact question contract the checkpoint was trained
   and evaluated under. Experts on different contracts must not
   share pooled verdicts without a reviewed mapping.
4. Record per-area measurements and the worker’s observed execution identity
   alongside the benchmark bytes in `routing_evidence`. Keep the complete
   report outside Git and store its qualification reference in the profile.
   A profile without a matching binding is unrated. Evaluate the combined
   routing/arbitration system separately before claiming an advantage over
   either specialist.
5. Run [router evaluation](benchmark.md#evaluate-the-router) on sampled cases.
   Check consultation, unsupported scopes, and thin-evidence reporting before
   the combined-system qualification described below.
6. Update the [RTL analysis skill](../.agents/skills/rtl-analysis/SKILL.md) when
   expert availability or documented failure modes change. Keep numerical
   routing support in measured profiles rather than assigning tiers by hand.

## Upgrading an expert

An upgrade has a new checkpoint, profile, and qualification reference.
To keep both loaded, use distinct registry ids and matching profile ids.
Duplicate ids in one registry are refused. Alternatively, retain the previous
registry generation and switch one stable id to the new immutable release.
Keep the earlier export and profile for rollback. Never overwrite released
artifacts in place.

Removing a model removes its registry entry. Recheck affected scopes: if no
remaining profile covers a scope, the router must report it unsupported.

## Routing and arbitration baselines

Before claiming that the combined system improves on individual experts,
freeze a comparison protocol and measure:

- Per-mechanism routing: which scopes each expert covers and
  which are reported unsupported.
- Arbitration outcomes versus each consulted expert's own
  label on the qualification set.
- Violation recall, false-safes, and false violations, split by mechanism and
  agreement/disagreement cases. A violation predicted as context is a missed
  violation but is not a false-safe.

A suitable acceptance criterion is non-regression against the best individual
expert on missed violations, a declared false-alarm budget, and explicit
unsupported scopes. Register exact count limits before evaluation. This is a
system-qualification protocol, not a result implied by profile validation or
registry admission.

## Running assessments

Use the [complete router example](router.md#run-one-assessment) for the state
file, scope, and matching arbitration requirement. The CLI returns planned and
effective routing, each successful expert response, and separate errors.
Optional arbitration adds a structured agent conclusion. Issue metadata can
then bind that conclusion to a finding and the caller's RTL locations.

## Capture the measured worker identity

A profile requires a unique id, release, full weights hash, question contract,
nonempty strengths and supported-mechanism lists, known weaknesses/failures,
qualification reference, and measurements. Its schema is
`EXPERT_CAPABILITY_SCHEMA` in `siliconshield/schemas.py`. The registry uses
the schema identifier and `models` list validated by
[registry.py](../siliconshield/experts/registry.py). Each entry names its `id`,
`display_name`, `checkpoint`, `python`, `laya_root`, `device`, and the complete
`expected_files` mapping. It also needs `profile` to participate in routing.
Use the complete file hashes in the [release manifest](../configs/model-releases.toml)
when registering the published exports. Replace local paths with those from
your [release setup](releases.md), then attach the measured profile.

The [executable example](../examples/qualify_experts.py) captures the ready
receipt and report from the same worker session, converts per-area counts,
validates the profile, and writes a separate registry. Use it as the reference
for custom integrations. The benchmark CLI alone does not retain the full ready
receipt needed for this binding.

## Capability evidence shape

The optional `routing_evidence` object contains `benchmark_source`,
`benchmark_sha256`, `execution_identity`, and `areas`. Each area supplies
integer `correct`, `total`, `violations`, `false_safe` and `errors` counts.
Use the identity returned by the measured worker, including its question,
runtime and inference settings. Do not copy an identity from a different
run or assign tiers by hand. Changes to the inference protocol need new
measurements even when the checkpoint bytes are unchanged.

When a benchmark requirement disagrees with its assertion, quarantine the
whole linked triplet from new capability evidence. Preserve the original
file and receipt, issue a separately identified correction, and review it
before restoring it to qualification. A wording correction is a new input
and cannot inherit old predictions.
