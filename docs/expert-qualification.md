# Qualifying experts and releases

[Documentation index](README.md) · [Router](router.md) · [Benchmark](benchmark.md)

A specialist compatible with the current task and worker protocol joins through
a registry entry, capability profile, question contract, and qualification
reference. A different output task or runtime protocol needs an integration
change. Registration, measured capability, and end-to-end router qualification
are separate stages. Loading an entry does not establish all three.

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

For an identity-bound measurement, capture the ready receipt from the same
worker used to score the population:

```python
from siliconshield.benchmark.dataset import load_benchmark
from siliconshield.benchmark.runner import BenchmarkRunner
from siliconshield.experts.registry import load_registry
from siliconshield.experts.worker import WorkerPool

entry = load_registry("/path/to/models.local.json")[0]
data = load_benchmark("/path/to/reviewed-benchmark.jsonl")
with WorkerPool() as pool:
    pool.load(entry)
    ready = pool.ready(entry["id"])
    report = BenchmarkRunner(data).run_model(
        entry["id"],
        entry["expected_files"]["model.safetensors"],
        lambda state: pool.analyze(entry["id"], state),
    )
```

Persist `ready` and `report` together outside Git. Convert each `by_area` record
to `correct`, `total` (from `rows`), `violations` (violation class support),
`false_safe`, and `errors`. Use `ready["execution_identity"]`, the measured
`benchmark_source`, and `benchmark_sha256` for `routing_evidence`. The command-line
benchmark report does not currently retain this full ready receipt.
Validate the resulting profile with `validate_capability_profile()` before
referencing it from a registry. Initial measurement can use an unprofiled model
entry. Benchmarking one model does not require a routing profile.

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
