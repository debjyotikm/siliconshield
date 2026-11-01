# Router architecture

Several decision models judge RTL better than one only when each is
consulted on its demonstrated strengths and the gaps stay explicit.
The router sends every assessment scope to the qualified experts,
ranks them by measured reliability, and flags thin evidence instead
of forcing a verdict. This page describes the flow, the evidence
behind it, and how future models join without code changes.

```
RTL + requirement + assumptions
              │
              ▼
     ┌────────────────┐   ┌────────────────────────┐
     │  inspection    │   │  router + routing_data │
     │  writers,guards│──▶│  profile gate → tier   │
     │  state,symbols │   │  rank → thin? rationale│
     └────────────────┘   └───────────┬────────────┘
                                      │ experts + tiers
                           ┌──────────▼───────────┐
                           │  service + workers   │
                           │  one subprocess each │
                           └──────┬───────┬───────┘
                                  │       │
                    ┌─────────────▼─┐ ┌───▼──────────┐
                    │  Contract-1   │ │  Contract-2  │
                    │  local/transfer│ │  fifo/counter│
                    └─────────────┬─┘ └───┬──────────┘
                                  │       │
                           ┌──────▼───────▼───────┐
                           │  responses (labels,  │
                           │  scores, checkpoint, │
                           │  input hash, errors) │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │  arbitration (Muse)  │
                           │  suitability + RTL   │
                           │  evidence + dissent  │
                           └──────────┬───────────┘
                                      │
                           ┌──────────▼───────────┐
                           │  findings (traceable,│
                           │  model_assessment,   │
                           │  open questions)     │
                           └──────────────────────┘
```

The GUI demo consumes this path. It owns none of it. All modules
below live in `siliconshield/experts/` (Qt-free) and
`siliconshield/benchmark/`. The two current experts are
[Contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1)
(local guards) and
[Contract-2](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2)
(FIFO/counter specialist), routed by tiers measured on the
[public RTL benchmark](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-benchmark).

## Routing flow

1. **Scope.** The caller names mechanism tags (for example `fifo`
   plus a mechanism id). Structural inspection grounds the scope in
   the RTL. Unratable tags stay visible instead of being dropped.
2. **Profile gate.** An expert qualifies only when its capability
   profile claims every tag. Unclaimed scopes route nowhere and are
   reported unsupported. Experts covering only some tags are
   reported as partial coverage for scope splitting, never
   consulted.
3. **Tier rank.** Qualified experts sort by worst area tier
   (strong, partial, weak, none, unrated), profile order breaking
   ties. Tiers come from the profile’s `routing_evidence` counts and full
   execution identity. The historical table in `routing_data` is
   not current authority. Changed weights, questions, runtime or
   inference settings produce an unrated result until remeasured.
4. **Thin flag.** The decision is thin-evidence when any tag is
   unsupported or unratable, nothing covers the scope, or the best
   consulted worst-tier is weak or worse. Rationale lines quote
   per-area tiers as evidence.
5. **Consult.** The shared service sends the same state to every
   routed expert over persistent worker subprocesses. Replies are
   validated through the result protocol. Transport and validation
   failures return as errors, never as labels. Models load on demand
   after routing, so an unrelated broken expert cannot block the scope.
   The returned `routing` is recomputed from successful replies.
   `planned_routing` preserves the original consultation plan.
6. **Arbitrate.** Muse weighs expert suitability, RTL evidence, and
   disagreements into one verdict with open questions. Votes and
   raw confidence never decide alone. Exact state, request, scope,
   full requirement and assumptions must match the expert evidence.
   Incompatible question contracts are refused. A failed expert cannot
   supply evidence strength to the surviving responses.
7. **Finding.** The verdict binds to the issue, requirement, RTL
   locations, consulted experts, and unresolved questions, always
   labeled a model assessment.

## Reliability tiers

Tiers derive from benchmark accuracy per (model, area): strong at
or above 80%, partial at 60%, weak at 40%, none below -- with a
one-tier demotion when false-safes exceed 15% of violations. The
profile records correct, total, violation, false-safe and error counts
plus benchmark bytes and the observed execution identity. Areas with
execution errors are unrated. Historical measurements alone do not
establish that the current runtime has the same behavior.
Re-measure from benchmark reports when data or checkpoints change.
never edit tiers by hand.

## Provisioning future models

A new checkpoint joins through configuration and measurement. No
router, service, or arbitration code changes.

1. **Registry entry.** Add id, checkpoint, interpreter, runtime,
   device, and the full file manifest. Keep the previous release
   entry for rollback.
2. **Capability profile.** Write strengths, weaknesses, claimed
   mechanisms, known failures, and measurements, all bound to the
   checkpoint hash. Claim only areas with measured evidence. The
   router treats unrated tags as thin.
3. **Benchmark.** Score the pinned public benchmark with
   `python -m siliconshield.benchmark run` and keep the report.
4. **Evidence binding.** Put the measured area counts and benchmark
   SHA-256 in the profile’s `routing_evidence`, together with the
   worker’s observed `execution_identity`. The router derives tiers
   automatically. Keep the full report alongside the profile.
   An absent or mismatched binding yields unrated coverage.
5. **Router eval.** Run `router-eval` over sampled examples to
   confirm the newcomer is consulted where it should be and the
   thin flag fires where it should not be trusted.
6. **Skill update.** Record the new expert, its tiers, and its
   failure modes in the RTL analysis skill so Muse routes and
   arbitrates from current evidence.

Upgrades follow the same path as new models: a new release keeps
its own profile, benchmark report, and execution identity, and the old
release stays available until the new one proves itself. Removing
a model means removing its registry entry. Scopes it
alone covered then route to explicit unsupported, never silently
to a weaker expert.

## Verification

- Unit: `tests/test_multi_expert.py` (routing, envelopes, service,
  arbitration) and `tests/test_benchmark.py` (metrics, runner,
  router sampling) run CPU-only with synthetic fixtures.
- Live: `benchmark run` and `benchmark router-eval` exercise real
  checkpoints through real workers. The fifo/counter slice is the
  fast smoke proof for both contracts.
- Gates: ruff, format, compile, mypy, and pylint per the Python
  style guide before any change lands.

## Runtime evidence contract

A validated ready receipt names the model, device, question contract,
checkpoint hash, full export digest, runtime Python tree digest, question
bytes and numerical settings. Settings include precision, Torch version,
batch size, option order, token budgets and length sorting. Every result
must come from that worker session and name the exact input hash.
Coverage must explicitly report zero dropped tokens and no truncated
questions. Missing or contradictory runtime coverage is an error.

Load and analysis timeouts are total request deadlines. Progress or stale
messages cannot extend them. A failed worker generation is invalidated and
reaped. Calls to one worker are serialized so simultaneous requests cannot
consume each other’s replies.

An injected capability profile is subject to the same validation as a
profile loaded from disk. A registry name alone never establishes identity.
Profiles without matching measured evidence can still be consulted, but
their results remain unrated and thin.

Inspection is a lexical aid, not a SystemVerilog parser. Comments are
excluded, declaration initializers do not declare their operands, and
balanced guards and indexed targets are handled. The result always states
that completeness, macro expansion, elaboration and reachability remain
unverified. The caller still supplies the scope tags.

Regression coverage for these boundaries lives in
`tests/test_expert_safety.py`. Live smoke tests show integration behavior,
not an independent model-quality or arbitration qualification.
