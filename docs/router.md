# Router architecture

[Documentation index](README.md) · [Expert qualification](expert-qualification.md) ·
[Benchmark](benchmark.md)

The router connects an explicit assessment scope to specialists with relevant
measured capabilities. It preserves each model's result, ranks the available evidence,
and makes coverage gaps explicit before optional Muse arbitration. No single
model covers the whole design space. Scope tags identify the property class,
and measured profiles determine which specialist evidence is available. The
Contract-1 decision model handles local guard requirements. The Contract-2
decision model handles FIFO and counter requirements. Anything outside measured
support stays visible instead of receiving an unqualified verdict. This is application-level expert routing,
not a jointly trained neural mixture of experts. Improvement over the best
single model requires a matched evaluation.

## Assessment transaction

![Sequence diagram of a scoped assessment, cold worker loading, validated responses, and optional Muse arbitration.](diagrams/expert-assessment.svg)

**Figure 02.** Request/reply sequence for the experts CLI/API. Solid arrows are
calls and dashed arrows are replies. The loop consults experts sequentially.
[Open full-size SVG](diagrams/expert-assessment.svg) ·
[Editable draw.io source](diagrams/siliconshield-architecture.drawio)

The experts CLI and Python API expose this path. Modules live in
`siliconshield/experts/`, with evaluation in
`siliconshield/benchmark/`. The two current decision models are
[Contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1)
(local guards) and
[Contract-2](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2)
(FIFO/counter specialist), routed by tiers measured on the
[public RTL benchmark](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-benchmark).

## Run one assessment

Prepare the input as a small verification problem:

1. Name the protected state and one required behavior. Distinguish a request
   from an accepted operation, particularly for FIFO and handshake logic.
2. Include every relevant writer, guard, helper definition, signal width, and
   priority branch. Preserve the source locations of extracted RTL.
3. State the reset scope and any environment constraints that affect the answer.
4. Choose mechanism tags supported by the registered capability profiles.

First register immutable exports and measured capability profiles as described
in [expert qualification](expert-qualification.md). A registry without
capability profiles provides no routable experts.

The state file uses literal section headers, without colons:

```text
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
```

Here `full` is an environment-supplied signal. The property checks hold behavior
when that signal is asserted. It does not check how `full` is generated, the
queue depth, or stored-data ordering. This is an authored format example, not a
measured model prediction. Save it outside Git and run:

```bash
python -m siliconshield.experts assess \
  --registry /path/to/models.local.json \
  --scope fifo --state-file /path/to/assessment.txt
```

For arbitration, add `--arbitrate` and pass `--requirement` equal to the full
requirement in that file. Muse uses the configured local CLI and existing login.
Adding `--issue`, `--relevance`, and repeatable `--rtl` locations requests a
traceable finding as well. For example:

```bash
python -m siliconshield.experts assess \
  --registry /path/to/models.local.json \
  --scope fifo --state-file /path/to/assessment.txt \
  --arbitrate \
  --requirement "After reset, a push while full must leave occupancy unchanged." \
  --issue "Occupancy update during backpressure" \
  --relevance "A count change could invalidate the full condition." \
  --rtl "topmodule:occupancy update"
```

The JSON contains `assessment`, and optionally `arbitration` and `finding`.
Inspect `assessment.errors`, effective `routing`, and `thin_evidence` before
using it. An `arbitration_error` preserves the expert assessment but supplies no
Muse verdict. A zero CLI exit alone is not a successful assessment.

Workers live for this command's duration. A persistent Python application can
reuse `InferenceService` and `WorkerPool` across calls. This is a local-pipe
service API, not a bundled HTTP endpoint. Muse-assisted paths may transmit
RTL and evidence to the configured provider.

## Driving assessments from Glimmer

Glimmer can drive the same assessment path through two tools,
`router.roster` and `router.assess`, with the registry bound by the
operator. The [router skill](../.agents/skills/glimmer-router/SKILL.md)
is the system prompt. It covers scope discipline, evidence reading,
and the verdict format. Run one session with:

```bash
python -m siliconshield.research_loop.procedures.glimmer_router \
  --registry /path/to/models.local.json \
  --endpoint http://127.0.0.1:18320/v1/chat/completions \
  --model glimmer-base \
  --task-file /path/to/task.txt --output /path/to/session
```

The session records messages, tool results, and the state files submitted for
assessment. Specialist predictions come from the validated router tool. Glimmer
chooses the tool arguments and writes a final conversational response. The
bridge does not pass that response through the experts CLI arbitration
validator. Review the trace to confirm that the assessed RTL and requirement
match the task and that the summary follows the tool evidence. See
[Glimmer inference](inference.md) for serving and model naming.

Demonstrated on two base-model sessions against local experts.
Broader driver reliability is not established. Session verdicts
inherit the experts' documented domain limits.

## Routing flow

1. **Scope.** The caller names mechanism tags (for example `fifo`).
   `inspection.py` can assist the caller with lexical evidence, but the CLI
   and service do not infer scope from it. Use only tags represented by the
   capability profiles. Unratable tags stay visible instead of being dropped.
2. **Profile gate.** An expert is eligible only when its capability
   profile claims every tag. Unclaimed scopes route nowhere and are
   reported unsupported. Experts covering only some tags are
   reported as partial coverage for scope splitting, never
   consulted.
3. **Tier rank.** Eligible experts sort by worst area tier
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

## Transaction invariants

| Invariant | Enforcement | Observable outcome |
|---|---|---|
| One input per consultation | The same state text is sent to all routed experts. Replies bind its SHA-256 and the active worker request. | A foreign or stale reply is rejected. |
| Full scope coverage | A profile must claim every scope tag to be consulted. A union of partial profiles is not treated as one qualified expert. | Partial and unsupported coverage are returned separately. |
| Evidence follows execution | Runtime identity must match the profile's measured execution identity for its area tier to apply. | Changed execution is unrated until measured. |
| Failures remain visible | Transport and validation errors are separate from successful model responses. | No error is converted to `insufficient_context`. |
| Coverage is recomputed | Effective routing uses only successful replies and their observed identities. | A failed expert cannot make remaining evidence appear stronger. |
| Arbitration is bound | State, complete requirement, assumptions, request, and scope must agree with the assessment. | Incompatible arbitration refuses while the original assessment remains available. |

The [wire reference](reference.md#decision-and-worker-contracts) specifies the
score tolerance, coverage fields, identity fields, and deadlines.

## Reliability tiers

Tiers derive from benchmark accuracy per (model, area): strong at
or above 80%, partial at 60%, weak at 40%, and none below 40%, with a
one-tier demotion when false-safes exceed 15% of violations. The
profile records correct, total, violation, false-safe and error counts
plus benchmark bytes and the observed execution identity. Areas with
execution errors are unrated. Historical measurements alone do not
establish that the current runtime has the same behavior.
Re-measure from benchmark reports when data or checkpoints change.
Never edit tiers by hand.

## Provisioning future models

Use [expert qualification](expert-qualification.md) to register an immutable
export, measure its capability profile, qualify routing, and retain a rollback
release. That guide also covers upgrades and removal. A different output task
or worker protocol needs an explicit integration change.

## Verification

- Unit: `tests/test_multi_expert.py` (routing, envelopes, service,
  arbitration) and `tests/test_benchmark.py` (metrics, runner,
  router sampling) run CPU-only with synthetic fixtures.
- Live: `benchmark run` and `benchmark router-eval` exercise real
  checkpoints through real workers. The fifo/counter slice is the
  fast integration smoke check for both contracts.
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
