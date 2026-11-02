---
name: rtl-analysis
description: Analyze RTL for functional bugs and security failures with explicit requirements, scoped evidence, and multi-expert arbitration.
---

# RTL Analysis

Use this skill when investigating RTL for functional bugs or security
failures with SiliconShield decision models. It covers what to look
for, how to scope the analysis, which expert to consult, and how to
reach a traceable final verdict.

## Functional bugs vs security failures

A functional bug makes the design behave differently from its intent
under reachable conditions: wrong reset values, dropped or duplicated
transactions, lost interrupts, incorrect counters, arbitration that
starves or misroutes. A security failure additionally breaks a
protection property: an authorization bypass (a write path that skips
a lock, privilege, or range check), information exposure (secret,
key, or capability state reaching an unauthorized reader or persisting
past a wipe), or an unsafe state transition (lockdown, wipe, or mode
change that can be skipped, reordered, or raced).

Name every finding as functional, security, or both, and say which.
A locked register that fails to update is functional. A debug path
that updates it while locked is a security failure. When the
requirement does not settle the question, say so instead of guessing.

## Requirements, assumptions, evidence

Every assessment starts from one explicit requirement in plain
language, the assumptions that bound it (reset behavior, clocking,
which inputs are trusted, what is out of scope), and the exact RTL
region judged. Quote the requirement verbatim in the finding, cite
file paths with line numbers for every relied-upon statement, and
mark any requirement you inferred yourself as inferred. Never present
an inferred requirement as the design's documented intent.

Verify each relied-upon fact against the shown RTL: the writer of
every assigned signal, the guard of every branch, the reset value of
every state element, and every path that can reach a sensitive sink.
If a wire, helper, or parameter needed for the verdict is not shown,
determine whether compatible completions could change the answer.
Only then is insufficient context justified. An irrelevant omitted
symbol alone does not make the requirement undecidable. Keep
execution errors (tool failures, truncated inputs) separate from
insufficient context: the first means the analysis did not run, the
second is a model verdict about missing information.

When a model disagrees with a solver-labeled example, compare the
plain-language requirement with the actual assertion before blaming
either model. An assertion may add a state or interface obligation
that the wording never states. Preserve the label and receipts for
semantic review. Do not call a mismatch a confirmed false-safe or
silently rewrite the example to make an evaluation pass.

`inspection.inspect_structure` supplies text-scanning hints, not a
SystemVerilog parse or elaboration. Check comments, declaration
initializers, nested conditions, and indexed writes against the RTL
before treating its writers or unknown symbols as evidence. The
current router consumes caller-supplied scope tags. It does not
discover a design's domain automatically.

## Experts and routing

Consult decision models through their capability profiles, never by
name alone. Current releases:

Public artifacts:
[Contract-1](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-1),
[Contract-2](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-2),
[RTL benchmark](https://huggingface.co/datasets/debjyotikm/siliconshield-rtl-benchmark).

- Contract-1 (registry id `contract-1`, checkpoint f4130223):
  strongest measured on local guard and direct-write requirements
  (register locks, privilege checks, exact address windows).
  Measured on RTL benchmark v1 (v1 prompt): 606/621
  transfer-pairs, 2601/2610 local-safety, 108/108 OT access
  control. (Ancestor legacy-prompt scores ran slightly higher.
  The v1 figures above are the routing evidence.) Known weakness:
  multi-cycle temporal behavior (FIFO/counter/handshake families)
  and unfamiliar implementation styles. Do not route those here
  alone.
- Contract-2 (registry id `contract-2`, checkpoint f916444f):
  FIFO/counter specialist. Measured on its selection set: 203/234
  decisions, 78/78 missing-context, 51/78 whole triplets. Perfect
  on 8 of 20 mechanisms (saturation, backpressure, handshakes).
  Area benchmark: fifo 97/108, counter 92/108, arbitration 14/18.
  NO transferable safety judgment -- local-safety 933/2610,
  requirement-clarity 271/756, write-paths 244/432, and it misses
  most access-control violations (31 false-safes among 36 violations
  in the 108-row OT benchmark area). In-domain
  violation recall is 61/78 with 17 false-safes. Commit-hold,
  arbiter-tree, and empty-data-blanking families are unreliable.
  Owner-selected research release. Never route local-safety,
  requirement, transfer, or access-control scopes here alone.

Prefer Contract-1 for qualified local requirements and the temporal
expert for its qualified temporal families. Consult both when a scope
overlaps, and report explicitly when no qualified expert covers the
scope instead of forcing a verdict from an unsuitable model. When the
router reports partial coverage (each expert covers only some tags),
split the scope into independently judgeable requirements and route
each part separately rather than consulting a partial expert on the
whole.

Registry ids differ from release names such as
`siliconshield-rtl-contract-1`. Match the loaded worker, registry,
profile, and evidence by actual artifact and question contract.
Do not reuse a tier just because the weights or display name match:
runtime, prompt, option order, and batching must match the measured
protocol. Changed protocols need fresh measurements or unrated
evidence. Configuration claims and a successful load alone do not
establish qualification.

## Arbitration

Each expert returns a three-way label with all scores, its checkpoint
identity, the exact input judged, and coverage status. Arbitration
receives the judged state, the requirement, routing tiers and thin
flags, expert profiles, and every response and error as one bound
object. Verify each response's `state_sha256` against the exact
state passed to arbitration. Match the complete REQUIREMENT section
and its assumptions, not a substring that could omit a constraint.
Do not arbitrate a changed RTL excerpt against cached predictions.
The shared arbitration helper enforces these bindings. Preserve them when
passing assessments to Muse or building a finding.

Read all evidence before concluding and cite only experts with
successful, identity-matched responses in this assessment. Explain
the final verdict from expert suitability for the
scope, the cited RTL evidence, and how disagreements were resolved.
Confidence scores and majority votes inform but never decide: a
confident verdict from an unsuitable expert loses to a cautious one
from the qualified expert, and any unresolved question stays open
in the finding. With no usable response, the verdict is unresolved,
not a guess. Reassess evidence strength after consultation: a failed
strong expert cannot lend its tier to a surviving weak expert.
Use `routing` for effective coverage and `planned_routing` only to explain
the attempted consultations. When a failed expert leaves only thin evidence,
the shared arbitration path returns unresolved without calling Muse.

Check the worker's ready reply against the requested model, device,
and question contract. Preserve load failures and timeouts as
execution errors, with no substitute verdict. Bound the total
request duration, including progress messages. A frozen source
version, full registry, and observed ready replies belong in the
private review receipt.

Simulation and formal verification are deferred: label every
conclusion a model assessment, note what would confirm or refute it,
and never claim a proof the tools did not run.

## Historical routing measurements

The reference figures below came from the public [RTL
benchmark](https://huggingface.co/datasets/debjyotikm/siliconshield-rtl-benchmark). Current tiers come from profile `routing_evidence` only when
its execution identity matches the successful worker response. Historical
figures do not establish current tiers. The rule is strong (>=80%),
partial (>=60%), weak (>=40%), none (below), with a one-tier
demotion when false-safes exceed 15% of violations. The router
ranks consulted experts by tier and flags thin evidence whenever
the best basis is weak or worse.

- local-safety-cwe: Contract-1 strong, Contract-2 none. Route here.
- transfer-pairs: Contract-1 strong, Contract-2 weak. Route here.
- ot-access-control: Contract-1 strong, Contract-2 weak. Route here.
- fifo, counter: Contract-2 partial, Contract-1 none. Route here.
- arbitration, interrupt-wake: Contract-1 partial, Contract-2 weak.
  Consult both. They err in opposite directions.
- input-filter, write-paths: both weak. Consult both, thin evidence.
- guard-semantics: Contract-2 weak, Contract-1 none. Thin evidence.
- requirement-clarity: Contract-1 weak, Contract-2 none. Thin evidence.

## Area playbooks

Check the listed facts in the RTL before trusting any verdict, and
harder when the router flags thin evidence.

- FIFO/queues: every writer of occupancy, data, and valid/ready. The
  exact full/empty conditions. What backpressure freezes. Whether empty
  slots blank stale data. Reset values of pointers.
- Counters: overflow and underflow behavior (saturate, wrap, or flag).
  Sampled-step sources. Commit vs hold priority. What clear-before-load
  orders. Reset values and widths.
- Arbitration: the grant condition per requester. Tree acceptance from
  leaf to root. Starvation and fairness. Grant-without-request and
  request-without-grant cases.
- Interrupts/wake: sticky vs transient pending. Enable masking.
  Edge-vs-level sensitivity. Completion-before-claim races. Rearm
  conditions. Reset behavior of pending bits.
- Access control: every write path including debug and test paths. Lock,
  privilege, role, and range checks on each. Default-deny vs
  default-allow. Reset values of permission state.
- State hold/update: the guard of every branch that touches the register.
  Simultaneous-write priority. Whether hold is explicit or implied. Reset
  and wipe behavior with competing updates.
- Requirement clarity: distinguish an ambiguous requirement from
  different implementations legitimately receiving different labels.
  For the supplied RTL and assumptions, missing policy or plausible
  competing readings must change the answer to justify insufficient
  context. Model disagreement alone means unresolved evidence.

## Thin-evidence protocol

When the router flags thin evidence (or no successful expert is rated
partial or better for the scope): consult the full-scope experts,
verify each relied-upon RTL fact yourself against the playbook,
state the best tier and its false-safe rate in the finding, and
keep the conclusion open when experts disagree. Never upgrade a
thin verdict to a confident one without simulation or proof.

Routing tests supplied with benchmark area tags check routing under
known scope. They do not measure domain discovery from arbitrary RTL.
Record per-expert results, failed or absent responses, and final Muse
arbitration separately. The first-ranked expert's score is not the
arbitrated result. A small public-benchmark smoke run establishes
integration behavior, not new independent qualification.
