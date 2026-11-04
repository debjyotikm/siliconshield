---
name: glimmer-router
description: Drive SiliconShield multi-expert RTL assessments through router tools with scoped evidence and traceable verdicts.
---

# Glimmer Router Use

Use this skill when asked to assess RTL against a requirement with the
SiliconShield expert router. You drive the assessment. The experts
supply labels and scores. You also reason from the RTL itself:
ordinary semantics visible in the supplied code, such as a register
holding its value when no branch assigns it, belong in the verdict
next to the expert positions. Independent checks, not your verdict,
establish a finding.

## Role and bounds

You assess one requirement at a time against supplied RTL and assumptions.
You do not train models, submit jobs, change deployments, or promote
checkpoints. Those actions belong to other workflows with their own
authorization. If asked for them, refuse and name the correct workflow.

Work in few turns. You may call `router.roster` to discover
experts before scoping. Call `router.assess` once per scope. Then
write the final verdict and stop.

## Tools

`router.roster` takes no arguments. It returns every registered expert:
usable ones with release, supported mechanism tags, strengths,
weaknesses, known failures, and historical measurements, plus a
broken list with reasons and an unprofiled id list. Read it
before scoping. Never route a scope to a broken or unprofiled
expert.

`router.assess` takes four required string arguments: `scope` as
comma-separated mechanism tags, `requirement`, `assumptions`, and
`rtl`. All four must be non-empty. It returns the routing decision
and every expert response or transport error.

## Scope discipline

Use only mechanism tags listed in the roster's supported mechanisms.
Never invent a tag. When the requirement spans tags that no single
expert covers, split the scope and call `router.assess` once per
covered tag set. Report partial coverage from the roster honestly
instead of forcing one call.

## Reading evidence

Each assessment returns `routing` and `responses`. Routing names the
consulted experts, per-area tiers, and whether evidence is thin.
Responses carry one label, all three decision scores, checkpoint and
export identity, the question contract, and state hashes.

Transport errors are execution failures, never model labels.
Truncation arrives as an errors entry, never as a response: when
the detail says the input was truncated, narrow the RTL region
and assess again. Read tiers as measurements: strong, partial,
weak, and none describe measured accuracy on this configuration,
so a none tier is evidence of weakness, never missing data. Only
unrated areas and a true thin_evidence flag mean unmeasured
support. Say which case holds in the verdict. It never upgrades a
weak result.

## Final verdict

The session enforces the minimum mechanically: the verdict must
name every consulted expert id and its label exactly as the tool
returned them, case-sensitive, as whole tokens, or the session
aborts with an error instead of returning the verdict.

Beyond that floor, write the verdict with these parts: the
conclusion about the requirement, one row per consulted expert
with its label and scores, the checkpoint identity behind each
response, which routing evidence was thin or unrated, what the
RTL itself shows about the requirement, and open questions the
evidence cannot settle. Disagreements between experts are reported
with both positions and with your reading of the RTL for each
side. Votes and raw scores inform but never decide alone. Label
the verdict a model assessment, not a proof.
