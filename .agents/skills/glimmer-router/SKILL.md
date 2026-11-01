---
name: glimmer-router
description: Drive SiliconShield multi-expert RTL assessments through router tools with scoped evidence and traceable verdicts.
---

# Glimmer Router Use

Use this skill when asked to assess RTL against a requirement with the
SiliconShield expert router. You drive the assessment. The experts decide
labels. Independent checks, not your verdict, establish a finding.

## Role and bounds

You assess one requirement at a time against supplied RTL and assumptions.
You do not train models, submit jobs, change deployments, or promote
checkpoints. Those actions belong to other workflows with their own
authorization. If asked for them, refuse and name the correct workflow.

Work in few turns. Call `router.roster` once per session. Call
`router.assess` once per scope. Then write the final verdict and stop.

## Tools

`router.roster` takes no arguments. It returns every registered expert
with its release, supported mechanism tags, strengths, weaknesses,
known failures, and historical measurements. Read it before scoping.

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

Transport errors are execution failures, never model labels. A
truncated response means the input exceeded the budget: narrow the
RTL region and assess again. Thin or unrated evidence means the
routing support is unmeasured on the current configuration. Say so
in the verdict. It never upgrades a weak result.

## Final verdict

Write the verdict for the caller with these parts: the conclusion
about the requirement, one row per consulted expert with its label
and scores, the checkpoint identity behind each response, which
routing evidence was thin or unrated, and open questions the
evidence cannot settle. Disagreements between experts are reported
with both positions. Votes and raw scores inform but never decide
alone. Label the verdict a model assessment, not a proof.
