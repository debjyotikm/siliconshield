# SiliconShield: secure silicon before it ships

[Documentation index](README.md) · [Demo guide](demo.md) ·
[Architecture](architecture.md)

## The problem, in one design

The fastest way to understand SiliconShield: a register that passes
every functional test yet lets the wrong requester change protected
state. The fix is not spotting a line — it is proving the violation,
proposing a repair, and proving legitimate behavior still works.

A configuration register accepts writes. Functional tests confirm that writes
work, but the implementation forgets to check who is writing. An unprivileged
request can change state that should be protected.

Fixing this requires more than spotting a suspicious line. The engineer needs
the intended access policy, an executable example of the violation, a proposed
change, and evidence that legitimate writes still work afterward.

SiliconShield brings those steps into one workflow:

**State the requirement → assess the design → execute checks → propose a repair
→ recheck the requirement and regressions → inspect the evidence.**

The initial users are RTL designers and verification engineers working on local
security requirements. The current interface is a CLI and evidence reports.
It is not a whole-chip sign-off tool or an automatic replacement for a
verification environment.

## What the engineer receives

The investigation report preserves the supplied RTL and assumptions, identifies
the security-model checkpoint, records relevant statements and check outcomes,
and shows the candidate patch alongside its verification results.

The report shows what failed, what changed, what passed afterward, and what
remains unresolved. A repair that blocks all writes is not acceptable when the
regression check requires legitimate writes to keep working.

Engineers can review model advice, RTL edits, and verification logs in one
place. We have not measured time savings. A user study should compare the time
needed to review a repair, false alarms, missed violations, and reviewer effort
on unfamiliar designs.

## Why use both a specialist and an agent?

**SiliconShield-RTL** is trained for a focused decision: does the implementation
satisfy the supplied requirement, violate it, or lack necessary context? Its
inputs include the requirement and assumptions, not just code. Residual adapters
specialize a LAYA foundation, which is identified in the model card.

**Meta Muse** supplies the broader agent behavior: proposing a patch from an
observed counterexample, analyzing training evidence, and preparing the next
experiment. Executable tools check a proposed patch. Campaign policy checks
whether an experiment is eligible for promotion.

A matched comparison is still needed to measure what the specialist adds to the
agent workflow. The report keeps the model's prediction visible when executable
evidence disagrees with it.

## Where Meta AI contributes

| Role | Implemented connection | Evidence to show |
|---|---|---|
| Repair proposer | Muse receives the visible RTL, requirement, assumptions, and executed failure evidence, then returns exact replacements. | A model-named proposal, patch diff, and unchanged checks rerun on the candidate. |
| Campaign investigator | The Muse coding agent analyzes round evidence, proposes a hypothesis and data plan, or recommends stopping. | A recorded proposal, the owned experiment it produced, and the resulting selection decision. |
| Trainable operator | Muse Glimmer supports LoRA SFT, offline DPO, structured evidence assessment, and six bounded job actions. | Matched per-action replay results with model/data/runtime identities. |

Name the model used in each demonstration. The Muse coding agent and the locally
fine-tuned Glimmer adapter have separate roles and evaluations. Label authored
repairs as authored.

Harness², from Google Research, is a separate procedure-learning integration.
It proposes prompt advice, qualifies it on replay, and publishes passing guidance
for future agent sessions. It is not Meta tooling or a model-weight update.

## The second loop: improving the specialist

A working investigation is the product-facing loop. The training campaign is
how the specialist can be improved systematically:

1. Construct source-backed or authored examples with explicit local contracts.
2. Compile and verify labels, including opposing completions for missing context.
3. Train against a frozen dataset version and recipe.
4. Evaluate new capability and retained capability separately.
5. Promote only when registered constraints pass. Otherwise keep the incumbent.
6. Ask the agent to explain the result and propose the next bounded step.

The loop records job ownership and can recover after an interruption. Selection
rules can reject a candidate that improves one score but increases false-safes
elsewhere. When a round fails, the campaign keeps its incumbent.

## What makes the implementation worth inspecting

- Requirement-conditioned judgments and verified counterfactuals address whether
  the model uses the policy as well as the RTL.
- Repair acceptance requires executable requirement and regression checks.
- Data versions, checkpoint identities, and evaluation receipts make decisions
  traceable to their actual inputs.
- The controller is evaluated on evidence, arithmetic, action choice, and final
  consistency against the recorded evidence.
- The campaign can reject a trained model or stop after failed hypotheses.

The foundation model, upstream datasets, and verification engines are credited
in [NOTICE.md](../NOTICE.md), the [model card](siliconshield-rtl.md), and the
[data guide](data-pipeline.md). SiliconShield's contribution is the specialized
training and the connected investigation, verification, and operating workflow.

## Review the project quickly

| Review dimension | What to inspect |
|---|---|
| Impact | The engineer's before/after task and the unauthorized-write example above. |
| Meta AI use | A real Muse proposal and its verification. A separately qualified Glimmer operator comparison where available. |
| Technical depth | Solver-backed data, protected checkpoint selection, owned jobs, and recovery behavior. |
| Demo | The short investigation sequence in [demo.md](demo.md), ending with concrete evidence. |

The repository includes the implementation and an authored demonstration.
Campaign results, trained checkpoints, and live-Muse benchmarks need a separate
release review. The current product is a CLI with evidence reports. Its
development scores apply to the evaluated tasks, not semiconductor security in
general.
