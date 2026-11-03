# Demonstrate SiliconShield

[Documentation index](README.md) · [System architecture](architecture.md) ·
[Router](router.md) · [Decision studies](decision-study.md)

Show two capabilities: Muse investigates RTL through the specialist router,
and the training infrastructure builds or improves those specialists under
reviewable constraints. Executable repair checks are a separate demonstration
of how a proposed finding can be followed up.

## Browser interface

Use the [browser demo](web-demo.md) to present the same infrastructure in Chrome.
The first two views expose Contract-1 and Contract-2 with editable RTL and
requirements. The third runs the capability router with optional local Glimmer
arbitration. The fourth follows a bounded toy training exercise through actual
operator receipts. Model loading supports CPU or a configured GPU. Training
requires separate, explicit configuration.

The examples are authored presentation cases. Show the returned class scores,
including incorrect predictions. The training view establishes workflow
operation, not an improvement in model quality.

## Three-minute CLI demonstration

This submission path uses public synthetic RTL and explicit commands. Prepare
models using [the release guide](releases.md), and record genuine execution
before editing a presentation. Keep the model and tool output exactly as
returned, including any failure.

| Time | Show | Explain |
|---|---|---|
| 0:00–0:20 | RTL, requirement, assumptions, and proposed scope | Distinguish a functional error from a protection-property failure. |
| 0:20–0:50 | Actual router assessment and expert identities | Show why each expert was consulted and its measured domain support. |
| 0:50–1:20 | Actual Muse arbitration and cited RTL | Explain the conclusion, disagreements, and unresolved questions. |
| 1:20–1:45 | Saved assessment, errors, and coverage | A model finding is provisional. Show a separate executed check if recorded. |
| 1:45–2:20 | One study manifest and training-round diagnosis | Show accepted data, frozen settings, delivered exposure, and quality gates. |
| 2:20–2:40 | Recorded promote-or-preserve decision | A failed improvement keeps the incumbent. Explain when to start a new specialist. |
| 2:40–3:00 | Released models, scoped results, and reproducible commands | C1 qualification and shift, C2 development status, and the current local Glimmer bridge. |

The time allocation is presentation pacing, not a latency promise. Label edited
or prerecorded sections. Do not replace a failed live output with a reference
answer. The authored smoke test and the real model/Muse run are separate entry
points. Use [router commands](router.md#run-one-assessment) for the main product
walkthrough. The one-call SSH example in [release setup](releases.md) demonstrates
the separate repair path.

## Recorded evidence

The completed release rehearsal covers model loading and the separate repair
path. It does not establish that the full storyboard above has been recorded.
It downloaded both pinned public models and loaded them through
the real worker on an RTX GPU. A separate invocation combined Contract-1 with
the existing base Glimmer service for the authored unauthorized-write case.
The live proposer generated the missing privilege guard. The original
requirement check failed, the candidate passed it, and legitimate-write checks
passed for both designs. The workflow returned `demonstrated_violation_fixed`.

This was GPU execution plus a live model-generated repair and Verilator
integration on one public authored case. It is not an unseen evaluation or an
engineering time-savings study. Raw input/output and check receipts remain in
the release-preparation bundle outside Git. Public media and evidence links are
not yet published. The steps below link to the maintained execution procedures.

## Prepare the demonstration

Use [setup](setup.md) for prerequisites and [pinned releases](releases.md) for
the model/runtime combination. Rehearse the exact input and commands before
recording. Startup, compilation, and generation time depend on the environment.
Label prerecorded sections and preserve actual outputs, including failures.

The experts CLI, investigation application, and training supervisor are separate
entry points. Show those transitions explicitly.

## Expert routing and Muse arbitration

Follow the [router walkthrough](router.md#run-one-assessment). Show the supplied
RTL and requirement, requested scope, consulted expert identities, returned
scores, and any errors or coverage gaps. Then request arbitration and inspect
the cited experts, dissent, and open questions. A generated finding remains a
model assessment.

## Execute a requirement check and repair

For the optional repair segment, use the
[authored investigation](investigation.md#run-the-authored-demonstration), then
[configure a Muse proposer](investigation.md#request-a-muse-repair) for a genuine
generated patch. The [release guide's SSH example](releases.md#real-muse-repair)
connects that path to an existing remote loopback service.

Show the original failure, proposed diff, unchanged requirement check, and
legitimate-write regression. Use the
[report interpretation](investigation.md#read-the-conclusion) to explain the
outcome. A rejected repair is a valid result. An authored patch or unavailable
assessor must retain that label in the presentation.

## Close with reproducible improvement

Show a [decision study](decision-study.md) with its registered parent, accepted
data, recipe, selection policy, and budget. Present the completed round's
exposure audit, class and whole-triplet results, SDK parity, and promotion
reasons. If a protected constraint failed, explain why the incumbent stayed.
Use an explicitly labeled fixture when no shareable round is available.

Keep private receipts outside Git. A curated demo establishes the demonstrated
integration. Claims of generalization, faster engineering review, or router
superiority require the separate [evaluation protocol](evaluation.md).
