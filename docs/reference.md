# Commands, configuration, and artifacts

[Documentation index](README.md)

Use this page to find commands, configuration fields, and output files. Run
`--help` in the appropriate environment for all supported arguments. Examples
use placeholder paths. None are deployment manifests.

## Entry points

Prefix module names below with `python -m`.

| Module | Purpose | Main inputs |
|---|---|---|
| `siliconshield.web_demo` | Open the optional local browser interface. | `--config`, `--port`, `--check` |
| `siliconshield.app` | Investigate a demo or supplied RTL. | `--case` or `--rtl --requirement`, `--output` |
| `siliconshield.experts` | Route and consult security experts from the CLI. | `assess --registry`, `--scope`, `--state-file`, `--arbitrate` |
| `siliconshield.benchmark` | List areas, score models, or sample routing. | `areas`, `run --registry --out-dir`, `router-eval --registry --out`, optional `--dataset` |
| `siliconshield.training.laya.build_source_security` | Build verified source-catalog data. | `--catalog`, `--output`, `--verification`, solver paths |
| `siliconshield.training.laya.build_security_data` | Build catalog cases. | `--sources`, `--cases-manifest`, output/evidence paths |
| `siliconshield.training.laya.security_train` | Distributed decision-model trainer. Also supports direct script launch. | `--model`, `--data`, `--output`, `--run-id` |
| `siliconshield.training.laya.security_campaign` | Training/finalization campaign configuration. | `--config` |
| `siliconshield.research_loop.execution.security_jobs` | Prepare a job. Add `--launch` to submit it. | `--profile`, `--experiment-id`, optional `--overrides`, `--launch` |
| `siliconshield.research_loop.execution.glimmer_operator` | Bounded live Glimmer operator. | `--config` |
| `siliconshield.research_loop.execution.strategy` | Configured fitting/development sequence. | `--config` |
| `siliconshield.research_loop.episodes.dataset` | Convert development reports into episodes. | `--runs`, `--output` |
| `siliconshield.research_loop.episodes.operator_curriculum` | Build catalog-defined operator exercises. | `--catalog`, `--output`, optional `--replay-train` |
| `siliconshield.research_loop.episodes.operator_baseline` | Build or replay operator checks. | `--build` or `--episodes`, `--output`, endpoint or `--fixture` |
| `siliconshield.research_loop.controller.evaluate` | Optional existing-adapter comparison, inference only. | model/adapter/episodes/output, optional `--final` |
| `siliconshield.research_loop.controller.serve` | Transformers reference server. | model/name, host/port/context, optional external adapter |
| `siliconshield.research_loop.controller.serve_fast` | Native vLLM launcher. | model, optional draft or external adapters, `--dry-run` |
| `siliconshield.research_loop.procedures.glimmer_router` | Local Glimmer session over roster and assessment tools. | `--registry`, `--endpoint`, `--model`, `--task-file`, `--output`, optional `--ssh-host` |
| `siliconshield.research_loop.procedures.muse_harness2` | Muse procedure trial or queue worker. | `--episodes` or `--queue`, `--checks`, `--output`, `--bundle` |
| `siliconshield.research_loop.procedures.harness2_adapter` | Glimmer procedure trials. | episodes/endpoint/model/output/adapter |

`laya_campaign`, `campaign_supervisor`, `dataset_versions`, `checker_adapter`,
`proposer_ingest`, `decision_study`, and `study_contract` are Python
integration APIs, not equivalent launch CLIs.
`owned_job` is the operator's execution worker, not an alternative way to bypass
preparation. See [campaigns](campaigns.md).

## Decision and worker contracts

The three labels are fixed: `satisfies_requirement`, `violates_requirement`,
and `insufficient_context`. They classify one supplied requirement. The last
label describes answer-changing missing information. Transport failure,
truncation, low confidence, and unsupported scope are separate conditions.

| Boundary | Specification | Implementation owner |
|---|---|---|
| State text | `REQUIREMENT`, optional `ASSUMPTIONS`, then `RTL` as exact full-line headers. Requirement and RTL are nonempty. Arbitration validates these sections. | [identity.py](../siliconshield/experts/identity.py) |
| Request transport | Local subprocess pipes, protocol `1`. Requests are `load`, `analyze`, or `shutdown`. Replies are `progress`, `ready`, `result`, or `error`. | [contracts.py](../siliconshield/experts/contracts.py), [worker.py](../siliconshield/experts/worker.py) |
| Score validation | Three required finite scores in [0, 1], total within `1e-3` of one, label equal to the maximum. Ties use satisfies, violates, context order. These checks do not calibrate confidence. | `contracts.validate_result` |
| Coverage | Explicit `truncated: false`, `state_tokens_dropped: 0`, and empty `truncated_questions`. Contradictory usage fields refuse the result. | `contracts.validate_result` |
| Identity | Weights, full export digest, question bytes/version, runtime Python tree, precision/backend, Torch version, device type, batch size, option order, token budgets, and length sorting. Each result also binds its input and worker session. | [identity.py](../siliconshield/experts/identity.py), `worker.py` |
| Deadlines | Default load deadline 180 seconds and analysis deadline 60 seconds. Progress does not reset a deadline. Calls to one worker are serialized. | `contracts.py`, `worker.py` |

The standard release budgets are 1,536 model tokens and 256 head tokens, as
recorded by each export. They are not character or source-line limits. A longer
file needs explicitly selected excerpts or modules. A passing excerpt does not
establish the status of the entire file.

## Assessment and error contract

| Record | Required interpretation |
|---|---|
| `Assessment` | Request, input, and scope hashes, planned/effective routing, successful responses, separate errors, elapsed time. |
| `ExpertResponse` | Expert/release identity, label, all three scores, input hash, question contract, observed execution identity, complete coverage. |
| `ExpertError` | Expert id, stage (`route`, `transport`, or `validate`), and error detail. It is never a model label. |
| Routing result | Consulted ids, tiers, partial coverage, unsupported tags, rationale, and `thin_evidence`. Effective routing excludes failed replies. |
| Arbitration / finding | Same assessment and requirement binding, model conclusion, explanation, dissent/open questions, and optional issue/relevance/RTL locations. A finding is marked `model_assessment`. |

The separate Glimmer router bridge returns a conversation history. Its tool
results use the expert assessment contract, but its final prose does not use
the structured arbitration validator. Keep these two agent paths distinct.

An expert CLI exit of zero means a JSON report was produced. Inspect
`assessment.errors`, effective coverage, and `arbitration_error` before using
its conclusion. The application report has a different evidence contract,
including requirement checks and candidate repair acceptance. See
[router](router.md), [repair report contract](#repair-report-contract), and
[response types](../siliconshield/experts/responses.py).

The runnable onboarding examples are maintained in
[decision-model qualification](expert-qualification.md#register-and-measure-the-published-models),
[study assembly](study-walkthrough.md), and
[local Glimmer operation](decision-model-training-with-glimmer.md). This reference owns field
and interface semantics, rather than repeating those procedures.

### Pre-submit data validation

`python -m siliconshield.training.laya.validate_study_inputs` runs CPU input and
sampler checks from the training environment. Its required input is `--data`.
Use `--question-contract`, `--pools`, `--recipe`, `--seed`, `--global-batch`,
`--world`, `--max-updates` and `--output-receipt` to bind the registered plan.
This is not a GPU launch or semantic-review approval. The complete command is
in [local training](decision-model-training-with-glimmer.md#validate-the-data-on-cpu).

## Configuration boundaries

| Input | Owner and meaning |
|---|---|
| Security job profile | Interpreter, frozen source/runtime, parent model, data, run root, GPUs, defaults, shared resource root. |
| Security recipe | Source weights, acceptance settings, and optional campaign selection reference or preservation teacher. |
| Development-views file | Population selectors (including exact `row_ids`), single-view fresh studies, triplet scoring, selection mode, and protected objectives. |
| Pool manifest | TRAIN row-to-pool assignments, pool masses, and per-pool `cycle`/`flat` modes. Separate from provenance. |
| Question contract | Pinned `v0`/`v1` question bytes validated on every training row. |
| Review ledger | JSONL English-to-assertion reviews enforcing template coverage. |
| Review packets | Packet mapping binding each approved template to its reviewed assertion bytes. |
| Reporting contract | Locally reviewed reporting requirements, bound by content hash. |
| New-study manifest | Parent/split identities and retired evaluation populations. |
| Decision-study manifest | Frozen parent, data, code, recipe, views, budget, launch command, and qualification receipts. |
| Glimmer operator config | Endpoint, model identity, one experiment, poll budget, and intervention budget. |
| Expert registry | Model id, interpreter, runtime, immutable export files, device, optional profile and release/qualification references. |
| Capability profile | Domain claims, known failures, and measurements. Matching `routing_evidence` plus observed identity determines tiers. |
| Supervisor config | Muse binary/model/workspace, bounded session settings, optional procedure queue/bundle. |

The profile examples in [configs](../configs/) are illustrative. Real values,
paths, thresholds, and budgets remain outside Git. A method accepting a config
object does not mean an arbitrary JSON file is a valid deployment configuration.

## Environment variables

| Variable | Consumer / purpose |
|---|---|
| `LAYA_ROOT` | Compatible external decision-model runtime checkout. |
| `SILICONSHIELD_SECURITY_PYTHON` | Interpreter used by the application assessment worker. |
| `SILICONSHIELD_SECURITY_CHECKPOINT` | Selected security-model export used by the application. |
| `SILICONSHIELD_LAYA_PYTHON`, `SILICONSHIELD_LAYA_CHECKPOINT` | Legacy aliases, lower priority than the security names. |
| `SILICONSHIELD_SECURITY_CASES` | Local catalog when not supplied explicitly. |
| `SILICONSHIELD_EXTERNAL_CASES_MODULE` | Trusted external case-provider Python module. |
| `SILICONSHIELD_SEMANTIC_RECIPE_MODULE` | Trusted local semantic-generation Python module. |
| `SILICONSHIELD_SECURITY_RECIPE` | Local security-training recipe. |
| `SILICONSHIELD_DATASET_KEY` | Private key file for authenticated Glimmer episode splits. |
| `SILICONSHIELD_EXPECTED_CODE_IDENTITY` | Expected Glimmer execution source identity. |
| `SILICONSHIELD_IMAGE_DIGEST` | Resolved Glimmer execution image identity. |
| `SILICONSHIELD_MODEL_FINGERPRINT` | Expected base snapshot fingerprint when configured. |
| `SILICONSHIELD_DRAFT_FINGERPRINT` | Expected speculative assistant identity when configured. |
| `SILICONSHIELD_GPU_LOCK` | Shared Glimmer serving/evaluation lease file. |
| `SILICONSHIELD_EXECUTION_MANIFEST` | Set by the owned job wrapper for the registered process. |

## Artifact ownership

| Artifact | Meaning |
|---|---|
| Application `report.json` / `report.md` | Inputs, assessment, checks, patch, and conclusion for one investigation. |
| Dataset `versions/<version>/manifest.json` | Pinned split paths/hashes/counts and optional pools. |
| Dataset `accepted.json` | Lineage, canonical, concentration, and release-class audit plus manifest binding. Not model acceptance. |
| Export `question_contract.json` | Pinned question version, digest, and bytes for inference. |
| Ingestion `current.json` | Pointer to the verified published proposal generation. |
| Run `run-manifest.json` | Operator identity, state, budget, attempts, and history. |
| Run `execution.json` | Owned process start/terminal evidence. |
| Run `status.json` | Progress snapshot, not the complete acceptance decision. |
| Security run `latest.pt` | Resumable model/optimizer/progress/sampler state. |
| Security run `eval_snapshots/` | Retained tensors for every evaluated checkpoint (rank 0). |
| Selected security export | Checkpoint chosen under development rules, potentially an earlier step. |
| Campaign `incumbent.json` | Accepted research checkpoint plus bound evidence. |
| Round receipt and decision intent | Frozen inputs, transitions, and recoverable promotion transaction. |
| Supervisor `handoff.json` | Pending proposal, observations, active round, and stop/park reasons. |
| Study `study.json` plus probe and resource receipts | Frozen plan with measured fitting and resource qualification. |
| Study `budget.json` | Durable round and time reservations. |
| Study `new-model-proposal.json` | Parked specialist proposal after budget exhaustion. |
| Expert registry and profiles | Pinned checkpoints with measured capability evidence. |
| Benchmark `comparison.json` and router-eval report | Model comparison and sampled routing outcomes. |
| Glimmer replay output | Per-task traces, grades, and comparison identities. |
| Procedure bundle | Qualified prompt advice with identity and replay grades. |

Files named `report.json` have different owners and schemas. Do not infer their
meaning from the basename or feed an arbitrary report into episode construction.
Serialized identifiers are maintained in [schemas.py](../siliconshield/schemas.py)
and the relevant module contracts. Preserve existing values when evolving an
interface. Package version ownership is separate and lives in `pyproject.toml`.

## Study-manifest compatibility

New studies use `siliconshield-decision-study-v2`. Version 1 records are
historical evidence and must be re-frozen before using the enforced launch
path. Existing jobs without a study binding retain their legacy behavior and
do not acquire the study's additional validation by running through the same
infrastructure. Use the [walkthrough](study-walkthrough.md) for new work.

## Selection compatibility reference

The six implemented paths are:

1. Legacy transfer ranking with a retention-accuracy tolerance. The trainer
   warns when a campaign selection reference is absent.
2. A frozen campaign reference sharing improvement, false-safe, missing-context,
   pair, and family-cap gates with the campaign. `v1-transfer` and `v2-retention`
   select different post-feasibility orderings.
3. A `protected-views-v1` objective declaring protection for every view, a target
   view, and explicit count-based quality targets. This path distinguishes
   protecting prior capabilities from achieving the new task's target.
4. A `protected-views-v2-triplets` objective ranking safety first, then whole
   triplets, correct decisions, false violations, unnecessary abstention, and
   NLL. Views carry `triplets`/`triplets_correct`. Gates support
   `max_false_violation`, `max_unnecessary_abstention`, and
   `whole_triplets_min` alongside false-safe floors.
5. A `protected-views-v3-balanced` objective ranking whole triplets first with
   false-safe safeguards in feasibility and ranking: capability first, safety
   constrained.

6. A `protected-views-v4-sourced` objective retaining the v3 ordering while
   applying exact per-source floors to the protected views. Declared sources
   must match the reported source breakdown, so a strong aggregate cannot hide
   a missing or regressing source.

See the [trainer guide](security-model-training.md#select-on-the-full-task)
for current objective selection and the distinction between checkpoint ranking
and promotion gates.

## Existing-adapter compatibility

The servers also accept externally supplied adapters. Native serving uses
`--adapter NAME=PATH` and supports LoRA ranks up to 16. Each adapter receives its
own identity and must be qualified with its parser and runtime. The
`controller.evaluate` command compares an existing adapter with a base or
reference adapter without optimizer updates. These are compatibility paths,
not prerequisites for the local Glimmer workflow.

The optional `controller.evaluate` compatibility path requires an existing
external adapter. Its `--final` mode requires authenticated split membership
and cannot be combined with `--harness2-output`. Dataset identity records final
use and blocks reuse for training. Bind the external adapter and serving
configuration before spending that population. This command performs inference
comparisons, not Glimmer training.

## Repair report contract

The optional `siliconshield.app` path runs supplied Verilator checks and can
request a repair. It is separate from multi-model routing. See the
[release example](releases.md#real-muse-repair) for model and proposer setup.
Custom RTL has no inherited demonstration testbenches. Without supplied checks,
its requirement remains `requirement_not_verified`.

`requirement_verified` means required checks completed with pass/fail outcomes,
not that the design passed. `repair_accepted` records whether the candidate
passed the supplied requirement and regression checks. Inspect incomplete
checks and execution logs. A timeout or abnormal exit is an execution error,
not a model verdict or proof of a violation.

Tool subprocesses have timeouts, resource limits, an allowlisted environment,
and owned process groups. They retain the host user's file permissions and do
not provide a filesystem or network sandbox. Untrusted RTL needs deployment
isolation before compilation or simulation.
