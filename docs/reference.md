# Commands, configuration, and artifacts

[Documentation index](README.md)

Use this page to find commands, configuration fields, and output files. Run
`--help` in the appropriate environment for all supported arguments. Examples
use placeholder paths. None are deployment manifests.

## Entry points

Prefix module names below with `python -m`.

| Module | Purpose | Main inputs |
|---|---|---|
| `siliconshield.app` | Investigate a demo or supplied RTL. | `--case` or `--rtl --requirement`, `--output` |
| `siliconshield.gui` | Native desktop analysis window (demo). | `--checkpoint`, `--python`, `--laya-root`, `--device` |
| `siliconshield.experts` | Route and consult security experts from the CLI. | `assess --registry`, `--scope`, `--state-file`, `--arbitrate` |
| `siliconshield.benchmark` | Score models on the RTL benchmark. | `run --registry`, `--dataset`, `--models`, `--areas`, `--out-dir` |
| `siliconshield.training.laya.build_source_security` | Build verified source-catalog data. | `--catalog`, `--output`, `--verification`, solver paths |
| `siliconshield.training.laya.build_security_data` | Build catalog cases. | `--sources`, `--cases-manifest`, output/evidence paths |
| `siliconshield.training.laya.security_train` | Distributed decision-model trainer. Also supports direct script launch. | `--model`, `--data`, `--output`, `--run-id` |
| `siliconshield.training.laya.security_campaign` | Training/finalization campaign configuration. | `--config` |
| `siliconshield.research_loop.execution.security_jobs` | Prepare a job. Add `--launch` to submit it. | `--profile`, `--experiment-id`, optional `--overrides`, `--launch` |
| `siliconshield.research_loop.execution.glimmer_operator` | Bounded live Glimmer operator. | `--config` |
| `siliconshield.research_loop.execution.strategy` | Configured fitting/development sequence. | `--config` |
| `siliconshield.research_loop.execution.campaign` | Watch Glimmer training and evaluate. | `--root`, `--run`, container/image/key options |
| `siliconshield.research_loop.episodes.dataset` | Convert development reports into episodes. | `--runs`, `--output` |
| `siliconshield.research_loop.episodes.operator_curriculum` | Build catalog-defined operator exercises. | `--catalog`, `--output`, optional `--replay-train` |
| `siliconshield.research_loop.episodes.operator_baseline` | Build or replay operator checks. | `--build` or `--episodes`, `--output`, endpoint or `--fixture` |
| `siliconshield.training.glimmer.train` | LoRA supervised training. | model/data/output, purpose, task-selection options |
| `siliconshield.training.glimmer.dpo` | Frozen reference log-probabilities or preference training. | `--mode reference\|train`, model/adapter/pairs/output |
| `siliconshield.research_loop.controller.evaluate` | Matched in-process controller comparison. | model/adapter/episodes/output, optional `--final` |
| `siliconshield.research_loop.controller.serve` | Transformers reference server. | model/adapter/name, host/port/context |
| `siliconshield.research_loop.controller.serve_fast` | Native vLLM launcher. | model, named adapters, optional draft, `--dry-run` |
| `siliconshield.research_loop.procedures.muse_harness2` | Muse procedure trial or queue worker. | `--episodes` or `--queue`, `--checks`, `--output`, `--bundle` |
| `siliconshield.research_loop.procedures.harness2_adapter` | Glimmer procedure trials. | episodes/endpoint/model/output/adapter |

`laya_campaign`, `campaign_supervisor`, `dataset_versions`, `checker_adapter`,
and `proposer_ingest` are Python integration APIs, not equivalent launch CLIs.
`owned_job` is the operator's execution worker, not an alternative way to bypass
preparation. See [campaigns](campaigns.md).

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
| Glimmer operator config | Endpoint, adapter identity, one experiment, poll budget, and intervention budget. |
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
| `SILICONSHIELD_GPU_LOCK` | Shared Glimmer training/serving/evaluation lease file. |
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
| Glimmer `latest` / `latest.previous` | Current and fallback adapter/resume generations. |
| Glimmer replay output | Per-task traces, grades, and comparison identities. |
| Procedure bundle | Qualified prompt advice with identity and replay grades. |

Files named `report.json` have different owners and schemas. Do not infer their
meaning from the basename or feed an arbitrary report into episode construction.
Serialized identifiers are maintained in [schemas.py](../siliconshield/schemas.py)
and the relevant module contracts. Preserve existing values when evolving an
interface. Package version ownership is separate and lives in `pyproject.toml`.
