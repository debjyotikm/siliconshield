# SiliconShield-RTL training modules

This package implements data construction, decision-model training, checkpoint
selection, and dataset admission. LAYA is the external model foundation and
runtime, supplied through `LAYA_ROOT` or a compatible installed package.

| Work | Modules | Canonical guide |
|---|---|---|
| Build and verify data | `build_*`, `source_contracts`, `checker_adapter`, `proposer_ingest` | [Data pipeline](../../../docs/data-pipeline.md) |
| Admit datasets and qualify exposure | `dataset_versions`, `review_ledger`, `coverage_audit`, `inventory` | [Data admission](../../../docs/data-pipeline.md#dataset-versions-and-acceptance) |
| Train and select checkpoints | `security_train`, `development_views`, `checkpoint_selection`, `replay_preservation`, `ddp_guard` | [Trainer](../../../docs/security-model-training.md) |
| Register a reproducible experiment | `decision_study`, `new_study` | [Decision studies](../../../docs/decision-study.md) |
| Report and finalize | `security_reporting`, `security_campaign`, `security_continuation` | [Evaluation](../../../docs/evaluation.md) |

Use [campaigns](../../../docs/campaigns.md) for owned jobs and supervision.
The [release recipes](../../../docs/training-recipes.md) document the historical
populations and objectives for the Contract-1 and Contract-2 decision models.
Local catalogs, recipes, data, and output artifacts stay outside this source tree.

Inspect the trainer's arguments from the prepared runtime environment:

```bash
python siliconshield/training/laya/security_train.py --help
```
