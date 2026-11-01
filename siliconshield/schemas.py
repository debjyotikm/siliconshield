"""Authoritative serialized schema identifiers.

Each constant names a versioned on-disk or on-the-wire contract. The
string values are frozen: changing a value breaks compatibility with
existing receipts, episodes, and checkpoints. Add a new identifier for a
new contract version; never edit an existing value in place.
"""

TRANSFER_CATALOG_SCHEMA = "siliconshield-transfer-catalog-v1"
TRANSFER_EPISODES_SCHEMA = "siliconshield-transfer-episodes-v1"
OPERATOR_EPISODES_SCHEMA = "siliconshield-operator-episodes-v1"
OPERATOR_BASELINE_SCHEMA = "siliconshield-operator-baseline-v1"
OPERATOR_DEV_SCHEMA = "siliconshield-operator-dev-v1"
CONTROLLER_EPISODES_SCHEMA = "glimmer-controller-episodes-v1"
REPLAY_SCHEMA = "glimmer-replay-v1"
RUN_MANIFEST_SCHEMA = "siliconshield-run-manifest-v1"
INVESTIGATION_SCHEMA = "siliconshield-investigation-v1"
NEW_STUDY_SCHEMA = "siliconshield-new-study-v1"
LAYA_CAMPAIGN_SCHEMA = "siliconshield-laya-campaign-v1"
CAMPAIGN_ROUND_SCHEMA = "siliconshield-campaign-round-v1"
PROPOSAL_SCHEMA = "siliconshield-rtl-proposal-v1"
DATASET_VERSION_SCHEMA = "siliconshield-dataset-version-v1"
DATASET_ACCEPTANCE_SCHEMA = "siliconshield-dataset-acceptance-v1"
# Strict training-readiness audit. New dataset versions must carry this
# schema id; legacy audits without it remain readable but can no longer
# back a new acceptance. Never edit an existing value.
DATASET_AUDIT_SCHEMA = "siliconshield-dataset-audit-v2"
PROCEDURE_BUNDLE_SCHEMA = "siliconshield-procedure-bundle-v1"
PROCEDURE_REQUEST_SCHEMA = "siliconshield-procedure-request-v1"
GUI_MODEL_REGISTRY_SCHEMA = "siliconshield-gui-model-registry-v1"
GUI_ANALYSIS_SCHEMA = "siliconshield-gui-analysis-v1"
GUI_SCAN_SCHEMA = "siliconshield-gui-scan-v1"

# Multi-population checkpoint protection and objective ranking.
CHECKPOINT_OBJECTIVE_SCHEMA = "protected-views-v1"
CHECKPOINT_TRIPLETS_SCHEMA = "protected-views-v2-triplets"
CHECKPOINT_BALANCED_SCHEMA = "protected-views-v3-balanced"
CHECKPOINT_SOURCED_SCHEMA = "protected-views-v4-sourced"
REPLAY_PRESERVATION_SCHEMA = "replay-teacher-v1"
REVIEW_LEDGER_SCHEMA = "siliconshield-review-ledger-v1"
INDEPENDENCE_AUDIT_SCHEMA = "siliconshield-independence-audit-v1"
EXPERT_CAPABILITY_SCHEMA = "siliconshield-expert-capability-v1"
STUDY_MANIFEST_SCHEMA = "siliconshield-decision-study-v1"
NEW_MODEL_PROPOSAL_SCHEMA = "siliconshield-new-model-proposal-v1"

STUDY_MANIFEST_V2_SCHEMA = "siliconshield-decision-study-v2"
