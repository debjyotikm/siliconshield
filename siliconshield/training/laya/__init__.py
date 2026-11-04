"""SiliconShield-RTL security-training script suite (LAYA runtime).

Run entry points with ``python -m`` from the repository root, for example:
``python -m siliconshield.training.laya.security_train --help``.
Sibling modules are imported by package path. A few data builders also
tolerate bare-script loading with the training directory on ``sys.path``
for test harnesses; that fallback is legacy, not the supported path.
"""

# Non-meta local decision models are not supported. Use at own risk.
# This package trains and serves the legacy local-worker path, which the
# demo and specialists refuse by default. The supported product path calls
# Meta models only: Muse Glimmer specialists with Contract skills, and
# Muse Spark final assessment. Only question_contracts is imported by
# the supported path, for its versioned contract constants.
