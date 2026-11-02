# Assemble a decision-model study

[Documentation](README.md) · [Training methods](security-model-training.md) ·
[Data qualification](data-pipeline.md) · [Decision-model training with Glimmer](decision-model-training-with-glimmer.md)

This walkthrough connects accepted data, measured baseline evidence, training
settings and the owned-job interface. Its output is a frozen study manifest and
a bound job profile. Freezing and verification run on CPU. They do not launch
training or grant dataset acceptance.

Use `improve_existing` to continue a decision model with protected retention
views. Use `create_new` for an untouched, adapter-free foundation with a verified
foundation record. The example below continues a published decision model using
residual adapters and its decision head.

## Understand a complete row and triplet

The [requirements guide](rtl-requirements.md) explains how to express the
intended behavior before constructing labels and verification properties.

A training row combines model-visible input with label and provenance metadata.
The following Python constructs an illustrative triplet without writing a file.
It uses the question-contract helper so instructions and label order match the
runtime. These fragments explain the format. They have not passed the solver,
semantic-review or acceptance process and are not a train-ready dataset.

```python
import hashlib

from siliconshield.training.laya.question_contracts import NEW_CONTRACT, question_for

requirement = "After reset, q may change only when privileged is asserted."
assumptions = (
    "Sample at rising clock edges. Reset is exempt. Inputs are unconstrained. "
    "The register and inputs have matching widths. Any omitted internal "
    "permit definition may depend on privileged."
)
assertion = (
    "if (past_valid && $past(rst_n) && rst_n && !$past(privileged)) "
    "assert(q == $past(q));"
)
branches = [
    ("else if (wr_en && privileged) q <= wdata;", "satisfies_requirement"),
    ("else if (wr_en) q <= wdata;", "violates_requirement"),
    ("else if (wr_en && permit) q <= wdata;", "insufficient_context"),
]
rows = []
for index, (branch, label) in enumerate(branches):
    rtl = "always_ff @(posedge clk) begin\n  if (!rst_n) q <= '0;\n  " + branch + "\nend"
    rows.append({
        "id": f"example-{index}",
        "state": f"REQUIREMENT\n{requirement}\nASSUMPTIONS\n{assumptions}\nRTL\n{rtl}",
        "questions": question_for(NEW_CONTRACT),
        "expected": {"security": label},
        "pair_id": "example-write-policy",
        "split_group_id": "example-register-family",
        "template_id": "example-privilege-template",
        "assertion_sha256": hashlib.sha256(assertion.encode()).hexdigest(),
        "mechanism_id": "privileged-write",
        "source": {"name": "authored-example", "family": "register-family"},
    })
```

The omitted `permit` is an internal implementation gap, not a declared
unconstrained primary input. Filling it with `privileged` or `1'b1` permits
opposing outcomes. A training release must establish that opposition with the
[fixed checker and review ledger](data-pipeline.md). Labels belong in metadata,
never in comments such as “faulty case” inside the model input.

Keep all three rows and related rewrites in one split group. Construct separate
lineages for development and reserved evaluation. Replicating or renaming this
triplet does not create independent examples.

## Prepare the input artifacts

Keep the study workspace outside Git. Start with these real, reviewed inputs:

| Artifact | Required content |
|---|---|
| Dataset version store | Immutable TRAIN/dev manifests and a valid acceptance record |
| Pools and review ledger | Accepted pool assignment and complete semantic template review |
| Parent export | Complete immutable inference export with question-contract record |
| Baseline and objectives | Integer counts from the parent on every development view, plus pre-registered target and retention limits |
| Views file | Exact population definitions, `score_triplets: true`, `selection_view`, and matching `checkpoint_objective` |
| Job profile | Interpreter, source/runtime snapshots, parent/data paths, GPUs, recipe, run/resource roots, timeout and defaults |
| Qualification receipts | Measured fitting gain, parameter-group movement and resources for this plan |

The views/objectives shapes and gate vocabulary are defined in
[evaluation](evaluation.md) and the study requirements below. Freeze
`rows`, per-class supports, correct counts, false-safes and whole-triplet counts
for each baseline view. Limits must correspond to those same populations.
Do not fill missing measurements with success-shaped fixture values.

For this adapter-continuation example, the profile defaults must include
`freeze_encoder: true`, `continue_training: true`, `sampling: grouped`,
`plateau_requires_full_exposure: true`, the question contract, learning rates,
seed, batch geometry, update/evaluation budgets, pools, review ledger and views.
The recipe's `training` mapping must equal these effective settings exactly.
Include reviewed packets when required by the trainer. Verify the proposed
schedule with the [CPU input validator](decision-model-training-with-glimmer.md#validate-the-data-on-cpu).

## Study requirements

A study freezes the training plan and acceptance criteria before launch. The
whitepaper covers the [research and experimental findings](whitepaper.md).
The following requirements govern execution:

- Use the current manifest schema, `siliconshield-decision-study-v2`.
- `improve_existing` protects established capabilities through per-view gates.
  `create_new` requires an untouched foundation and a separately reviewed
  identity record. Removing adapters does not establish an untouched model.
- Pin a source snapshot, runtime, tokenizer, question contract, accepted data,
  review ledger, sampling pools, optimization settings, hardware, command,
  selection objective, and budget. A live checkout is not a frozen snapshot.
- Keep qualification and test reserved. Their identities are bound without
  reading their rows. An unbuilt reserved split needs an explicit
  `not_materialized` declaration, not an invented hash or qualification claim.
- Stage measured fitting and resource receipts before launch. Both bind the
  study, parent, TRAIN data, and evidence artifacts. The fitting probe must
  demonstrate movement in trainable groups, none in frozen groups, and TRAIN
  fit improvement. Resource checks require finite capacity and budget values.
- Bind the same manifest to the campaign and owned-job profile. Verification
  repeats before submission, including retries. Changed inputs or selection
  policy require a reviewed amendment.

The operator reserves the full attempt timeout and stop grace in a locked
`budget.json`. Retries consume the cumulative budget. Crash reservations are
not reclaimed, and repeated submission of the same attempt is not charged twice.
Checkpoint selection, campaign promotion, and reserved evaluation remain
separate decisions.

## Freeze and bind the study

Set `SILICONSHIELD_STUDY` to the private workspace containing these inputs.
The example expects `job-profile.json`, `objectives.json`, `policy.json`, and
`environment.json`. The latter records `dataset_version`, `runtime_revision`, `devices` and
`precision` from the actual environment. The profile records `image_digest`
from that environment. Its timeout must leave room for the owned-job stop grace
inside the study budget.

This complete assembly uses measured files rather than embedded experimental
values. Use absolute paths in the profile. The source snapshot is the checkout
root containing the full `siliconshield` package, not only the trainer directory.

```python
import hashlib
import json
import os
from pathlib import Path

from siliconshield.research_loop.execution import study_contract
from siliconshield.research_loop.execution.security_jobs import command_for
from siliconshield.training.laya import decision_study
from siliconshield.training.laya.question_contracts import read_export_record

root = Path(os.environ["SILICONSHIELD_STUDY"]).resolve()
def load(path):
    return json.loads(Path(path).read_text())

def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

profile = load(root / "job-profile.json")
settings = profile["defaults"]
parent = Path(profile["model"])
contract = read_export_record(parent)["version"]
views = load(settings["development_views"])
environment = load(root / "environment.json")
experiment = "local-study-round-01"
command, output = command_for(profile, experiment, {})
# This pilot declares reserved sets unbuilt. Replace with real owner-supplied
# identities if already reserved. Never open sealed rows during assembly.
pending = {"status": "not_materialized", "reason": "Pilot only, release evaluation not constructed"}
spec = {
    "study_id": "local-study",
    "mode": "improve_existing",
    "parent": {
        "path": str(parent),
        "weights_sha256": sha(parent / "model.safetensors"),
        "adapters_present": True,
        "question_contract": contract,
    },
    "dataset": {
        "root": str(root / "datasets"),
        "version": environment["dataset_version"],
        "splits": {
            "train": {"path": str(Path(profile["data"]) / "train.jsonl")},
            "dev": {"path": str(Path(profile["data"]) / "dev.jsonl")},
            "qualification": dict(pending),
            "test": dict(pending),
        },
        "pools": settings["pools"],
        "review_ledger": settings["review_ledger"],
    },
    "code": {
        "path": profile["source"],
        "runtime": {"name": "laya-next", "revision": environment["runtime_revision"], "path": profile["laya_root"]},
        "python": profile["python"],
        "image_digest": profile["image_digest"],
        "tokenizer_path": str(parent / "tokenizer"),
        "question_contract": contract,
        "question_contract_path": str(parent / "question_contract.json"),
    },
    "training": {
        "trainables": {"mode": "adapters_only"},
        "learning_rates": {"adapter": settings["adapter_lr"], "head": settings["head_lr"]},
        "schedule": settings["lr_schedule"],
        "seed": settings["seed"],
        "batch": {
            "micro": settings["micro_batch"],
            "accumulation": settings["accumulation"],
            "devices": len(profile["gpus"]),
            "global": settings["micro_batch"] * settings["accumulation"] * len(profile["gpus"]),
        },
        "max_updates": settings["max_updates"],
        "eval_every_updates": settings["eval_every_updates"],
        "hardware": {"devices": environment["devices"], "precision": environment["precision"]},
        "settings": settings,
        "recipe": load(profile["recipe"]),
        "fitting_probe": str(root / "probe.json"),
        "resource_check": str(root / "resources.json"),
    },
    "evaluation": {
        "views": views["views"],
        "selection_objective": views["checkpoint_objective"],
        "selection_policy": load(root / "policy.json"),
        "objectives": load(root / "objectives.json"),
    },
    "budget": {"max_rounds": 1, "max_hours": (profile["timeout_seconds"] + 40) / 3600},
    "launch_command": command,
}
manifest = decision_study.freeze_study_manifest(spec)
study_path = root / "study.json"
with study_path.open("x") as stream:
    stream.write(json.dumps(manifest, indent=2) + "\n")
profile["decision_study_manifest"] = str(study_path)
profile["decision_study_sha256"] = manifest["manifest_sha256"]
study_contract.verify_job(profile, experiment, {})
with (root / "bound-job-profile.json").open("x") as stream:
    stream.write(json.dumps(profile, indent=2) + "\n")
print(decision_study.verify_study_manifest(manifest))
```

Expected output is `study_id: local-study` and `verified: true`. This confirms
input integrity and agreement between the profile and manifest. It does not
establish fitting, resource availability, or a quality improvement.

The probe/resource receipts must bind the manifest digest, parent weights and
TRAIN bytes and reference actual measured artifacts. Check them before launch:

```python
study_contract.verify_receipts(
    manifest, load(root / "probe.json"), load(root / "resources.json")
)
```

A fitting receipt records `trainable_groups`, measured `movement` for encoder,
adapter and head, and `train_fit` with `rows`, `correct_before`, `correct_after`.
A resource receipt records `requirement` and `available` mappings for `gpus`,
`memory_gb`, and `hours`. Both include `study_sha256`, `parent_sha256`,
`train_sha256` and an `artifacts` path-to-digest mapping. Their measurements
must come from the qualified probe and environment, not this walkthrough.

For a supervisor-managed campaign, store the manifest under
`campaign/studies/<study_id>/study.json` and configure that `study_id`.
The job profile binds the same file and digest. The [campaign guide](campaigns.md)
owns supervisor integration and round assessment.

## Launch and interpret the result

Pass the bound profile to the [local Glimmer operator](decision-model-training-with-glimmer.md#run-one-frozen-study-job).
Keep the experiment ID and effective settings identical to the frozen command.
An amendment or another round requires a new, reviewed binding.

At closeout, compare per-view counts and whole triplets, confirm delivered
exposure, inspect failure reasons and re-score the selected export through the
intended inference path. Preserve the parent when the candidate fails a gate.
Reserved evaluation is a separate protocol. A pilot with unbuilt reserved sets
cannot claim release qualification.
