# Configuration examples

For analysis inputs, see the [router state-file example](../docs/router.md#run-one-assessment)
and the [RTL requirement examples](../docs/rtl-requirements.md). These examples
demonstrate interfaces and selected behaviors, not benchmark accuracy.

These configuration files show the formats accepted by the training tools. The values and
case names are examples. Supply your own dataset definitions and training
settings before running a job.

| File | Purpose |
|---|---|
| [qualify_experts.py](qualify_experts.py) | Real inference measurement to validated routing profiles, with a new output registry |
| [security_cases.example.json](security_cases.example.json) | Dataset cases, split assignments, and mutation locations |
| [security_recipe.example.json](security_recipe.example.json) | Source sampling weights and checkpoint acceptance thresholds |
| [glimmer-security.example.json](../configs/glimmer-security.example.json) | Paths and settings for a three-GPU SiliconShield-RTL job |

Copy the files to a local directory outside the public checkout. Case IDs in
the catalog must match the source dataset. The example IDs are placeholders.
The recipe values demonstrate the format and have not been tuned for a task.

Small synthetic RTL examples are also available through the Python builders.
Their regression tests run without the experiment catalogs. See the
[SiliconShield-RTL training guide](../siliconshield/training/laya/README.md) for the loader
interfaces and [setup](../docs/setup.md) for commands.
