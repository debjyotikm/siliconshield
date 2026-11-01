# SiliconShield documentation

SiliconShield combines requirement-conditioned RTL decisions, executable
investigation, and agent-driven training. This guide describes the checked-in
implementation. Model weights, experiment catalogs, run results, and deployment
configuration are separate artifacts supplied by the operator.

## Review the product first

The [project brief](project.md) explains who SiliconShield is for, what it does,
and where Meta AI is used. The [demo guide](demo.md) walks through an
investigation and explains how to present an authored example or a live model
run.

## Start here

| Goal | Read |
|---|---|
| Understand the components and controller boundaries | [Architecture](architecture.md) |
| Run an authored RTL investigation | [Investigation walkthrough](investigation.md) |
| Install the right environment | [Setup](setup.md) |
| Build and qualify security training data | [Data pipeline](data-pipeline.md) |
| Train or continue the decision model | [Security-model training](security-model-training.md) |
| Wire an unattended campaign | [Campaigns and controllers](campaigns.md) |
| Fine-tune and qualify Muse Glimmer | [Glimmer training](glimmer.md) |
| Refine agent instructions with Harness² | [Muse procedure learning](muse-procedures.md) |
| Serve models and measure inference | [Inference](inference.md) |
| Interpret results and reserve final evaluation | [Evaluation](evaluation.md) |
| Understand measured model strengths and limitations | [Model limitations](LIMITATIONS.md) |
| Diagnose jobs, recover, or stop a campaign | [Operations](operations.md) |
| Find commands, settings, and artifact owners | [Reference](reference.md) |
| Develop and verify changes | [Development and testing](development.md) |
| Understand the security-model release scope | [Model-family card](siliconshield-rtl.md) |
| Reproduce the Contract-1/2 training recipes | [Training recipes](training-recipes.md) |
| Plan a reproducible model study | [Decision-model studies](decision-study.md) |

## Choose the workflow

**Investigating RTL:** start with the walkthrough. A model assessment is one
input to the investigation. Executed checks establish the report's evidence. The
shipped application does not generate a complete verification environment for
arbitrary designs.

**Training SiliconShield-RTL:** qualify a dataset, register the development
selection rules, and run the distributed trainer. Validate the selected export
before treating it as a deployable artifact. Calibration and final evaluation
are separate from checkpoint selection.

**Operating experiments:** choose the Muse coding-agent supervisor or the
bounded Glimmer operator. They are separate integrations. The supervisor
supports successive rounds through an application-supplied driver. The bounded
operator handles one authorized experiment and its permitted interventions.

**Improving the controller:** Glimmer SFT/DPO updates adapter weights. Harness²
updates prompt procedures. Both require replay qualification. Neither implicitly
deploys a new controller or grants additional execution authority.

## Reading conventions

Paths beginning `/path/to/` are placeholders. Run commands from the repository
root unless a guide says otherwise. Keep outputs outside that checkout.
Commands that launch a trainer, model server, or agent are instructions for an
explicitly configured deployment, not part of the lightweight test gate.

Each guide states what its checks establish. A fixture test checks code
behavior, while a model evaluation measures predictions on a specified
population. Promotion and deployment have their own acceptance rules.

## Availability and remaining integration work

| Interface | Available here | Still supplied by a deployment |
|---|---|---|
| Investigation CLI | Authored examples, report generation, repair checks | Custom design checks and trained model export |
| Decision-model trainer | Distributed training, selection, resume, export | Compatible pinned runtime, qualified data/recipe, GPU environment |
| Muse campaign supervisor | Round state machine and agent-session integration | Campaign initialization, evaluator callback, service/polling driver |
| Glimmer controller | LoRA/DPO, replay grading, bounded job operator | Signed curriculum, qualified adapter and serving configuration |
| Native inference | vLLM command builder and lease/identity checks | Compatible vLLM build and deployment image |

The model card names the current release checkpoint,
SiliconShield-RTL Contract-1, and its provenance. Weights are published at
[siliconshield-rtl-contract-1](https://huggingface.co/danieltombay/siliconshield-rtl-contract-1),
alongside the FIFO/counter specialist
[siliconshield-rtl-contract-2](https://huggingface.co/danieltombay/siliconshield-rtl-contract-2),
the [RTL benchmark](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-benchmark),
and the [combined training set](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-training).
Accuracy and throughput must be measured for that checkpoint and runtime.

## Terms used in these guides

- **Requirement:** the specific behavior assessed, together with its assumptions.
- **False-safe:** a violation classified as satisfying the requirement.
- **Lineage / split group:** related designs or experiments kept together when
  partitioning data. Count these groups when estimating independent sample size.
- **Selected checkpoint:** weights chosen within one run under its frozen rules.
- **Incumbent:** the model currently retained by the campaign's promotion policy.
- **Research promotion:** acceptance as the next research incumbent.
- **Deployment acceptance:** a separate set of suitability thresholds. It does
  not by itself publish or serve the model.
- **Receipt:** structured evidence about inputs, execution, identity, or a decision.
- **Replay:** execution of a declared controller task through controlled tools.
  Fixture replay and live-model replay measure different things.
