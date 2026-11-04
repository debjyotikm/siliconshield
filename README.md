<div align="center">
  <img src="docs/assets/siliconshield-banner.png" alt="SiliconShield: Secure silicon before it ships." width="1200">

  <p><strong>RTL bug investigation. Specialist decision models. Reproducible training.</strong></p>

  <p>
    <a href="https://github.com/debjyotikm/siliconshield/actions/workflows/checks.yml"><img src="https://github.com/debjyotikm/siliconshield/actions/workflows/checks.yml/badge.svg" alt="Repository checks"></a>
    <a href="docs/setup.md"><img src="https://img.shields.io/badge/Python-3.11%2B-444444?style=flat-square" alt="Python 3.11 or newer"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/Code_license-Apache--2.0-444444?style=flat-square" alt="Code license: Apache-2.0"></a>
  </p>

  <p>
    <a href="#get-started">Get started</a> ·
    <a href="#decision-models">Decision models</a> ·
    <a href="docs/study-walkthrough.md">Train a specialist</a> ·
    <a href="docs/README.md">Engineering documentation</a>
  </p>
</div>

Every AI solution depends on the chips - processors, accelerators, and memory
systems beneath it. SiliconShield brings AI investigation to the hardware that
powers AI. It evaluates RTL against explicit requirements through a capability
router that consults locally trained specialist decision models. Meta Muse
interprets their results against the implementation, producing a finding with
supporting evidence and open questions. The goal is to help hardware engineers
focus their effort where it matters.

The platform provides the data, training, evaluation, and recovery
infrastructure to improve an existing specialist or develop one for another
design domain. Model development and investigation share the same principle: an
apparently convincing answer must remain distinguishable from an independently
checked result.

Model conclusions guide RTL review and investigation. Simulation and repair
checks run through a separate application that records what was executed.
The [research white paper](docs/whitepaper.md) traces the models, datasets,
training decisions, and measured failure modes behind the system.

## Three Ways to Use SiliconShield

1. **Investigate RTL with Muse Glimmer and specialist decision models.** Assess
   a design against its functional or security requirement. The router selects
   relevant specialists, and Glimmer examines their predictions alongside the
   RTL to explain suspected mismatches, disagreements, and missing context.

2. **Use the decision models directly.** Integrate Contract-1 and Contract-2
   into existing engineering tools through the CLI and Python interfaces.
   Obtain structured verdicts, scores, and model identity without an agent in
   the inference path.

3. **Build and improve decision models.** Train a new domain specialist or
   continue an existing model using qualified datasets, reproducible studies,
   explicit evaluation criteria, and recoverable GPU jobs.

## Decision Models

**Give agents specialist RTL judgment they can call as a tool.** SiliconShield’s
decision models assess a design against an explicit requirement and return a
structured result: **satisfies the requirement**, **violates the requirement**,
or **insufficient context**. Muse uses the router to consult the relevant
specialist, interpret its assessment, and focus the investigation on the logic
that matters. The models run locally and expose a common interface, so new
specialists can extend the workflow to additional design domains.
Both models were created with the help of Muse Wonder and Muse Spark 1.3
through training studies on local GPU hardware.

### Contract-1 · Local guards and register protection

Contract-1 is a decision model for the control logic that protects register and
state updates. Its strength is assessing **visible privilege checks, lock
guards, clear enables, and address qualification** in a self-contained RTL
region.

It helps agents investigate questions such as: *Can an unprivileged request
change this register? Does a clear operation require authorization? Can an
update occur outside the permitted address range?* By evaluating the
implementation together with the requirement, Contract-1 gives the agent a
focused assessment of whether the relevant guard enforces the intended policy.

### Contract-2 · FIFO, counter, and arbitration behavior

Contract-2 is a decision model for **focused state-transition requirements** in
FIFOs, counters, and arbitration logic. It extends the workflow from local
access guards to questions about how control state responds to requests,
boundaries, and competing operations.

Its strongest demonstrated areas include FIFO occupancy control and counter
behavior. It helps agents examine questions such as: *Does a push while full
preserve occupancy? Does a saturating counter stop at its limit?* Focused
arbitration support adds assessment of grant and handshake behavior. Each
question is evaluated against the supplied RTL and its stated operating
assumptions.

[Explore model capabilities and evaluation](docs/siliconshield-rtl.md) ·
[Known limitations](docs/LIMITATIONS.md) ·
[Download and run locally](docs/releases.md)

## RTL Investigation and Model Development

SiliconShield brings specialist models and Muse into two practical workflows:
investigating RTL against a requirement, and building better decision models
for the designs engineers need to review.

![Two separate workflows: RTL and a requirement pass through the router, decision models, and Muse to a finding with next checks. Reviewed training data produces a candidate that is evaluated for promotion or retention, with Muse proposing further experiments within the study budget.](docs/diagrams/product-workflows.svg)

[System architecture](docs/architecture.md)

### Investigate a design with Muse

**Start with the behavior you need to protect.** Supply an RTL region and a
requirement, such as preventing unauthorized register writes or preserving
FIFO occupancy under backpressure. The router consults decision models with
relevant domain coverage.

Muse interprets their assessments alongside the design and requirement,
identifies relevant logic, and explains disagreements or missing information.
The result is a structured finding that helps an engineer decide **where to
look, which requirement is at risk, and what to check next**.

[Explore the investigation workflow](docs/router.md)

### Build decision models for your design domain

**Extend the analysis as your hardware changes.** SiliconShield provides
reproducible training infrastructure to improve an existing decision model or
develop a specialist for another class of RTL behavior.

Muse proposes training cases and experiments. Solver checks and semantic review
establish the data, while the study fixes the training recipe, evaluation
criteria, and compute budget. Each candidate is measured against explicit
improvement and retention requirements before promotion.

Completed experiments feed the next decision: refine the training strategy,
develop a new specialist, or retain the existing model when the evidence does
not support an improvement. This gives teams a controlled path from an
observed model weakness to a measured capability gain.

[Explore decision-model training](docs/study-walkthrough.md)

## Specialist Routing and Muse Analysis

**Match each RTL question to the right decision model.** A register-access
requirement and a FIFO-control requirement call for different expertise.
SiliconShield uses the declared design domain and measured model capabilities
to select relevant specialists, bringing their assessments into a common
investigation workflow.

Muse interprets those assessments against the RTL and the stated requirement.
The [RTL analysis skill](.agents/skills/rtl-analysis/SKILL.md) guides its
attention to register writers, branch guards, competing updates, and hidden
dependencies. It also distinguishes functional defects from security failures
by identifying the intended behavior or protection property at risk.

Each result stays linked to the assessed input and the model that produced it.
Expert disagreements, gaps in model coverage, and failed calls remain visible,
so engineers can understand what supports a finding and what still needs
investigation. Muse brings this evidence together into a traceable assessment
with relevant RTL references and next checks.

[Explore the router architecture](docs/router.md) ·
[Inspect the RTL analysis skill](.agents/skills/rtl-analysis/SKILL.md)

## Train Decision Models Locally with Muse Glimmer

**Build and improve RTL decision models on your own hardware.** SiliconShield
supports a fully local training workflow with Muse Glimmer as the operator.
Model serving, GPU training, evaluation, and orchestration stay within your
infrastructure, keeping training data, weights, and experiment records under
your control.

### From a prepared dataset to an evaluated candidate

Start with a dataset and a study that defines the model, training recipe,
compute budget, and acceptance criteria. SiliconShield validates the inputs,
checks that the model can fit the training examples, and runs development
training under the registered settings.

Locally served Muse Glimmer reads progress and training evidence, then proposes
operational actions through controlled tools. SiliconShield manages job
execution, checkpoints, resource ownership, and completion. Persistent state
supports recovery after interruptions, while recorded actions and results make
each run inspectable.

### Improvement measured against the capabilities that matter

Each study defines what must improve and what must be preserved. Evaluation
considers violation detection, missing-context handling, consistency across
related examples, and performance on protected capabilities. A candidate
advances only when it satisfies the study’s promotion criteria. Otherwise,
the existing model is retained and the experiment provides evidence for the
next training decision.

### Demonstrated on local hardware

The workflow has been exercised with **Muse Glimmer serving on NVIDIA DGX
Spark** and **decision-model training on NVIDIA RTX PRO 4000**. Documented runs
include a complete small-scale training and reload-verification cycle,
followed by a real-scale specialist continuation. Both ran without human turns
inside the configured loop. The specialist continuation retained its parent
when no candidate met all acceptance criteria.

Decision models build on the LAYA foundation and compatible LAYA Next runtime.
SiliconShield supplies the RTL training task, data qualification, reproducible
study infrastructure, and evaluation controls.

[Explore the Glimmer-operated decision-model training case study](docs/case-study-decision-model-training-with-glimmer.md) ·
[Run training with local Glimmer](docs/decision-model-training-with-glimmer.md) ·
[Design a study](docs/study-walkthrough.md) ·
[Prepare training data](docs/data-pipeline.md) ·
[Training recipes](docs/training-recipes.md)

## Get started

Complete [model and runtime setup](docs/setup.md), including the measured
registry, then activate the controller environment from the repository root:

```bash
source .venv-controller/bin/activate
export REGISTRY="$SILICONSHIELD_REGISTRY"
```

The first two examples call one decision model directly. The last two use the
complete infrastructure including the decision models and ask Muse to interpret
their predictions alongside the RTL.

### 1. Contract-1: register privilege protection

[The design](examples/rtl/privilege-register.sv) updates `cfg_reg` only when
both the write enable and privilege input are asserted. Reset is exempt.

```bash
python -m siliconshield.experts assess \
  --registry "$REGISTRY" --expert contract-1 \
  --design examples/rtl/privilege-register.sv \
  --requirement "Outside reset, cfg_reg may change only when privileged is asserted." \
  --assumptions "Two-state synchronous logic. Inputs are defined and may vary independently. Reset is exempt from the requirement." \
  --brief
```

### 2. Contract-2: simultaneous push and pop

[The design](examples/rtl/push-pop-count.sv) increments `count` when push and
pop are both asserted. This conflicts with the requirement that simultaneous
requests leave the count unchanged.

```bash
python -m siliconshield.experts assess \
  --registry "$REGISTRY" --expert contract-2 \
  --design examples/rtl/push-pop-count.sv \
  --requirement "Outside reset, when push and pop are both asserted, count must remain unchanged." \
  --assumptions "Two-state synchronous logic sampled at the rising clock edge. Reset is exempt from the requirement. Inputs may vary independently." \
  --brief
```

Both commands print the prediction, all three scores, and load/inference time.

### 3. Investigate with local Muse Glimmer

Both Muse examples use [this design](examples/rtl/push-pop-hold.sv), where
simultaneous push and pop take the default branch and leave `count` unchanged.

Start Glimmer using [local deployment](docs/inference.md). Set
`SILICONSHIELD_MUSE_ENDPOINT` to its loopback chat endpoint. This command uses
the served model name `glimmer-base`. If it runs on another machine, add
`--ssh-host YOUR_SSH_ALIAS` and use the endpoint on that host.

```bash
python -m siliconshield.experts assess \
  --registry "$REGISTRY" --scope fifo \
  --design examples/rtl/push-pop-hold.sv \
  --requirement "Outside reset, when push and pop are both asserted, count must remain unchanged." \
  --assumptions "Two-state synchronous logic sampled at the rising clock edge. Reset is exempt from the requirement. Inputs may vary independently." \
  --arbitrate --model glimmer-base \
  --endpoint "$SILICONSHIELD_MUSE_ENDPOINT" --live --brief
```

### 4. Investigate with Muse Spark 1.3

With the Muse CLI installed and signed in, select Spark in the same command.
This sends the RTL, requirement, assumptions, and specialist evidence to the
configured Muse backend.

```bash
python -m siliconshield.experts assess \
  --registry "$REGISTRY" --scope fifo \
  --design examples/rtl/push-pop-hold.sv \
  --requirement "Outside reset, when push and pop are both asserted, count must remain unchanged." \
  --assumptions "Two-state synchronous logic sampled at the rising clock edge. Reset is exempt from the requirement. Inputs may vary independently." \
  --arbitrate --model muse-spark-1.3 --live-logs --brief
```

`--live` shows routing, model loading and inference, Muse session events,
elapsed waiting time, and the returned structured arbitration response in the
terminal. For Spark, `--live-logs` shows one compact line per native lifecycle
event, including the model name, call counts, and completion status. Repeated
output-chunk events are collapsed. JSON payloads, prompts, and local paths stay
out of the terminal. Complete audit logs remain unchanged on disk.
`--brief` displays the specialist scores, coverage gaps, and Muse's conclusion
and explanation without local absolute paths. Omit `--brief` for the full JSON
report. Live-call messages go to stderr so JSON output remains readable by
other tools. Model conclusions remain predictions, with execution errors
reported separately.

[Browser demo](docs/web-demo.md) · [Router and Python API](docs/router.md) ·
[Executable repair example](docs/releases.md#real-muse-repair)

## Research Assets and Deployment Options

**Inspect the data behind the research.** The public
[training collection](https://huggingface.co/datasets/debjyotikm/siliconshield-rtl-training)
contains **19,050 rows**, and the
[development benchmark](https://huggingface.co/datasets/debjyotikm/siliconshield-rtl-benchmark)
provides **5,589 decisions across 11 RTL areas**. The
[training recipes](docs/training-recipes.md) trace each released decision
model to its datasets and training stages. The collection also includes
follow-up study rows that trained candidates the selectors did not promote.
They are published for further training, not as rows the releases saw. Benchmark
results measure development performance, with independent evaluation reported
separately.

**Choose where your RTL is processed.** Decision models run on your own
hardware. Muse arbitration and repair use the configured agent backend,
giving teams a deployment choice that matches their design-confidentiality
requirements. When using a hosted backend, the requested analysis can transmit
RTL and requirements to that service.

**Bring Muse Glimmer into the local workflow.** The
[Glimmer router skill](.agents/skills/glimmer-router/SKILL.md) connects locally
served Glimmer to the decision models through structured tool calls. Glimmer
can inspect the available specialists, request assessments for a design
domain, and reason over the returned evidence. The repository includes
[serving, tool-use, and replay tooling](docs/inference.md) to operate and
evaluate this capability. Glimmer supplies the agent reasoning. Training
updates the decision models, not Glimmer's weights.

Router-driven assessments have been demonstrated in two base-model sessions
([router evidence](docs/router.md#driving-assessments-from-glimmer)). Broader
driver reliability is not established and session verdicts inherit the experts'
documented domain limits. Unattended router operation still requires
qualification for that role,
separate from the demonstrated local training workflow. For research
campaigns, optional Harness² integration supplies replay-qualified procedures
within the study’s existing permissions and acceptance rules.

## Hackathon contribution scope

SiliconShield-specific work was developed during the Meta Global AI Developer
Hackathon building period, October 27–November 4, 2026. This includes the RTL
decision task and dataset curation, Contract-1 and Contract-2 training, the
capability router, Muse integrations and agent skills, recoverable training
studies, and the browser demonstration.

The [demo contribution map](docs/web-demo.md#hackathon-contribution-scope)
identifies the new work shown in each segment. LAYA/LAYA Next, Meta's Muse
models, upstream RTL, verification tools, and optional Harness² are reused
foundations, credited in [NOTICE](NOTICE.md). SiliconShield trains the decision
models. Glimmer operates the workflows without changes to its weights.

## Engineering references

Browse the [documentation index](docs/README.md) for recommended reading paths,
or go directly to a guide below.

| Area | Documentation |
|---|---|
| Getting started | [Environment setup](docs/setup.md) · [Model downloads and releases](docs/releases.md) · [Demo setup and walkthrough](docs/web-demo.md) · [Example formats](examples/README.md) |
| RTL investigation | [Requirements and RTL examples](docs/rtl-requirements.md) · [Router and API](docs/router.md) · [Decision-model qualification](docs/expert-qualification.md) · [Command and configuration reference](docs/reference.md) |
| Architecture | [System architecture](docs/architecture.md) · [Architecture diagrams](docs/diagrams/README.md) |
| Decision models | [Contract-1 and Contract-2 capabilities](docs/siliconshield-rtl.md) · [Measured limitations](docs/LIMITATIONS.md) · [Release training recipes](docs/training-recipes.md) |
| Data and evaluation | [Dataset construction and qualification](docs/data-pipeline.md) · [Evaluation protocol](docs/evaluation.md) · [Benchmarking models and routing](docs/benchmark.md) |
| Model training | [Study walkthrough](docs/study-walkthrough.md) · [Training methods and controls](docs/security-model-training.md) · [Trainer entry points](siliconshield/training/laya/README.md) |
| Campaign operation | [Multi-round training campaigns](docs/campaigns.md) · [Training operations and recovery](docs/training-operations.md) |
| Optional integrations | [Muse and Harness² procedure learning](docs/muse-procedures.md) |
| Local Muse Glimmer | [Deployment and qualification](docs/inference.md) · [Train decision models with Glimmer](docs/decision-model-training-with-glimmer.md) · [Local training case study](docs/case-study-decision-model-training-with-glimmer.md) |
| Research | [Scientific white paper](docs/whitepaper.md) · [Research graphs and figure sources](docs/whitepaper/figures/README.md) |
| Development | [Contributing](CONTRIBUTING.md) · [Coding-agent guidance](AGENTS.md) |
| Agent skills: investigation | [RTL analysis](.agents/skills/rtl-analysis/SKILL.md) · [Glimmer routing](.agents/skills/glimmer-router/SKILL.md) |
| Agent skills: training | [Study design](.agents/skills/study-design/SKILL.md) · [Glimmer studies](.agents/skills/glimmer-study/SKILL.md) · [Round diagnosis](.agents/skills/round-diagnosis/SKILL.md) · [Campaign operations](.agents/skills/campaign-ops/SKILL.md) |

Repository code is **[Apache-2.0](LICENSE)**. Model weights and datasets have
separate terms. See [release terms](docs/releases.md#release-terms) and
[NOTICE](NOTICE.md) for dependency licenses and attribution.

**Upstream projects and models:**
[LAYA](https://github.com/NandhaKishorM/laya) ·
[LAYA Next runtime](https://github.com/debjyotikm/laya-next) ·
[Meta Muse models](https://dev.meta.ai/docs/models) ·
[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) ·
[Google Research Harness²](https://github.com/google-research/harness2)

## Built with Muse

This repository was built with [Meta Muse](https://dev.meta.ai/docs/models).
Muse Spark 1.3 and Muse Wonder models, working through Muse Code and
OpenCode agentic coding sessions, were used to build the infrastructure:
training, the expert router, agent skills, and evaluation tooling. They
supported the Contract-1 and Contract-2 studies on local GPU hardware and
helped draft the documentation.
