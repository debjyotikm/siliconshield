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
    <a href="docs/decision-study.md">Train a specialist</a> ·
    <a href="docs/README.md">Engineering documentation</a>
  </p>
</div>

An unguarded debug write can bypass a lock that is enforced on the normal
path. A FIFO count can drift when simultaneous push/pop or backpressure is
handled incorrectly. Reviewing these failures requires **the RTL, a precise
requirement, and explicit assumptions**.

**SiliconShield evaluates RTL against explicit requirements.** A capability
router consults locally trained specialist decision models. Meta Muse
interprets their results against the RTL and produces a finding with supporting
evidence and open questions. The platform also provides the data, training,
evaluation, and recovery infrastructure to improve an existing decision model
or develop one for another design domain.

Model conclusions guide RTL review and investigation. Simulation and repair
checks run through a separate application that records what was executed.
The [research white paper](docs/whitepaper.md) traces the models, datasets,
training decisions, and measured failure modes behind the system.

## Decision Models

**Give agents specialist RTL judgment they can call as a tool.** SiliconShield’s
decision models assess a design against an explicit requirement and return a
structured result: **satisfies the requirement**, **violates the requirement**,
or **insufficient context**. Muse uses the router to consult the relevant
specialist, interpret its assessment, and focus the investigation on the logic
that matters. The models run locally and expose a common interface, so new
specialists can extend the workflow to additional design domains.

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
look, what may be wrong, and what to check next**.

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

[Explore decision-model training](docs/decision-study.md)

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

[Explore the local Glimmer training case study](docs/case-study-glimmer-local-training.md) ·
[Design a study](docs/decision-study.md) ·
[Prepare training data](docs/data-pipeline.md) ·
[Training recipes](docs/training-recipes.md)

## Get started

### Assess RTL with the decision models and Muse

Start with a design region and a concrete requirement. SiliconShield routes
the assessment to the relevant decision models, then Muse examines their
predictions alongside the RTL to explain the conclusion and any disagreement.

Follow the [setup guide](docs/setup.md) to download the models and configure
the runtime, then [register measured capability profiles](docs/router.md).
The [FIFO walkthrough](docs/router.md#run-one-assessment) provides the complete
input file for this example.

```bash
python -m siliconshield.experts assess \
  --registry /path/to/models.local.json \
  --scope fifo \
  --state-file /path/to/assessment.txt \
  --arbitrate \
  --requirement "After reset, a push while full must leave occupancy unchanged."
```

The response includes each decision model’s prediction and scores, its measured
support for the requested scope, and Muse’s assessment with supporting
reasoning. Disagreements, execution failures, and coverage gaps remain visible.

Arbitration requires the configured Muse CLI and login. To inspect
decision-model evidence alone, omit `--arbitrate` and `--requirement`, keeping
the requirement in the input file. Python integrations can retain model
workers across requests.

### Check a repair against the requirement

The included `unauthorized-write` example demonstrates a common
register-protection defect: a write enable updates state without checking
privilege. The candidate repair restores the missing guard.

```diff
- else if (wr_en) cfg_reg <= wdata;
+ else if (wr_en && privileged) cfg_reg <= wdata;
```

With Python 3.11+, Verilator, a C++ compiler, and `make` installed, run:

```bash
python -m pip install -e .
python -m siliconshield.app --case unauthorized-write \
  --output /tmp/siliconshield-run
```

Open `/tmp/siliconshield-run/report.md` to inspect the original failure,
candidate change, and executed checks. The repair must reject unprivileged
writes **and preserve legitimate writes**. Disabling every write does not meet
the acceptance criteria.

This installation smoke test uses an authored repair and needs no model
weights or Muse account. Without a configured checkpoint, the report marks
the model assessment as unavailable. The
[investigation guide](docs/investigation.md) explains how to connect a decision
model and a live Muse proposer.

A successful run establishes that the candidate passed the supplied
requirement and regression checks. Model predictions and executed verification
results remain separate throughout the report.

## Research Assets and Deployment Options

**Inspect the data behind the research.** The public
[training collection](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-training)
contains **19,050 rows**, and the
[development benchmark](https://huggingface.co/datasets/danieltombay/siliconshield-rtl-benchmark)
provides **5,589 decisions across 11 RTL areas**. The
[training recipes](docs/training-recipes.md) trace each released decision
model to its datasets and training stages. The collection also includes
follow-up experiments that did not produce the released weights. Benchmark
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
[adaptation, serving, and replay tooling](docs/glimmer.md) to develop and
evaluate this capability.

Router-driven assessments have been demonstrated in two base-model sessions.
Unattended router operation still requires qualification for that role,
separate from the demonstrated local training workflow. For research
campaigns, optional Harness² integration supplies replay-qualified procedures
within the study’s existing permissions and acceptance rules.

## Engineering references

Browse the [documentation index](docs/README.md) for recommended reading paths,
or go directly to a guide below.

| Area | Documentation |
|---|---|
| Getting started | [Environment setup](docs/setup.md) · [Model downloads and releases](docs/releases.md) · [Demo walkthrough](docs/demo.md) · [Example formats](examples/README.md) |
| RTL investigation | [Router and API](docs/router.md) · [Investigation and repair checks](docs/investigation.md) · [Expert qualification](docs/expert-qualification.md) · [Command and configuration reference](docs/reference.md) |
| Architecture | [System architecture](docs/architecture.md) · [Architecture diagrams](docs/diagrams/README.md) |
| Decision models | [Contract-1 and Contract-2 capabilities](docs/siliconshield-rtl.md) · [Measured limitations](docs/LIMITATIONS.md) · [Release training recipes](docs/training-recipes.md) |
| Data and evaluation | [Dataset construction and qualification](docs/data-pipeline.md) · [Evaluation protocol](docs/evaluation.md) · [Benchmarking models and routing](docs/benchmark.md) |
| Model training | [Decision studies](docs/decision-study.md) · [Training methods and controls](docs/security-model-training.md) · [Trainer entry points](siliconshield/training/laya/README.md) |
| Campaign operation | [Campaigns and controllers](docs/campaigns.md) · [Operations and recovery](docs/operations.md) · [Muse and Harness² procedures](docs/muse-procedures.md) |
| Local Muse Glimmer | [Controller training and qualification](docs/glimmer.md) · [Inference and serving](docs/inference.md) · [Local training case study](docs/case-study-glimmer-local-training.md) |
| Research | [Scientific white paper](docs/whitepaper.md) · [Research graphs and figure sources](docs/whitepaper/figures/README.md) |
| Development | [Development and tests](docs/development.md) · [Python style](docs/python_style.md) · [Contributing](CONTRIBUTING.md) · [Coding-agent guidance](AGENTS.md) |
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
