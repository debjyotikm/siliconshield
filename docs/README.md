# SiliconShield engineering documentation

[Project overview](../README.md) · [Model releases](siliconshield-rtl.md) ·
[Source contribution guide](../CONTRIBUTING.md)

SiliconShield has two core workflows: investigate RTL bugs and security flaws
with the router, Muse, and the Contract skills, and improve specialist
coverage through reproducible studies. The document set covers their
design, integration, scientific evidence, and operation. Specifications describe
implemented contracts. Guides provide procedures. Model cards and evaluation
documents state the evidence supporting released behavior.

For a reproducible first run, use [setup](setup.md).
For the experimental rationale and results, read the [research white paper](whitepaper.md).
Upstream attribution and dependency terms are in [NOTICE](../NOTICE.md).

## Reading paths

| Reader | Start | Continue |
|---|---|---|
| Hardware engineer | [Writing RTL requirements](rtl-requirements.md) | [Router and Muse assessment](router.md) → [Model capabilities](siliconshield-rtl.md) |
| Application integrator | [System architecture](architecture.md) | [Assessment protocol](router.md) → [Interface reference](reference.md) |
| Model researcher | [Model card](siliconshield-rtl.md) | [Data qualification](data-pipeline.md) → [Research white paper](whitepaper.md) |
| Local Glimmer operator | [Local Glimmer deployment](inference.md) | [Setup](setup.md) → [Measured case study](case-study-training-with-glimmer.md) |
| Platform operator | [Environment setup](setup.md) | [Browser demo](web-demo.md) |
| Contributor or reviewer | [Architecture](architecture.md) | [Contributing and tests](../CONTRIBUTING.md) |

## Architecture and interfaces

| Document | Owns |
|---|---|
| [System architecture](architecture.md) | Process boundaries, interfaces I01–I04, artifact ownership, failure model, design trade-offs, extension points |
| [Expert routing and assessment](router.md) | Scope selection, reliability tiers, transaction sequence, validated responses, Muse arbitration |
| [Interface reference](reference.md) | Commands, wire fields, identity, score/coverage validation, timeouts, configuration and artifact names |
| [Architecture source](diagrams/README.md) | Editable draw.io diagrams and SVG previews for component, sequence, and lifecycle views |

## Applications and operation

| Document | Owns |
|---|---|
| [Writing requirements for RTL analysis](rtl-requirements.md) | English requirements, satisfying and violating RTL, reset and transfer semantics, and missing context |
| [Setup](setup.md) | Separate controller, model-worker, and training environments |
| [Browser demo](web-demo.md) | Browser setup, demonstration sequence, and recorded integration evidence |

## Models, data, and research

| Document | Owns |
|---|---|
| [Research white paper](whitepaper.md) | Scientific account of Muse-guided research, Contract-1 and Contract-2, training methods, hardware, results, and bottlenecks |
| [Glimmer-operated training case study](case-study-training-with-glimmer.md) | Two Glimmer-operated training runs on local hardware: a verified toy loop and a real-scale null result |
| [Contract-1 and Contract-2 model card](siliconshield-rtl.md) | Released artifacts, task definitions, domain strengths, headline evidence, lineage |
| [Capabilities and limitations](LIMITATIONS.md) | Confusion matrices, failure examples, scope limits, and confidence behavior |
| [Dataset construction and qualification](data-pipeline.md) | Solver evidence, English-to-assertion reviews, lineage isolation, immutable acceptance, sampling coverage |
| [Training](security-model-training.md) | Trainable parameters, optimization objective, DDP, selection versions, resume and export |
| [Glimmer training guide](training-guide.md) | LoRA adaptation recipe, dataset reuse, execution failures, verification |
| [Local Muse Glimmer deployment and qualification](inference.md) | Local Glimmer operation, serving, and replay qualification |

## Advanced integrations

These integrations are optional. Retained local inference, router assessment,
and basic local training do not require Harness².

| Document | Owns |
|---|---|
| [Muse and Harness²](muse-procedures.md) | Replay-qualified procedure learning and guidance adoption |

## Conventions and evidence

Run examples from the repository root unless specified otherwise. Replace
`/path/to/` values with local paths. Store generated outputs outside the
checkout. A command's prerequisites and side effects are part of its guide.
A `--help` check validates the interface, not a GPU or model-quality result.

Published weights: [Contract-1](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-1)
and [Contract-2](https://huggingface.co/debjyotikm/siliconshield-rtl-contract-2).
Published data: [training collection](https://huggingface.co/datasets/debjyotikm/siliconshield-rtl-training)
and [development benchmark](https://huggingface.co/datasets/debjyotikm/siliconshield-rtl-benchmark).
The published collections record which populations contributed to which
release. Downloading a previously used development set does not make it
an independent final evaluation.

Fixture tests, live integration, GPU execution, model quality, research promotion,
and publication are distinct evidence levels. Each result should name its
artifact, population, protocol, and acceptance rule. The [research white paper](whitepaper.md)
is the canonical reference for scientific claims.

## Terms

- **Contract skill:** versioned domain knowledge that Muse Glimmer applies
  to assess RTL against a requirement and assumptions. Contract-1 and
  Contract-2 each return one of three labels with supporting evidence.
- **Retained local model:** the weight-based research path kept in this
  repository. It is disabled by default and is not part of the shipped
  product.
- **Requirement:** the behavior assessed together with explicit assumptions.
- **False-safe:** a true violation predicted as satisfying the requirement.
- **Whole triplet:** linked satisfying, violating, and insufficient-context
  decisions, scored correct only when all three are correct.
- **Lineage / split group:** related designs or experiments assigned together
  when partitioning data. Use these groups as the clustering unit for uncertainty
  estimates. A distinct group ID alone does not establish independence.
- **Selected checkpoint:** the weights chosen inside one run by its rules.
- **Incumbent:** the model retained by the campaign's promotion policy.
- **Thin evidence:** insufficient routing support, weak measured capability, or
  unresolved coverage. It is not the model's `insufficient_context` label.
- **Receipt:** structured evidence about inputs, execution, identity, or a decision.
- **Replay:** a declared controller task executed through controlled tools.
  Fixture replay and live generation answer different questions.

### Training control hierarchy

| Term | Responsibility |
|---|---|
| Study | Immutable experimental contract: parent, data, settings, evaluation rules and budget |
| Campaign | Persistent series of proposed and assessed rounds, with an incumbent |
| Round | One registered hypothesis and candidate, assessed against frozen rules |
| Supervisor | Reconciles round boundaries and invokes an agent for diagnosis or a proposal |
| Strategy driver | Bounded fitting-arm comparison followed by one development job |
| Operator | Owns one job's prepare, submit, observe, intervention and finalization state |
| Recipe | Training and reporting settings. It does not itself authorize a job |
| Scope | Caller-declared design behavior for routing, such as `fifo` |
| Mechanism / family | Dataset-specific behavioral category / related design lineage. They are not interchangeable with independent statistical samples |

A supervisor may call an operator for each round. The strategy driver is a
separate bounded entry point, not a synonym for that supervisor. The detailed
[configuration reference](reference.md#configuration-boundaries) names the
artifacts connecting these layers.
