# Browser demonstration

[Documentation](README.md) · [Release setup](releases.md)

The browser app exposes four views over SiliconShield's model, router, and
training interfaces. Models run in owned worker processes,
routing uses the capability service, and the optional training view submits a
bounded toy study through the operator. The browser never supplies model scores
or substitutes a reference answer when execution fails.

| View | Operation | Required configuration |
|---|---|---|
| Contract-1 | Edit RTL, requirement, and assumptions. Load the local-guard decision model and inspect all three scores. | Registered Contract-1 export and worker environment |
| Contract-2 | Examine FIFO and counter examples with the temporal decision model. | Registered Contract-2 export and worker environment |
| Expert router | Consult measured specialists for a selected scope. Optionally ask local Muse Glimmer to interpret their evidence. | Registry with capability profiles. Glimmer is optional. |
| Train with Muse | Follow dataset checks, TRAIN fitting, development, and closeout of a small Contract-1 decision-model adapter-training exercise. | Linux user services, an available NVIDIA GPU, training environment, and local Glimmer |

## Start the app

Use the lightweight controller environment from [setup](setup.md). No Qt,
JavaScript build tools, model download, or GPU allocation is required to open
these views:

```bash
python -m pip install -e .
python -m siliconshield.web_demo
```

Open **http://127.0.0.1:8765** in Chrome. Without a configuration, the app shows
authored examples and disables model loading and training. It listens only on
loopback. `Ctrl+C` stops the HTTP server and its inference workers.

For configured operation, copy [web-demo.example.json](../configs/web-demo.example.json)
to a workspace outside the repository, edit it, and run:

```bash
python -m siliconshield.web_demo --config /path/to/web.local.json --check
python -m siliconshield.web_demo --config /path/to/web.local.json
```

`--check` validates local settings and registry structure without loading weights,
contacting Glimmer, or starting training. `--port 8766` selects another port.
Use a separate `state_dir` for each running instance. Relative paths resolve
against the configuration file. Model files, request receipts, and generated
training files remain outside the source tree.

## Register the decision models

Download the immutable exports and prepare the worker environment using
[release setup](releases.md). Then create the unprofiled registry with the
[registration script](expert-qualification.md#register-and-measure-the-published-models).
For individual CPU inference, set `SILICONSHIELD_DEVICE=cpu` when creating that
registry. The example registry shape is [models.example.json](../configs/models.example.json).
Use the release manifest's complete `expected_files` mapping, not the example's
placeholder digest.

The browser recognizes the registry IDs `contract-1` and `contract-2`. Either
can be registered alone. This configuration enables CPU inference:

```json
{
  "registry": "models-unprofiled.json",
  "state_dir": "browser-state",
  "port": 8765,
  "gpu": null,
  "glimmer": null,
  "training": {"enabled": false}
}
```

To offer CPU and GPU choices, change `gpu` to the available device index,
for example `0`. Select the device in a model view, click **Load model**, wait
for its ready receipt, then click **Run analysis**. Changing devices requires
a reload. CPU means the server's CPU, not the browser machine's CPU. The worker
runs FP32 on CPU and BF16 on CUDA. Predictions near a decision boundary can
differ across these execution settings.

The model views accept typed RTL or an uploaded design file. They assess the
supplied region against the entered requirement and assumptions. Inputs that
exceed the worker's token budget return a coverage error without a verdict.
The bundled cases are authored presentation examples, including challenge
cases. They are not an independent benchmark or a promise of correct output.

## Route and arbitrate

Attach measured capability profiles using [decision-model qualification](expert-qualification.md).
Set `registry` to the generated `models.local.json`. Select a supported scope in
the router view and submit the RTL, requirement, and assumptions. The response
preserves consulted experts, scores, coverage, worker identity, and errors.
A registry without profiles can support individual model views but supplies no
measured routing coverage.

Router inference uses each entry's registered device. It does not inherit the
CPU/GPU choice from a model tab. Any CUDA device used by the registry must equal
the demo's configured `gpu`. Capability measurements must match the observed
worker configuration. A CPU run cannot inherit a GPU measurement's tier.

For optional arbitration, configure an already running local Glimmer service:

```json
"glimmer": {
  "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
  "model": "your-served-model-name"
}
```

This is a field inside `web.local.json`. For a service on another machine under
your control, add `"ssh_host": "your-ssh-alias"`. The endpoint remains that
machine's loopback address. SSH uses the user's existing key and host verification,
without requiring TCP forwarding. See [inference](inference.md) for serving.

Arbitration starts unchecked. Enabling it sends the submitted RTL, requirement,
assumptions, and specialist evidence to the configured Glimmer service. Invalid
or incomplete agent output is reported as an arbitration error while specialist
results remain available. Health checks only inspect model availability.

## Enable the bounded training example

Training stays disabled until an administrator explicitly configures it. The
browser offers start, status, evidence download, and a stop request. It cannot
change the parent, dataset, learning rate, update budget, or executable paths.

The training host needs Linux, working `systemctl --user` and `systemd-run`, a
training environment from [setup](setup.md), and the pinned LAYA runtime. Set a
GPU index and Glimmer configuration as above, then add:

```json
"training": {
  "enabled": true,
  "python": "/path/to/training-env/bin/python",
  "runtime": "/path/to/pinned-laya-runtime",
  "output_dir": "/path/to/external-workspace/toy-runs",
  "resource_root": "/path/to/shared-job-resources"
}
```

The training interpreter must import SiliconShield. Use the same `resource_root`
as other SiliconShield jobs on the host so their resource claims are visible.
The app verifies the registered Contract-1 parent files before launch and trains
a child export. The registered decision models are never replaced.

The example trains a child of the registered Contract-1 model on 18 TRAIN and
6 development decisions. It runs a TRAIN-only fitting phase, then a separate
development phase from the parent, under a 30-minute outer service limit.

<details>
<summary>Fixed training configuration</summary>

| Fixed setting | Value |
|---|---|
| Parent | Registered Contract-1 export |
| Data | 18 TRAIN and 6 development decisions, balanced across three labels |
| Sampling | Complete three-label groups, global batch 6 |
| Parameters | Frozen encoder, continued residual adapters and decision head |
| Optimizer rates | Adapter and head learning rates of `1e-5` |
| Fitting phase | 3 updates, TRAIN-only evidence |
| Development phase | Fresh parent, 3 updates, development selection |
| Controller | Local Glimmer, bounded operator actions, no intervention budget |
| Outer service limit | 30 minutes |

</details>

Input validation checks contracts, split isolation, and delivered sampling
coverage before submission. Preflight releases only this app's inference
workers and refuses training if any GPU compute workload remains on the host.
This is deliberately conservative. A Glimmer service on that same GPU host also
blocks this toy launch. A service on separate local hardware can remain running.
The output volume needs 64 GiB free and the state volume needs 5 GiB free.

The two phases demonstrate the fitting/development workflow. They do not form
a six-update continuation. TRAIN and development share authored template
patterns, so their results establish operability only. No calibration or final
test is used, no model-quality improvement is established, and no release is
promoted. Use [study walkthrough](study-walkthrough.md) for a qualified experiment.

**Stop** requests a checkpoint-boundary stop for owned training jobs and prevents
the next phase. Closing the browser or HTTP server does not stop systemd jobs.
Restart with the same configuration to reconnect to their receipts. If the
controller fails while a child job is still active, the app keeps that ownership
visible and blocks another launch. Keep the source, runtime, and local settings
unchanged until the run settles.

## Open it from a Mac

Run the server on the Linux workstation. From the Mac, forward the same port
through your existing SSH alias:

```bash
ssh -N -L 8765:127.0.0.1:8765 your-workstation-alias
```

Open **http://127.0.0.1:8765** in Chrome on the Mac. Keep the SSH connection open.
If the port is occupied, choose another port on both ends and start the server
with the matching `--port`. Host and origin validation require the browser's
port to match the server's port.

## Evidence and operating boundary

**Download result** saves the actual response, including all three class scores,
submitted input, worker identity, and execution errors. Server receipts are
retained in `state_dir`. Treat them as design data. Training evidence is stored
under the configured output directory and remains available after closeout.

This is a single-user local demonstration, not an Internet-facing service.
Loopback binding, request tokens, and origin checks prevent accidental browser
cross-site dispatch. They do not provide user authentication, an untrusted-code
sandbox, or a multi-tenant deployment boundary. Do not expose its port publicly.

The browser integration is covered by synthetic HTTP, router, configuration,
training-submission, and recovery tests, plus installed-wheel asset checks.
These checks do not constitute a new GPU execution or model-quality evaluation.
Published model evidence remains in the [model card](siliconshield-rtl.md) and
[limitations](LIMITATIONS.md).

## Hackathon contribution scope

The SiliconShield-specific work below was built during the October 27–November 4,
2026 hackathon building period. Identify these contributions in the video and
submission description so reviewers can distinguish them from the reused
foundation models and tools.

| Demo segment | SiliconShield contribution | Implementation and evidence |
|---|---|---|
| Contract-1 and Contract-2 | Requirement-conditioned RTL datasets, specialist training, released decision models, and structured assessment interfaces | [Training code](../siliconshield/training/laya/) · [Release recipes](training-recipes.md) · [Measured results](siliconshield-rtl.md) |
| Expert router | Capability-based routing, input-bound specialist evidence, Muse arbitration, and RTL analysis/tool-use skills | [Router code](../siliconshield/experts/) · [Glimmer router skill](../.agents/skills/glimmer-router/SKILL.md) |
| Local decision-model training | Frozen studies, recoverable GPU jobs, bounded Glimmer actions, checkpoint selection, and retained closeout evidence | [Execution code](../siliconshield/research_loop/execution/) · [Local training case study](case-study-decision-model-training-with-glimmer.md) |
| Browser application | Four demonstration views connected to the model workers, router, and training infrastructure | [Browser source](../siliconshield/web_demo/) |

LAYA supplies the decision-model foundation and runtime. Meta supplies the Muse
models. OpenTitan and HardSecBench supply upstream RTL used in dataset
construction. EDA tools and optional Harness² are external dependencies. Their
attribution remains in [NOTICE](../NOTICE.md). Glimmer's weights are unchanged.
The released decision models and demonstrated workflow results are SiliconShield
outputs, with their measured scope recorded in the linked evidence.

## Demonstration sequence

Prepare the pinned models and rehearse the authored inputs before recording.
Use this three-minute sequence to show the product through the browser:

| Time | Show | Explain |
|---|---|---|
| 0:00–0:30 | Contribution scope, Contract-1 RTL, and its requirement | State what was built for the hackathon, then identify the protected register update and guard. |
| 0:30–1:00 | Contract-1 assessment before and after a guard edit | Show all class scores and the actual response to the changed design. |
| 1:00–1:30 | Contract-2 FIFO or counter example | Explain the state-update requirement and the temporal specialist's prediction. |
| 1:30–2:15 | Router assessment with optional Glimmer arbitration | Show consulted experts, scope coverage, disagreements, and unresolved questions. |
| 2:15–2:45 | Bounded training view or recorded closeout | Show the parent, fitting/development phases, and accepted or rejected outcome. |
| 2:45–3:00 | Model releases and evidence | Point to the measured capabilities, limitations, and reproducible setup. |

This is presentation pacing, not an execution-time estimate. Label recordings
and fixture outputs. Preserve actual responses, including incorrect predictions
and failed requests. A toy training run demonstrates operation, not an improved
release. Do not imply that the sequence itself has been qualified or recorded.

For a terminal presentation, use the [router walkthrough](router.md#run-one-assessment)
and the [study walkthrough](study-walkthrough.md). Executable repair is a separate
optional segment: follow the [release repair example](releases.md#real-muse-repair), configure
a live proposer, and show the original failure, proposed diff, and unchanged
requirement and legitimate-write checks. An authored repair must retain that
label. The [release rehearsal](releases.md#real-muse-repair) documents that path.

## Recorded evidence

The completed release rehearsal covers model loading and the separate repair
path. It does not establish that the full storyboard above has been recorded.
It downloaded both pinned public models and loaded them through
the real worker on an RTX GPU. A separate invocation combined Contract-1 with
the existing base Glimmer service for the authored unauthorized-write case.
The live proposer generated the missing privilege guard. The original
requirement check failed, the candidate passed it, and legitimate-write checks
passed for both designs. The workflow returned `demonstrated_violation_fixed`.

This was GPU execution plus a live model-generated repair and Verilator
integration on one public authored case. It is not an unseen evaluation or an
engineering time-savings study. Raw input/output and check receipts remain in
the release-preparation bundle outside Git. Follow the
[release repair example](releases.md#real-muse-repair) to reproduce this workflow.
