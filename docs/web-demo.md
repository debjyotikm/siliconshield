# Browser demonstration

[Documentation](README.md)

The browser app exposes three views over SiliconShield's Glimmer specialists
and capability router. Muse Glimmer on configured serving hardware reads the
Contract skills and returns specialist verdicts. The same router flow that the
command line uses plans and validates every consultation. The browser never
supplies model scores or substitutes a reference answer when execution fails.
No local decision weights load in this app.

| View | Operation | Required configuration |
|---|---|---|
| Contract-1 | Edit RTL, requirement, and assumptions. Consult the local-guard skill and inspect all three scores. | Glimmer service plus a registry entry for `contract-1` |
| Contract-2 | Examine FIFO and counter examples through the temporal skill. | Glimmer service plus a registry entry for `contract-2` |
| Expert Analysis | Consult measured specialists for a selected scope, with Muse Spark final assessment on by default. | Registry with capability profiles, a Glimmer service, and the `muse` CLI |

## Start the app

Use the lightweight controller environment from [setup](setup.md). No Qt,
JavaScript build tools, model download, or GPU allocation is required to open
these views:

```bash
python -m pip install -e .
python -m siliconshield.web_demo
```

Open **http://127.0.0.1:8765** in Chrome. Without a configuration, the app shows
authored examples and disables assessment until Glimmer is linked. It listens
only on loopback. `Ctrl+C` stops the HTTP server.

For configured operation, copy [web-demo.example.json](../configs/web-demo.example.json)
to a workspace outside the repository, edit it, and run:

```bash
python -m siliconshield.web_demo --config /path/to/web.local.json --check
python -m siliconshield.web_demo --config /path/to/web.local.json
```

`--check` validates local settings and registry structure without contacting
Glimmer. `--port 8766` selects another port. Use a separate `state_dir` for
each running instance. Relative paths resolve against the configuration file.
Request receipts remain outside the source tree.

## Register the specialists

The browser recognizes the registry IDs `contract-1` and `contract-2`. Either
can be registered alone. Each entry must be a `glimmer` specialist: a skill
directory, pinned `SKILL.md` and serving digests, and a capability profile.
The skill bytes are digest-checked on every consultation, and the serving
fingerprint must match the registry and profile binding exactly. A registry
that names local-worker entries is refused with an explicit error.

```json
{
  "registry": "models.glimmer.json",
  "state_dir": "browser-state",
  "port": 8765,
  "glimmer": {
    "endpoint": "http://127.0.0.1:18400/v1/chat/completions",
    "model": "glimmer-base",
    "fingerprint": "64-hex-serving-digest",
    "ssh_host": "spark"
  }
}
```

After any skill edit, re-pin that skill's digest in the registry. A stale pin
fails fast with a skill-digest mismatch instead of consulting changed text.

The specialist views accept typed RTL or an uploaded design file. They assess
the supplied region against the entered requirement and assumptions. Each
verdict panel lists its live Glimmer calls with endpoint, model, timing, the
skill digest actually sent, and the exact response text. Call progress also
streams into the result panel while the request runs, with a heartbeat every
ten seconds. Inputs
that exceed the context budget return a coverage error without a verdict.
The bundled cases are authored presentation examples, including challenge
cases. They are not an independent benchmark or a promise of correct output.

## Configure Glimmer

Glimmer is required: every verdict in this app comes from the configured
service. The endpoint must be loopback HTTP ending in `/chat/completions`.
For a service on another machine under your control, set `ssh_host` to your
SSH alias and keep the endpoint as that machine's loopback address. SSH uses
the user's existing key and host verification, without requiring TCP
forwarding. See [inference](inference.md) for serving.

The `fingerprint` is the serving weights digest. It is recorded in every
response identity and checked against the registry and profiles, so a verdict
cannot silently come from unregistered weights. Health checks only inspect
model availability through the models list.

An optional demo configuration serves the fine-tuned adapter on local
GPUs instead of the Spark chat service. Its registry uses `adapter`
entries and its endpoint is the scoring shim. Verdicts arrive as
normalized answer-token scores in a fraction of a second. Final
assessment still needs a generative arbiter, configured separately as
`arbiter` with `{"engine": "spark", "model":
"muse-spark-1.3-contributor"}`. One demo serves one engine. Mixed
registries are refused.

## Route and arbitrate

Attach measured capability profiles to the registry. Select a supported scope
in the router view and submit the RTL, requirement, and assumptions. The
response preserves consulted experts, scores, coverage, skill identity, and
errors. A registry without profiles can support individual specialist views
but supplies no measured routing coverage.

Routing uses the frozen router flow: planned coverage, profile tiers, and
validated specialist envelopes. Failures narrow who answered, never what the
plan covered, so split-scope guidance survives exactly when it is needed most.

Final assessment starts checked. Uncheck it to keep the assessment to specialist evidence only. When enabled, it sends the submitted RTL,
requirement, assumptions, and specialist evidence to Muse Spark
(`muse-spark-1.3-contributor` through the local `muse` CLI). Invalid or
incomplete agent output is reported as an arbitration
error while specialist results remain available.

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

**Download result** saves the actual response, including all three class
scores, submitted input, skill identity, and execution errors. Server receipts
are retained in `state_dir`. Treat them as design data.

This is a single-user local demonstration, not an Internet-facing service.
Loopback binding, request tokens, and origin checks prevent accidental browser
cross-site dispatch. They do not provide user authentication, an untrusted-code
sandbox, or a multi-tenant deployment boundary. Do not expose its port publicly.

The browser integration is covered by synthetic HTTP, router, configuration,
and recovery tests, plus installed-wheel asset checks. These checks do not
constitute a new GPU execution or model-quality evaluation. Capability scope
and known limits remain in the [model card](siliconshield-rtl.md) and
[limitations](LIMITATIONS.md).

## Hackathon contribution scope

The SiliconShield-specific work below was built during the October 27–November 4,
2026 hackathon building period. Identify these contributions in the video and
submission description so reviewers can distinguish them from the reused
foundation models and tools.

| Demo segment | SiliconShield contribution | Implementation and evidence |
|---|---|---|
| Contract-1 and Contract-2 skills | Specialist skills for local-guard and temporal RTL review, validated through Glimmer with digest-bound registries | [Contract-1 skill](../.agents/skills/contract-1/SKILL.md) · [Contract-2 skill](../.agents/skills/contract-2/SKILL.md) |
| Expert Analysis | Capability-based routing, input-bound specialist evidence, Muse Spark final assessment, and RTL analysis skills | [Router code](../siliconshield/experts/) · [Specialist flow](../siliconshield/research_loop/procedures/glimmer_specialists.py) |
| Browser application | Three demonstration views connected to the Glimmer specialists and router | [Browser source](../siliconshield/web_demo/) |

Meta supplies the Muse models. Their attribution remains in
[NOTICE](../NOTICE.md). Glimmer's weights are unchanged. The demonstrated
workflow results are SiliconShield outputs, with their measured scope recorded
in the linked evidence.

## Demonstration sequence

Prepare the serving endpoint and rehearse the authored inputs before recording.
Use this three-minute sequence to show the product through the browser:

| Time | Show | Explain |
|---|---|---|
| 0:00–0:30 | Contribution scope, Contract-1 RTL, and its requirement | State what was built for the hackathon, then identify the protected register update and guard. |
| 0:30–1:00 | Contract-1 assessment before and after a guard edit | Show all class scores and the actual response to the changed design. |
| 1:00–1:30 | Contract-2 FIFO or counter example | Explain the state-update requirement and the temporal specialist's prediction. |
| 1:30–2:15 | Expert Analysis with Spark final assessment | Show consulted experts, scope coverage, disagreements, and unresolved questions. |
| 2:15–3:00 | Skills and evidence | Point to the skill text, pinned digests, measured capabilities, and limitations. |

This is presentation pacing, not an execution-time estimate. Label recordings
and fixture outputs. Preserve actual responses, including incorrect predictions
and failed requests. Do not imply that the sequence itself has been qualified
or recorded.

For a terminal presentation, use the [router walkthrough](router.md#run-one-assessment).
Executable repair is a separate optional segment: run an executable repair
demonstration, configure a live proposer, and show the original failure,
proposed diff, and unchanged requirement and legitimate-write checks. An
authored repair must retain that label.

## Recorded evidence

The current verification covers the specialist path end to end. The fixture
suite exercises configuration, HTTP, routing, skill-digest refusal,
fingerprint mismatch, malformed verdicts, and recovery without live services.
A live serving check through the demo backend consulted the Contract-1 skill
over the real Glimmer service for the authored guarded-register case and
returned satisfies at 0.95 with validated scores and skill identity.

This was live serving verification on authored cases. It is not an unseen
evaluation or an engineering time-savings study. Raw input/output and check
receipts remain in the release-preparation bundle outside Git. Use that bundle
to reproduce this workflow.
