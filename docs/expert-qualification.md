# Qualifying experts and releases

New decision models join SiliconShield through the registry, never
through code changes. Each release is a registry entry plus a
capability profile, a question contract, and a qualification
reference. Routing and arbitration read those artifacts at load
time.

## Admitting a new expert

1. Add a registry entry with a unique id, checkpoint path,
   `profile`, `release`, `question_contract`, and `qualification`
   fields. Profile paths resolve against the registry file.
2. Author a capability profile matching
   `EXPERT_CAPABILITY_SCHEMA`. Bind every strength claim to a
   checkpoint SHA-256 and a measured evaluation. Record known
   failure cases explicitly.
3. Name the exact question contract the checkpoint was trained
   and evaluated under. Experts on different contracts must not
   share pooled verdicts without a reviewed mapping.
4. Record qualification evidence outside Git: benchmark runs of
   routing plus arbitration against each individual-model
   baseline, with missed violations and false alarms tracked
   per mechanism. Retain the worker’s observed execution identity and
   benchmark bytes alongside the counts in `routing_evidence`. A profile
   with no matching binding is unrated. The registry stores the reference.

## Upgrading an expert

An upgrade is a new release of the same registry id: a new
checkpoint, a new profile, and a new qualification reference.
Keep the previous release entry available so any regression
rolls back by switching the registry back. Never overwrite a
released checkpoint or profile in place.

## Routing and arbitration baselines

Before a release qualifies, measure:

- Per-mechanism routing: which scopes each expert covers and
  which are reported unsupported.
- Arbitration outcomes versus each consulted expert's own
  label on the qualification set.
- Missed violations (false-safe) and false alarms, split by
  mechanism and by agreement versus disagreement cases.

A release qualifies when arbitration meets or beats the best
individual expert on missed violations without increasing
false alarms beyond the recorded budget, and unsupported
scopes stay explicit rather than silently judged.

## Running assessments

Router work lives in `siliconshield.experts` and runs from the
command line or Muse, never from the GUI demo:

```
python -m siliconshield.experts assess --registry models.json \
    --scope fifo,counter --state-file state.txt --arbitrate \
    --requirement "..." --issue "..." --relevance "..."
```

The command prints the routing decision, one validated envelope
per consulted expert, the Muse arbitration verdict when asked,
and a traceable finding when issue metadata is supplied.

## Capability evidence shape

The optional `routing_evidence` object contains `benchmark_source`,
`benchmark_sha256`, `execution_identity`, and `areas`. Each area supplies
integer `correct`, `total`, `violations`, `false_safe` and `errors` counts.
Use the identity returned by the measured worker, including its question,
runtime and inference settings. Do not copy an identity from a different
run or assign tiers by hand. Changes to the inference protocol need new
measurements even when the checkpoint bytes are unchanged.

When a benchmark requirement disagrees with its assertion, quarantine the
whole linked triplet from new capability evidence. Preserve the original
file and receipt, issue a separately identified correction, and review it
before restoring it to qualification. A wording correction is a new input
and cannot inherit old predictions.
