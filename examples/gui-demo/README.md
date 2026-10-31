# GUI demo designs

These authored synthetic examples form the desktop GUI's main showcase. Each
plain-English requirement is paired with two small RTL designs: one omits or
weakens a required guard, and the other implements that guard. Choose an entry
from the `Demo:` dropdown in `python -m siliconshield.gui` to fill the
requirement, assumptions, and RTL in one step.

| File | Expected decision | Security rule |
|---|---|---|
| `privilege-check-missing.sv` | Violates | Protected writes require privilege |
| `privileged-write-guarded.sv` | Satisfies | Protected writes require privilege |
| `clear-enable-omitted.sv` | Violates | Status clear requires authorization |
| `clear-enable-enforced.sv` | Satisfies | Status clear requires authorization |
| `address-check-too-broad.sv` | Violates | Register writes require an exact address |
| `exact-address-guard.sv` | Satisfies | Register writes require an exact address |

The expected decision describes each authored fixture. The GUI still reports
the model's actual decision and probabilities. Full requirement and assumption
text lives in
[`requirements.example.json`](requirements.example.json).

The `Large file scan` tab uses
`peripheral-config-bank.sv` as its public showcase. It is one coherent,
label-free design with twelve configuration modules under one privilege
requirement. Module names and comments do not reveal the authored answers.
Expected module outcomes exist only in the test suite and are never included
in the RTL, model input, or Muse excerpt-selection input.

The directory also retains the earlier lock, missing-context, and large-file
examples as regression fixtures:

| File | Purpose |
|---|---|
| `unauthorized-write.sv` | Original missing-privilege fixture |
| `legitimate-access.sv` | Original guarded-write fixture |
| `lock-bypass-second-path.sv` | Lock bypass through a second path |
| `missing-context.sv` | Undefined authorization signal |
| `debug-lock-respected.sv` | Both write paths respect the lock |
| `write-once-then-locked.sv` | Writes stop after lock |
| `peripheral-config-bank.sv` | Label-free large-file showcase |
| `big-faulty-design.sv` | Twelve-module scan fixture |
| `big-mixed-design.sv` | Six renamed regression fixtures |

All files are authored examples. They are not excerpts from training,
calibration, or test datasets.

Fastest path: open the GUI, pick a demo, click `Load Demo`, then click
`Run Analysis`. To load a file manually, browse to
`examples/gui-demo/<name>.sv` and paste its matching requirement. On the
`Large file scan` tab, choose `Peripheral configuration bank`, click
`Load Demo`, then click `Scan`.
