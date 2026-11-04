# Writing requirements for RTL analysis

[Documentation index](README.md) · [Router](router.md) ·
[Model card](siliconshield-rtl.md) · [Dataset construction](data-pipeline.md)

An RTL implementation describes what a circuit does. A requirement describes
what it must do. Assessing a bug means comparing the two under stated operating
conditions. A missing privilege check is a security defect only if the design's
access policy requires that check. A wrapping counter is correct for modulo
arithmetic and incorrect for a saturating event count.

SiliconShield takes the RTL, an English requirement, and explicit assumptions
together. The requirement is part of the problem specification, not a prompt
to adjust until the model returns a preferred answer.

This guide uses small, authored SystemVerilog examples to explain that
relationship. The classifications below describe the intended semantics under
the stated assumptions. They are not predictions measured from Contract-1 or
Contract-2, benchmark results, or evidence of full-block verification.

## Why the requirement comes first

A requirement normally admits several implementations. A priority chain, a
multiplexer, and a next-state expression may implement the same rule. There is
rarely one uniquely “equivalent RTL” for an English requirement.

Separate three questions during review:

| Question | Example |
|---|---|
| What behavior is required? | An unprivileged request must not change the configuration register. |
| What does this implementation permit? | A write branch updates that register whenever the request is asserted. |
| What observation would demonstrate a mismatch? | With privilege low, request a value different from the current register value and observe an update. |

A requirement can prohibit behavior without requiring useful work. “Change only
when privileged” does not require any writes to succeed. A register that never
changes could satisfy that prohibition. Add a separate functional requirement
such as “an enabled privileged write must store the supplied data” to prevent a
repair from disabling legitimate operation.

A functional defect violates the required behavior of the block. A security
defect violates a protection policy, such as authorization, isolation, or
confidentiality. The categories can overlap. A FIFO counter error becomes
security-relevant when a specified protection boundary depends on that count.
The presence of a FIFO alone does not establish a security impact.

## State the conditions that change the answer

Write requirements in plain English. Use signal names to remove ambiguity,
without turning the requirement into a copy of the RTL expression.

| Element | What to specify |
|---|---|
| State and scope | Which register, output, or transaction the requirement protects |
| Event | Whether the rule applies to a request, an accepted transfer, or a completed operation |
| Time | The sampling edge and when the required effect is observed |
| Reset | Whether reset is exempt, and what reset must do when it is included |
| Priority | What happens when writes, clears, debug requests, or transfers coincide |
| Bounds | Width, signedness, capacity, overflow, saturation, and relevant initial state |
| Environment | Input constraints that follow from the interface contract |
| Dependencies | Other writers, internal helpers, or policy definitions needed to decide the result |

**Requirement** states the obligation. **Assumptions** describe the environment
and interpretation under which it must hold. Do not assume away the behavior
being tested. Assuming that unprivileged writes never occur makes an access
control check uninformative unless an independently established upstream
contract guarantees that condition.

For the complete modules below, inputs are two-state and stable around rising
edges of `clk`. Reset is synchronous and active low. Comparisons use the state
immediately before an edge and after its nonblocking updates settle. Reset is
asserted for at least one edge before normal operation. These assumptions do
not cover asynchronous reset release, clock-domain crossings, or X propagation.

## Example 1: privilege-qualified register writes

**Security requirement:** Outside reset, `cfg_reg` may change only when
`privileged` is asserted. Unprivileged write requests must leave it unchanged.

**Companion functional requirement:** Outside reset, when both `wr_en` and
`privileged` are asserted, `cfg_reg` must capture `wdata` at that clock edge.
Otherwise it must hold its value. Reset must clear it to zero.

This implementation meets both requirements:

```systemverilog
module privileged_register (
  input  logic       clk, rst_n, wr_en, privileged,
  input  logic [7:0] wdata,
  output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule
```

Removing the privilege condition introduces an unauthorized update path:

```diff
-    else if (wr_en && privileged)
+    else if (wr_en)
```

For a concrete counterexample, start with `cfg_reg = 0`, keep reset inactive,
and apply `wr_en = 1`, `privileged = 0`, and `wdata = 1`. The original holds
zero. The changed implementation stores one, violating the security requirement.
A write of zero would not demonstrate a value change, so choose test data that
distinguishes hold from update.

The requirement's choice of authorization signal matters. If the branch uses
`wr_en && debug_enable`, it does not enforce a privilege requirement when the
two inputs are independent. If the actual policy instead requires
`debug_enable`, that branch can be correct. Renaming the requirement after a
failed assessment changes the specification. It does not repair the design.

## Example 2: a lock must cover every write path

**Requirement:** Outside reset, `locked = 1` must prevent both normal and debug
writes from changing `cfg_reg`. When unlocked, a debug write captures
`dbg_data` and takes priority over a normal write. If only a normal write is
requested, capture `wdata`. With neither request, hold the value.

```systemverilog
module locked_register (
  input  logic       clk, rst_n, locked, wr_en, dbg_wr,
  input  logic [7:0] wdata, dbg_data,
  output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (dbg_wr && !locked)
      cfg_reg <= dbg_data;
    else if (wr_en && !locked)
      cfg_reg <= wdata;
  end
endmodule
```

A guard on the normal path is insufficient if the debug path bypasses it:

```diff
-    else if (dbg_wr && !locked)
+    else if (dbg_wr)
```

With reset inactive, `locked = 1`, `dbg_wr = 1`, `dbg_data = 1`, and a current
value of zero, the changed branch writes one. The guarded normal branch cannot
prevent that update because the debug branch has priority.

Reset is another writer. If the requirement literally says “never change while
locked,” without a reset exemption, clearing a nonzero register during reset
can violate it. State the intended policy explicitly. Do not silently treat
reset as exempt or automatically classify that conflict as missing context.

This example covers one sequential process. In a larger design, include all
assignments to the protected state, including helper tasks and next-state
logic. A selected excerpt showing only the normal path cannot establish a
property about every writer. Multi-path analysis is also a measured model
limitation, discussed in [capabilities and limitations](LIMITATIONS.md).

## Example 3: FIFO occupancy follows accepted transfers

**Requirement:** Reset must clear `occupancy` to zero. Outside reset, one
accepted push without an accepted pop must increment occupancy by one. One
accepted pop without an accepted push must decrement it by one. If both or
neither are accepted, occupancy must hold. Occupancy must remain between zero
and four inclusive.

**Acceptance policy:** A push is accepted when `push` is asserted and the FIFO
was not full before the edge. A pop is accepted when `pop` is asserted and the
FIFO was not empty before the edge. This interface does not accept a push into
a full FIFO on the same edge as a pop, or bypass data through an empty FIFO.

```systemverilog
module fifo_occupancy (
  input  logic       clk, rst_n, push, pop,
  output logic [2:0] occupancy
);
  logic push_accept, pop_accept;
  assign push_accept = push && (occupancy < 3'd4);
  assign pop_accept = pop && (occupancy != 3'd0);

  always_ff @(posedge clk) begin
    if (!rst_n)
      occupancy <= 3'd0;
    else begin
      case ({push_accept, pop_accept})
        2'b10: occupancy <= occupancy + 3'd1;
        2'b01: occupancy <= occupancy - 3'd1;
        default: occupancy <= occupancy;
      endcase
    end
  end
endmodule
```

Requests and accepted operations are different at the boundaries:

| Occupancy before edge | Push request | Pop request | Accepted operations | Occupancy after edge |
|---:|---:|---:|---|---:|
| 2 | 1 | 0 | Push | 3 |
| 2 | 1 | 1 | Push and pop | 2 |
| 4 | 1 | 0 | Neither | 4 |
| 4 | 1 | 1 | Pop only | 3 |
| 0 | 1 | 1 | Push only | 1 |
| 0 | 0 | 1 | Neither | 0 |

Changing the push acceptance logic to ignore capacity breaks the requirement:

```diff
-  assign push_accept = push && (occupancy < 3'd4);
+  assign push_accept = push;
```

At occupancy four, a push without a pop then produces five. The rejection of
that request is part of the required interface behavior, not an assumption
that the requester must obey.

“Push and pop together must hold the count” is therefore too imprecise for this
interface. Both requests can be asserted while only one transfer is accepted.
Another FIFO may support full-boundary replacement or empty bypass, but its
acceptance policy and RTL must implement those choices consistently.

This module models occupancy only. It says nothing about stored data, ordering,
read latency, or whether the oldest word is returned. Those need separate
requirements and the corresponding storage and read-path implementation.

## Example 4: saturating versus wrapping counters

**Requirement:** Reset must clear the four-bit unsigned `count`. Outside reset,
`inc = 0` must leave it unchanged. With `inc = 1`, increment values below 15
and hold at 15. The counter must not wrap to zero.

```systemverilog
module saturating_counter (
  input  logic       clk, rst_n, inc,
  output logic [3:0] count
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      count <= 4'd0;
    else if (inc && count != 4'd15)
      count <= count + 4'd1;
  end
endmodule
```

Removing the terminal-count condition changes the behavior:

```diff
-    else if (inc && count != 4'd15)
+    else if (inc)
```

From count 15, the changed implementation increments to zero because the stored
result is four bits wide. This violates the saturating requirement. It would
satisfy a different requirement to increment modulo 16. The arithmetic is not
inherently faulty. The required overflow behavior decides which design is
correct.

Specify simultaneous clear and increment priority if a clear input is added.
Specify signedness and step size if the interface changes. Neither can be
inferred safely from the word “counter.”

## When missing context changes the verdict

Consider the privilege requirement from Example 1, but only this fragment is
available:

```systemverilog
always_ff @(posedge clk) begin
  if (!rst_n)
    cfg_reg <= 8'h00;
  else if (wr_en && permit)
    cfg_reg <= wdata;
end
```

Suppose `permit` is an internal one-bit signal whose combinational definition
has been omitted. Assume the displayed process is the only writer, and all
other context matches Example 1. Two possible definitions lead to different
answers:

| Omitted definition | Consequence for the privilege requirement |
|---|---|
| `assign permit = privileged;` | Unprivileged writes are blocked. |
| `assign permit = 1'b1;` | An unprivileged request can change the register. |

That omitted definition is material: the fragment alone cannot determine the
answer. This is the semantic basis of `insufficient_context`.

If `permit` is instead a declared, unconstrained primary input, it is allowed
to be high while `privileged` is low. That supplies a counterexample, not
missing context. An upstream guarantee relating them must be supplied and
justified before it can change the assessment.

Similarly, a visible bypass is a violation, not an omission. An unknown signal
spelling can be a syntax or integration error. An ambiguous policy needs
clarification. A low model score or unfamiliar coding style does not by itself
establish answer-changing missing information. Model predictions can still be
wrong in each of these cases.

## Prepare an input for SiliconShield

Use the three literal section headers below, without trailing colons. This
complete example asks only about privilege protection:

```text
REQUIREMENT
Outside reset, cfg_reg may change only when privileged is asserted.
Unprivileged write requests must leave cfg_reg unchanged.

ASSUMPTIONS
Inputs are two-state and sampled at rising edges of clk. Reset is synchronous,
active low, and exempt from the protection requirement. Reset is asserted for
at least one edge before normal operation. All other inputs are unconstrained.

RTL
module privileged_register (
  input  logic       clk, rst_n, wr_en, privileged,
  input  logic [7:0] wdata,
  output logic [7:0] cfg_reg
);
  always_ff @(posedge clk) begin
    if (!rst_n)
      cfg_reg <= 8'h00;
    else if (wr_en && privileged)
      cfg_reg <= wdata;
  end
endmodule
```

Keep labels, expected answers, and comments such as “faulty case” out of the
model input. Include the complete relevant logic, not a character-clipped
window. Preserve signal widths and dependency definitions. Ask separate
questions when protection, functional progress, and data ordering require
different evidence.

Use [setup](setup.md#prepare-one-complete-assessment) for input preparation and
[the router guide](router.md#run-one-assessment) for submission. Choose the scope
for the behavior being assessed. A scope tag selects relevant expertise but
does not add omitted design context or prove the requirement.

| Model label | Interpretation |
|---|---|
| `satisfies_requirement` | Predicts that the supplied implementation meets the stated requirement under its assumptions. |
| `violates_requirement` | Predicts a mismatch between the implementation and that requirement. |
| `insufficient_context` | Predicts that missing implementation or policy information can change the answer. |

These are model assessments, not simulation or formal proof. A score is not a
calibrated correctness guarantee. The examples above teach specification and
review practice, not a promise that either specialist classifies every
variant correctly. Use the [model card](siliconshield-rtl.md) and
[limitations](LIMITATIONS.md) to select the appropriate domain. Dataset labels
require the separate [solver evidence and semantic review](data-pipeline.md)
process before training admission.
