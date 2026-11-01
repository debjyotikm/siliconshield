# Architecture figure sources

[Documentation](../README.md) / Architecture figures

The README uses a compact product overview, with the detailed specifications
linked separately. Each engineering figure answers a different question. A
component map shows ownership and process boundaries, a sequence diagram shows
message ordering, and a lifecycle
view shows persisted states and transition effects. Each is referenced by its
figure number in the corresponding specification.

## Product overview

[Product workflows](product-workflows.svg) is the front-page figure. Two lanes
show the path from an RTL requirement to a finding, and from reviewed training
data to a candidate selection. The training feedback arrow shows Muse proposing
the next experiment within the study budget. Assessment and model development
remain separate workflows.

The editable source is [product-workflows.drawio](product-workflows.drawio),
with an SVG rendered from its native shapes and connectors. Both use the same
monochrome palette as the detailed figures. Detailed validation and recovery
steps belong in the architecture specifications linked below.

## Detailed architecture

| Figure | View | Preview | Owning specification |
|---|---|---|---|
| 01 | Runtime components and interfaces I01–I04 | [SVG](system-overview.svg) | [System architecture](../architecture.md) |
| 02 | Expert assessment request/reply sequence | [SVG](expert-assessment.svg) | [Router](../router.md#assessment-transaction) |
| 03 | Persisted training-round lifecycle and promotion | [SVG](decision-study.svg) | [Campaigns](../campaigns.md#round-lifecycle) |

**Editable source:** [siliconshield-architecture.drawio](siliconshield-architecture.drawio)
contains all three pages as native editable shapes and connectors. Open it in
diagrams.net. The SVG previews are rendered from those page models. No private run paths, model weights, or experiment
records are embedded in either format.

## Notation

All figures use black and gray on white. Light gray fills separate containers
from the page without assigning meaning through color. Labels, interface IDs,
arrow direction, and line patterns carry the technical meaning.

Figure 01 uses enclosing frames for execution/configuration boundaries. Solid
connectors show local execution and result flow. Dashed connectors show
configuration or optional integration, as named in the interface schedule.

Figure 02 uses vertical participant lifelines. Solid arrows are calls, dashed
arrows are replies, and the loop frame represents sequential expert consultation.
The optional frame is a caller-controlled arbitration step.

Figure 03 uses the literal campaign phase names. Descriptions under each state
identify its handler's work. State persistence, job submission, and promotion
are different events. The table in the campaign guide supplies exact transition
semantics and explains refusals that leave a nonterminal phase intact.

## Editing and export

1. Edit the relevant page in the draw.io source and verify its semantics against
   the owning code and tests. Keep interface IDs and phase names consistent with
   the specification.
2. Keep the grayscale palette and export to SVG with a white background.
   Preserve text as text. Use Inter with Arial or a sans-serif fallback, and
   keep detailed labels readable at normal documentation width. Avoid external
   images, embedded scripts, and HTML-only labels in the exported SVG.
3. Update the matching preview in the table above. Keep names stable so existing
   links continue to work.
4. Inspect both the full-size SVG and its rendering at documentation width.
   Check arrow destinations, loop/optional frames, text overflow, and labels.
5. Validate XML, local links, and the export allowlist. Commit source and preview
   together. Temporary PNGs and browser previews stay outside Git.

These are design views of implemented interfaces, not measured performance
charts. Add a separate qualified figure for a new result rather than placing
live training metrics into the architecture diagrams.
