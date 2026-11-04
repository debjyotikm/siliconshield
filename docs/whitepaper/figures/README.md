# White-paper figure sources

These figures accompany the [SiliconShield research white paper](../../whitepaper.md).

| Figure | Content | Editable source | Preview |
|---|---|---|---|
| 1 | Specialist assessment architecture | [draw.io](01-system.drawio) | [SVG](01-system.svg) |
| 2 | Supervised decision adaptation | [Graphviz DOT](02-method.dot) | [SVG](02-method.svg) |
| 3 | Frozen-candidate development-monitor results | SVG preview only | [SVG](03-monitor.svg) |
| 4 | Recorded training loss through the frozen checkpoint | SVG preview only | [SVG](04-training.svg) |
| 5 | Historical specialist results and the frozen adapter | SVG preview only | [SVG](05-comparison.svg) |

The Matplotlib previews (figures 3-5) plot the aggregate values already
reported in the paper. Their render scripts read private run records and are
not shipped. The checked-in SVGs contain no private paths or measurements
beyond the published aggregates.

## Edit and render

Open `01-system.drawio` in draw.io Desktop or diagrams.net. Use **File → Export
as → SVG** for the preview. Labels use explicit line breaks and plain text to
avoid HTML-only SVG content. Keep a white background.

Render `02-method.dot` with Graphviz:

```bash
dot -Tsvg 02-method.dot -o 02-method.svg
```
