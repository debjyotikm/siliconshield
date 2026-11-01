# White-paper figure sources

These figures accompany the [SiliconShield research white paper](../../whitepaper.md).
The diagrams use editable engineering sources. SVG previews keep labels selectable
and remain sharp in the repository and in print.

| Figure | Content | Editable source | Preview |
|---|---|---|---|
| 1 | Specialist assessment architecture | [draw.io](01-system.drawio) | [SVG](01-system.svg) |
| 2 | Verified labels and dataset admission | [draw.io](02-data.drawio) | [SVG](02-data.svg) |
| 3 | Separate model ancestries | [Graphviz DOT](03-lineage.dot) | [SVG](03-lineage.svg) |
| 7 | Temporal checkpoint gates | [Python / Matplotlib](plot_temporal.py) | [SVG](07-temporal.svg) |
| 9 | Bounded research loop | [draw.io](09-feedback.drawio) | [SVG](09-feedback.svg) |

Figures 4–6 and 8 retain the paper's other research plots. The FIFO occupancy
example remains in the text, without a separate timing plot.

## Edit and render

Open a `.drawio` file in draw.io Desktop or diagrams.net. Use **File → Export as →
SVG** for the preview. Labels use explicit line breaks and plain text to avoid
HTML-only SVG content. Keep a white background. The checked-in previews were
rendered with the draw.io renderer, not redrawn approximations.

With draw.io Desktop, Graphviz, Python, Matplotlib and the Inter font installed,
run these commands from this directory:

```bash
drawio --export --format svg --output 01-system.svg 01-system.drawio
drawio --export --format svg --output 02-data.svg 02-data.drawio
dot -Tsvg 03-lineage.dot -o 03-lineage.svg
python3 plot_temporal.py
drawio --export --format svg --output 09-feedback.svg 09-feedback.drawio
```

See the official [SVG export guide](https://www.drawio.com/docs/manual/export/export-to-svg/)
for editor options. Font substitution can change line widths, so inspect the
rendered output after exporting. Retain the SVG accessibility title and description
when replacing a preview. Remove editor-generated identifiers, timestamps and
machine metadata before publication. Keep sources and previews together in a
change.

## Scientific checks

The temporal plot's source contains the seven published aggregate observations.
Its dashed lines show the final study's limits of at most 14 false-safes and at
least 50 whole triplets. The chart applies those two limits to each recorded
checkpoint for comparison. It does not imply that every earlier stage used that
policy, or that passing these two gates satisfies the full promotion policy.

Before changing a research plot, check the counts, denominators, population and
selection policy against the corresponding paper section. Preserve the distinction
between development selection and independent evaluation. Do not smooth discrete
checkpoints into a learning curve or add uncertainty intervals without supporting
data. Inspect figures at ordinary document width as well as full resolution.
