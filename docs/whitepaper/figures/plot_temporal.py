"""Render the temporal comparison from aggregate counts in the white paper.

Run from this directory with Python, Matplotlib and the Inter font installed.
The chart compares recorded stages on one development view. It is not a
controlled comparison of recipes or an independent qualification result.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Name, false-safes among 78 violations, correct whole triplets among 78.
CHECKPOINTS = [
    ("Adapter initialization", 36, 27),
    ("Earlier focused", 10, 44),
    ("Later focused", 21, 47),
    ("Low-rate control", 19, 48),
    ("Preserved parent", 16, 48),
    ("Contract-2", 17, 51),
    ("Full-core candidate", 17, 32),
]
INK = "#1E3040"
MUTED = "#647482"
PASS = "#277567"
FAIL = "#A45A35"


def main() -> None:
    """Write the comparison as a portable SVG with direct count labels."""
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "temporal-comparison",
        }
    )
    fig = plt.figure(figsize=(12.5, 7.8), facecolor="white")
    fig.text(0.035, 0.942, "07", fontsize=12, color=MUTED, weight="bold")
    fig.text(
        0.08,
        0.938,
        "Temporal checkpoints against two selection gates",
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.035,
        0.887,
        "Same 234-row development view. Read across each row to see which gate is met.",
        fontsize=13,
        color=MUTED,
    )
    fig.add_artist(
        Line2D(
            [0.035, 0.965],
            [0.856, 0.856],
            transform=fig.transFigure,
            color="#DCE4E9",
            linewidth=1,
        )
    )
    axes = [
        fig.add_axes([0.315, 0.205, 0.235, 0.525]),
        fig.add_axes([0.70, 0.205, 0.235, 0.525]),
    ]
    selected_y = 5
    for ax in axes:
        ax.set_ylim(6.6, -0.6)
        ax.set_yticks([])
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#DCE4E9")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#EBEFF2", linewidth=0.8)
        ax.tick_params(axis="x", length=0, pad=8, labelsize=11)
        ax.set_facecolor("none")

    # The selected release is highlighted across labels and both panels.
    y_bottom = (
        axes[0].transData.transform((0, selected_y + 0.45))[1] / fig.bbox.height
    )
    y_top = (
        axes[0].transData.transform((0, selected_y - 0.45))[1] / fig.bbox.height
    )
    fig.add_artist(
        Rectangle(
            (0.028, y_bottom),
            0.949,
            y_top - y_bottom,
            transform=fig.transFigure,
            facecolor="#EDF3F7",
            edgecolor="none",
            zorder=-1,
        )
    )
    for i, (name, fs, triplets) in enumerate(CHECKPOINTS):
        y = axes[0].transData.transform((0, i))[1] / fig.bbox.height
        fig.text(
            0.035,
            y,
            name,
            va="center",
            fontsize=13,
            weight="bold" if name == "Contract-2" else "normal",
        )
        for ax, value, passed, status_x in [
            (axes[0], fs, fs <= 14, 0.59),
            (axes[1], triplets, triplets >= 50, 0.957),
        ]:
            color = PASS if passed else FAIL
            ax.barh(i, value, height=0.40, color=color, alpha=0.82)
            ax.text(
                value + 0.7,
                i,
                str(value),
                va="center",
                fontsize=12,
                weight="bold",
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
            )
            fig.text(
                status_x,
                y,
                "PASS" if passed else "FAIL",
                va="center",
                ha="center",
                fontsize=10,
                weight="bold",
                color=color,
            )
    axes[0].set_xlim(0, 42)
    axes[0].set_xticks([0, 10, 20, 30, 40])
    axes[1].set_xlim(0, 82)
    axes[1].set_xticks([0, 20, 40, 60, 78])
    for ax, limit in zip(axes, [14, 50]):
        ax.axvline(limit, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
    fig.text(
        0.315, 0.81, "A  False-safe violations", fontsize=14, weight="bold"
    )
    fig.text(
        0.315,
        0.77,
        "At most 14 of 78 · fewer is better",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.70, 0.81, "B  Whole triplets correct", fontsize=14, weight="bold"
    )
    fig.text(
        0.70,
        0.77,
        "At least 50 of 78 · more is better",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.118,
        "Contract-2 meets the triplet gate, but exceeds the false-safe limit by 3.",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.035,
        0.065,
        "No checkpoint shown meets both gates. Class, NLL and domain floors are additional requirements.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.035,
        0.032,
        "Different training stages and recipes. These points are not independent repeats or a controlled ablation.",
        fontsize=11,
        color=MUTED,
    )
    output = Path(__file__).resolve().parent / "07-temporal.svg"
    fig.savefig(output, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    svg = output.read_text()
    svg = svg.replace(
        "font-family: 'Inter'", "font-family: 'Inter', Arial, sans-serif"
    )
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S)
    for i, identifier in enumerate(
        dict.fromkeys(re.findall(r'id="([mp][0-9a-f]{8,})"', svg))
    ):
        svg = svg.replace(identifier, f"chart-resource-{i + 1}")
    svg = re.sub(
        r"(<svg\b[^>]*)(>)",
        r'\1 role="img" aria-labelledby="chart-title chart-description"\2',
        svg,
        count=1,
    )
    svg = re.sub(
        r"(<svg\b[^>]*>)",
        lambda m: m.group(1)
        + '<title id="chart-title">Temporal checkpoints against two selection gates</title>'
        '<desc id="chart-description">Aligned rows compare false-safes and whole triplets. '
        "Contract-2 has 17 false-safes, failing a maximum of 14, and 51 whole triplets, "
        "passing a minimum of 50. No checkpoint shown passes both gates.</desc>",
        svg,
        count=1,
    )
    svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    ET.fromstring(svg)
    output.write_text(svg)
    print(output.name)


if __name__ == "__main__":
    main()
