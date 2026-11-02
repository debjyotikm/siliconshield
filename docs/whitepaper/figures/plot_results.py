"""Render white-paper Figures 4–6 and 8 from published aggregate values.

Run from any directory with Matplotlib and NumPy installed. No research files,
model weights or machine configuration are read. Output is deterministic SVG.
"""

import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent
DATA = {
    "contract_one": {
        "n": [288, 288, 756],
        "class_n": [96, 96, 252],
        "correct": [257, 249, 330],
        "vio_correct": [95, 92, 225],
        "context_correct": [66, 76, 25],
        "triplets": [66, 59, 12],
        "confusions": [
            [[81, 0, 15], [0, 92, 4], [16, 4, 76]],
            [[80, 159, 13], [25, 225, 2], [49, 178, 25]],
        ],
    },
    "preservation": {
        "ratio_percent": [0, 5.20, 9.53, 13.07, 11.79],
        "adapter_displacement": [0.0811, 0.0750, 0.0679, 0.0600, 0.0424],
    },
}
INK = "#172635"
MUTED = "#53616D"
GRID = "#DBE2E7"
BLUE = "#286181"
TEAL = "#257568"
RUST = "#AA5833"
PALE = "#F4F7F9"
plt.rcParams.update(
    {
        "font.family": "Inter",
        "font.size": 14,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "svg.fonttype": "none",
        "svg.hashsalt": "siliconshield-public-figures",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def chart(height: float, number: int, title: str, subtitle: str) -> Figure:
    """Create the shared research-figure header and canvas."""
    fig = plt.figure(figsize=(12.5, height))
    pixels = height * 96
    fig.text(
        0.035,
        1 - 48 / pixels,
        f"{number:02d}",
        fontsize=12,
        color=MUTED,
        weight="semibold",
    )
    fig.text(0.080, 1 - 48 / pixels, title, fontsize=21, weight="semibold")
    fig.text(0.035, 1 - 85 / pixels, subtitle, fontsize=13, color=MUTED)
    fig.add_artist(
        plt.Line2D(
            [0.035, 0.965],
            [1 - 108 / pixels] * 2,
            transform=fig.transFigure,
            color=GRID,
            lw=0.8,
        )
    )
    return fig


def foot(fig: Figure, text: str) -> None:
    """Add a scope note below the measurement panels."""
    fig.add_artist(
        plt.Line2D(
            [0.035, 0.965],
            [0.086, 0.086],
            transform=fig.transFigure,
            color=GRID,
            lw=0.8,
        )
    )
    fig.text(0.035, 0.040, text, fontsize=11.5, color=MUTED)


def save_chart(fig: Figure, name: str, title: str, description: str) -> None:
    """Save accessible vector output without machine or timestamp metadata."""
    # Native SVG: no author/date metadata, raster heatmaps, or opaque element IDs.
    path = OUT / f"{name}.svg"
    fig.savefig(path, metadata={"Date": None, "Creator": None})
    raw = path.read_text()
    raw = re.sub(r"\s*<metadata>.*?</metadata>", "", raw, flags=re.S)
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    identifiers = re.findall(r'id="([^"]+)"', raw)
    count = 0
    for ident in identifiers:
        if re.search(r"[0-9a-f]{7,}", ident):
            count += 1
            new = f"drawing-resource-{count}"
            raw = raw.replace(f'id="{ident}"', f'id="{new}"').replace(
                f"#{ident}", f"#{new}"
            )
    raw = re.sub(
        r"(<svg\b[^>]*)(>)",
        r'\1 role="img" aria-labelledby="title description"\2',
        raw,
        count=1,
    )
    insert = f'<title id="title">{html.escape(title)}</title>\n<desc id="description">{html.escape(description)}</desc>'
    raw = re.sub(
        r"(<svg\b[^>]*>)", lambda m: m.group(1) + "\n" + insert, raw, count=1
    )
    raw = "\n".join(line.rstrip() for line in raw.splitlines()) + "\n"
    ET.fromstring(raw)
    path.write_text(raw)
    plt.close(fig)


def main() -> None:
    """Regenerate the four plots from the values documented in the paper."""
    # Figure 4: identical axes, directly labeled values, explicit denominators.
    c = DATA["contract_one"]
    fig = chart(
        6.5,
        4,
        "Contract-1 across evaluation populations",
        "Each panel uses the same 0–100% scale. Development and reserved evaluations answer different questions.",
    )
    metrics = [
        "Accuracy",
        "Violation recall",
        "Context recall",
        "Whole triplets",
    ]
    for k, (label, color) in enumerate(
        zip(
            ["Development", "Qualification", "Shifted evaluation"],
            [BLUE, TEAL, RUST],
        )
    ):
        x0 = 0.158 + k * 0.275
        ax = fig.add_axes([x0, 0.225, 0.205, 0.485])
        nums = [
            c["correct"][k],
            c["vio_correct"][k],
            c["context_correct"][k],
            c["triplets"][k],
        ]
        denoms = [c["n"][k]] + [c["class_n"][k]] * 3
        values = np.array(nums) / denoms * 100
        for j, (val, n, den) in enumerate(zip(values, nums, denoms)):
            y = 3 - j
            ax.hlines(y, 0, 100, color=GRID, lw=1)
            ax.plot(val, y, "o", color=color, ms=7, zorder=3)
            ax.text(
                106,
                y,
                f"{val:.1f}%",
                fontsize=12.5,
                va="center",
                weight="semibold",
                color=color,
                clip_on=False,
            )
            ax.text(0, y - 0.25, f"{n:,} / {den:,}", fontsize=11, color=MUTED)
        ax.set(
            xlim=(-2, 103),
            ylim=(-0.55, 3.45),
            xticks=[0, 50, 100],
            yticks=[3, 2, 1, 0],
        )
        ax.set_yticklabels(metrics if k == 0 else [""] * 4)
        ax.tick_params(axis="y", length=0, pad=16)
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Percent", fontsize=12, labelpad=12)
        fig.text(
            x0,
            0.782,
            f"{chr(65 + k)}  {label}",
            fontsize=15,
            weight="semibold",
            color=color,
        )
        fig.text(
            x0,
            0.742,
            f"{c['n'][k]:,} decisions · {c['class_n'][k]} per class",
            fontsize=11.5,
            color=MUTED,
        )
    foot(
        fig,
        "Whole-triplet scores require all three labels to be correct. Counts below each mark show the relevant support.",
    )
    save_chart(
        fig,
        "04-contract-one",
        "Contract-1 across evaluation populations",
        "Three panels compare accuracy, violation recall, context recall, and complete triplets. All values are recorded aggregates, with exact counts.",
    )

    # Figure 5: pure vector heatmaps with row proportions and false-safe cells.
    fig = chart(
        7.6,
        5,
        "Where Contract-1 makes its errors",
        "Counts and row percentages distinguish false alarms, false-safes, and missed semantic context.",
    )
    cmap = LinearSegmentedColormap.from_list("research-blue", ["#F3F7FA", BLUE])
    for k, (matrix, title, n) in enumerate(
        zip(c["confusions"], ["Qualification", "Shifted evaluation"], [96, 252])
    ):
        x0 = 0.11 + k * 0.46
        ax = fig.add_axes([x0, 0.31, 0.31, 0.43])
        a = np.array(matrix)
        rows = a / a.sum(axis=1)[:, None]
        for i in range(3):
            for j in range(3):
                share = rows[i, j]
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=cmap(share),
                        edgecolor="white",
                        lw=2,
                    )
                )
                color = "white" if share > 0.75 else INK
                ax.text(
                    j,
                    i - 0.07,
                    str(a[i, j]),
                    ha="center",
                    va="center",
                    fontsize=22,
                    weight="semibold",
                    color=color,
                )
                ax.text(
                    j,
                    i + 0.24,
                    f"{share:.1%}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                )
        ax.add_patch(
            Rectangle(
                (-0.48, 0.52), 0.96, 0.96, fill=False, edgecolor=RUST, lw=2.2
            )
        )
        ax.set(
            xlim=(-0.5, 2.5),
            ylim=(2.5, -0.5),
            xticks=[0, 1, 2],
            yticks=[0, 1, 2],
        )
        ax.set_xticklabels(["S", "V", "I"])
        ax.set_yticklabels(["S", "V", "I"])
        ax.tick_params(length=0, pad=9)
        ax.set_aspect("equal")
        ax.set_xlabel("Predicted label", labelpad=12)
        ax.set_ylabel("True label", labelpad=10)
        fig.text(
            x0, 0.809, f"{chr(65 + k)}  {title}", fontsize=16, weight="semibold"
        )
        fig.text(
            x0,
            0.772,
            f"{n * 3} decisions · {n} per true class",
            fontsize=12,
            color=MUTED,
        )
    cax = fig.add_axes([0.73, 0.14, 0.17, 0.014])
    bar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 100), cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 50, 100],
    )
    bar.solids.set_rasterized(False)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=10)
    fig.text(0.73, 0.18, "Share of true class (%)", fontsize=10.5, color=MUTED)
    fig.text(
        0.075,
        0.16,
        "S  Satisfies     V  Violates     I  Insufficient context",
        fontsize=12,
        color=MUTED,
    )
    foot(
        fig,
        "The outlined V → S cell is the false-safe error: a violating implementation predicted satisfying.",
    )
    save_chart(
        fig,
        "05-confusion",
        "Contract-1 confusion matrices",
        "Qualification and shifted evaluations have independent row-normalized color scales fixed to zero through one hundred percent. False-safe cells are outlined.",
    )

    # Figure 6: two different diagnostics, with their denominators kept separate.
    fig = chart(
        6.5,
        6,
        "Training fit does not establish transfer",
        "Two recorded experiments expose a large fitting-to-development gap. The panels are not a model comparison.",
    )
    gaps = [
        ("A  Temporal pair discrimination", [379, 3], [432, 108]),
        ("B  Coherent terminal decisions", [7079, 215], [7101, 324]),
    ]
    for k, (title, nums, denoms) in enumerate(gaps):
        x0 = 0.13 + k * 0.46
        ax = fig.add_axes([x0, 0.265, 0.32, 0.40])
        for j, (num, den, color) in enumerate(zip(nums, denoms, [BLUE, RUST])):
            y = 1 - j
            val = 100 * num / den
            ax.barh(y, val, height=0.34, color=color, alpha=0.92)
            ax.text(0, y + 0.27, f"{num:,} / {den:,}", fontsize=12, color=MUTED)
            ax.text(
                val + 2,
                y,
                f"{val:.2f}%",
                va="center",
                fontsize=13,
                weight="semibold",
                color=color,
                clip_on=False,
            )
        ax.set(
            xlim=(0, 105),
            ylim=(-0.5, 1.6),
            yticks=[1, 0],
            xticks=[0, 25, 50, 75, 100],
        )
        ax.set_yticklabels(["TRAIN", "DEV"])
        ax.tick_params(length=0, pad=8)
        ax.set_xlabel("Percent correct", labelpad=12)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        fig.text(x0 - 0.06, 0.785, title, fontsize=15, weight="semibold")
    foot(
        fig,
        "Panel A measures complete S/V pairs. Panel B measures individual three-way decisions from one coherent export.",
    )
    save_chart(
        fig,
        "06-learning-gap",
        "Training fit versus development transfer",
        "Separate experiments: temporal training pairs 379 of 432 versus development pairs 3 of 108. Coherent terminal TRAIN accuracy 7079 of 7101 versus development 215 of 324.",
    )

    # Figure 8: behavioral constraint and loss share on the same tested sweep.
    p = DATA["preservation"]
    fig = chart(
        6.7,
        8,
        "Loss share is not preservation strength",
        "Five 50-update TRAIN-only probes, summarized after warmup. No accuracy claim follows from this sweep.",
    )
    specs = [
        (
            "A  Weighted KL / CE",
            "Percent",
            p["ratio_percent"],
            RUST,
            (-0.6, 16),
            [0, 5, 10, 15],
            lambda x: f"{x:.2f}",
        ),
        (
            "B  Adapter displacement",
            "L₂ norm",
            p["adapter_displacement"],
            BLUE,
            (-0.004, 0.100),
            [0, 0.025, 0.05, 0.075, 0.10],
            lambda x: f"{x:.4f}",
        ),
        (
            "C  Teacher agreement",
            "Percent",
            [np.nan, 96.7, 97.2, 97.6, 99.2],
            TEAL,
            (95, 100.7),
            [95, 97.5, 100],
            lambda x: f"{x:.1f}",
        ),
    ]
    for k, (title, ylabel, vals, color, ylim, ticks, fmt) in enumerate(specs):
        x0 = 0.08 + k * 0.31
        ax = fig.add_axes([x0, 0.30, 0.235, 0.40])
        xx = np.arange(5)
        ax.scatter(xx, vals, color=color, s=48, zorder=3)
        for x, y in zip(xx, vals):
            if np.isfinite(y):
                ax.annotate(
                    fmt(y),
                    (x, y),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10.8,
                    color=color,
                )
        ax.set(
            xlim=(-0.5, 4.5),
            ylim=ylim,
            xticks=xx,
            xticklabels=["0", ".41", "1", "2", "8"],
            yticks=ticks,
        )
        ax.set_xlabel("Preservation weight", labelpad=12, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7)
        fig.text(
            x0 - 0.015,
            0.775,
            title,
            fontsize=14.5,
            weight="semibold",
            color=color,
        )
    fig.text(
        0.08,
        0.145,
        "Discrete tested weights are equally spaced. Teacher agreement was not recorded for W = 0.",
        fontsize=12,
        color=MUTED,
    )
    foot(
        fig,
        "W = 8 produces the smallest adapter displacement and highest teacher agreement, despite a lower loss ratio than W = 2.",
    )
    save_chart(
        fig,
        "08-preservation",
        "Preservation loss and behavioral diagnostics",
        "The same five tested weights are shown in three panels: weighted KL divided by CE, adapter displacement, and teacher-correct agreement. Each point is an observed aggregate. No curve is fitted.",
    )


if __name__ == "__main__":
    main()
