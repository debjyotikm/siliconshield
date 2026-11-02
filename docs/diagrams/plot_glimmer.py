"""Render the local Glimmer case-study charts from recorded aggregate counts.

The arrays reproduce the published case-study observations. They are not raw
training rows. Run this file to regenerate the three SVG previews in place.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent
NAVY = "#1f4e79"
TEAL = "#2e7d6f"
RUST = "#b85450"
GRAY = "#6b7280"
LIGHT = "#e5e7eb"


def svg_open(title: str, desc: str, width: int, height: int) -> str:
    """Open a portable SVG with an accessible title and description."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'style="background: #ffffff;" version="1.1" '
        f'width="{width}px" height="{height}px" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="figure-title figure-description">\n'
        f'  <title id="figure-title">{title}</title>\n'
        f'  <desc id="figure-description">{desc}</desc>\n'
    )


def text(
    x: float,
    y: float,
    content: str,
    size: int = 13,
    anchor: str = "middle",
    fill: str = "#111827",
    bold: bool = False,
) -> str:
    """Render a positioned label using the shared figure typography."""
    weight = ' font-weight="bold"' if bold else ""
    return (
        f'  <text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="sans-serif" font-size="{size}" '
        f'fill="{fill}"{weight}>{content}</text>\n'
    )


def toy_parity() -> str:
    """Grouped trainer versus SDK correctness bars for Exhibit A."""
    width, height = 760, 432
    parts = [
        svg_open(
            "Exhibit A trainer and SDK correctness",
            "Trainer and SDK correct counts for the parent, the fitting "
            "export, and the development export on TRAIN and dev rows. "
            "The children match across paths. The parent does not.",
            width,
            height,
        )
    ]
    parts.append(
        text(
            380,
            30,
            "Exhibit A: trainer and SDK correctness",
            size=16,
            bold=True,
        )
    )
    left, top, plot_w, plot_h = 60, 60, 660, 280
    ymax = 18
    groups = [
        (
            "TRAIN (n = 18)",
            [("Parent", 13, 11), ("Fitting", 18, 18), ("Selected", 13, 13)],
        ),
        (
            "Dev (n = 6)",
            [("Parent", 3, 4), ("Fitting", None, 6), ("Selected", 6, 6)],
        ),
    ]
    parts.append(
        f'  <line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top + plot_h}" stroke="#111827"/>\n'
    )
    for value in (0, 6, 12, 18):
        y = top + plot_h - value / ymax * plot_h
        parts.append(
            f'  <line x1="{left}" y1="{y}" '
            f'x2="{left + plot_w}" y2="{y}" stroke="{LIGHT}"/>\n'
        )
        parts.append(
            text(left - 10, y + 4, str(value), anchor="end", fill=GRAY)
        )
    parts.append(
        '<text x="18" y="200" text-anchor="middle" '
        'transform="rotate(-90 18 200)" font-family="sans-serif" '
        'font-size="13" fill="#6b7280">Correct rows</text>\n'
    )
    group_w = plot_w / 2
    for gi, (glabel, exports) in enumerate(groups):
        gx = left + gi * group_w
        parts.append(
            text(gx + group_w / 2, top + plot_h + 50, glabel, bold=True)
        )
        slot = group_w / 3
        for ei, (ename, trainer, sdk) in enumerate(exports):
            cx = gx + ei * slot + slot / 2
            parts.append(text(cx, top + plot_h + 22, ename, size=12))
            for bi, (label, value, color) in enumerate(
                (("T", trainer, NAVY), ("S", sdk, TEAL))
            ):
                bx = cx - 26 + bi * 28
                if value is None:
                    parts.append(
                        text(
                            cx + 14, top + plot_h - 6, "n/a", size=11, fill=GRAY
                        )
                    )
                    parts.append(
                        text(
                            bx + 12,
                            top + plot_h + 38,
                            label,
                            size=11,
                            fill=GRAY,
                        )
                    )
                    continue
                bh = value / ymax * plot_h
                parts.append(
                    f'  <rect x="{bx}" y="{top + plot_h - bh}" '
                    f'width="24" height="{bh}" fill="{color}"/>\n'
                )
                parts.append(
                    text(
                        bx + 12,
                        top + plot_h - bh - 6,
                        str(value),
                        size=12,
                        bold=True,
                    )
                )
                parts.append(
                    text(bx + 12, top + plot_h + 38, label, size=11, fill=GRAY)
                )
    parts.append(
        text(
            380,
            420,
            "T is the trainer path. S is the SDK "
            "replay. Fitting has no trainer dev score.",
            size=12,
            fill=GRAY,
        )
    )
    parts.append("</svg>\n")
    return "".join(parts)


TRAJ_STEPS = [0] + [10 * i for i in range(1, 19)]
TRAJ_CORRECT = [
    203,
    200,
    201,
    199,
    201,
    201,
    199,
    199,
    199,
    197,
    199,
    203,
    203,
    202,
    202,
    202,
    202,
    203,
    200,
]
TRAJ_FS = [
    17,
    18,
    17,
    19,
    15,
    17,
    16,
    18,
    17,
    17,
    18,
    15,
    16,
    16,
    17,
    16,
    16,
    16,
    18,
]
TRAJ_TRIP = [
    51,
    49,
    50,
    48,
    49,
    51,
    48,
    49,
    47,
    46,
    48,
    51,
    52,
    52,
    52,
    51,
    51,
    51,
    49,
]


TRAJ_SAT = [
    64,
    62,
    62,
    62,
    60,
    62,
    59,
    61,
    60,
    58,
    61,
    62,
    63,
    62,
    63,
    62,
    62,
    63,
    62,
]


def trajectory() -> str:
    """Four stacked development panels for Exhibit B."""
    width = 760
    panel_h, gap = 150, 44
    top = 56
    height = top + 4 * panel_h + 3 * gap + 40
    parts = [
        svg_open(
            "Exhibit B development trajectory",
            "Correct decisions, false-safe errors, and whole triplets "
            "and satisfies-class correct counts across 18 development evaluations. Floor lines mark the "
            "frozen parent counts. No evaluation passes all floors.",
            width,
            height,
        )
    ]
    parts.append(
        text(380, 30, "Exhibit B: development trajectory", size=16, bold=True)
    )
    left, plot_w = 60, 620
    panels = [
        (
            "Correct decisions (n = 234)",
            TRAJ_CORRECT,
            203,
            196,
            204,
            "floor 203",
        ),
        (
            "False-safe errors (lower is better)",
            TRAJ_FS,
            17,
            14,
            20,
            "ceiling 17",
        ),
        ("Whole triplets (n = 78)", TRAJ_TRIP, 51, 45, 53, "floor 51"),
        (
            "Satisfies correct (n = 78): every trained point fails",
            TRAJ_SAT,
            64,
            57,
            65,
            "floor 64",
        ),
    ]
    for pi, (label, values, gate, lo, hi, gate_label) in enumerate(panels):
        py = top + pi * (panel_h + gap)
        parts.append(text(left + plot_w / 2, py - 10, label, bold=True))
        parts.append(
            f'  <line x1="{left}" y1="{py}" x2="{left}" '
            f'y2="{py + panel_h}" stroke="#111827"/>\n'
        )
        parts.append(
            f'  <line x1="{left}" y1="{py + panel_h}" '
            f'x2="{left + plot_w}" y2="{py + panel_h}" '
            'stroke="#111827"/>\n'
        )
        span = hi - lo

        def x_of(step: int) -> float:
            return left + step / 180 * plot_w

        def y_of(value: int) -> float:
            return py + panel_h - (value - lo) / span * panel_h

        for tick in range(lo, hi + 1, 2):
            parts.append(
                f'  <line x1="{left}" y1="{y_of(tick)}" '
                f'x2="{left + plot_w}" y2="{y_of(tick)}" '
                f'stroke="{LIGHT}"/>\n'
            )
            parts.append(
                text(
                    left - 8, y_of(tick) + 4, str(tick), anchor="end", fill=GRAY
                )
            )
        gy = y_of(gate)
        parts.append(
            f'  <line x1="{left}" y1="{gy}" x2="{left + plot_w}" '
            f'y2="{gy}" stroke="{RUST}" stroke-dasharray="6,4" '
            'stroke-width="1.5"/>\n'
        )
        parts.append(
            text(
                left + plot_w + 8,
                gy + 4,
                gate_label,
                anchor="start",
                size=12,
                fill=RUST,
            )
        )
        cx = x_of(114)
        parts.append(
            f'  <line x1="{cx}" y1="{py}" x2="{cx}" '
            f'y2="{py + panel_h}" stroke="{GRAY}" '
            'stroke-dasharray="3,3"/>\n'
        )
        if pi == 0:
            parts.append(
                text(
                    cx,
                    py + panel_h - 8,
                    "full coverage at 114",
                    size=11,
                    fill=GRAY,
                )
            )
        dots = " ".join(
            f"{x_of(s):.1f},{y_of(v):.1f}" for s, v in zip(TRAJ_STEPS, values)
        )
        parts.append(
            f'  <polyline points="{dots}" fill="none" '
            f'stroke="{NAVY}" stroke-width="2"/>\n'
        )
        for s, v in zip(TRAJ_STEPS, values):
            parts.append(
                f'  <circle cx="{x_of(s):.1f}" cy="{y_of(v):.1f}" '
                f'r="3" fill="{NAVY}"/>\n'
            )
        if pi == 3:
            for s in range(0, 181, 30):
                parts.append(
                    text(x_of(s), py + panel_h + 20, str(s), fill=GRAY)
                )
            parts.append(
                text(left + plot_w / 2, py + panel_h + 38, "Update", fill=GRAY)
            )
    parts.append("</svg>\n")
    return "".join(parts)


def deviated_deltas() -> str:
    """Step-20 minus parent deltas under both eval paths."""
    width, height = 640, 360
    parts = [
        svg_open(
            "Deviated observation deltas",
            "Step-20 weights minus the parent under the trainer path "
            "and the SDK path. All bars point up, so the deviated "
            "observation improves the three reported metrics in both paths.",
            width,
            height,
        )
    ]
    parts.append(
        text(320, 30, "Deviated observation minus parent", size=16, bold=True)
    )
    left, top, plot_w, plot_h = 60, 70, 520, 200
    ymax = 2.5
    zero = top + plot_h - 0.4 / (ymax + 0.4) * plot_h
    parts.append(
        f'  <line x1="{left}" y1="{zero}" '
        f'x2="{left + plot_w}" y2="{zero}" '
        'stroke="#111827"/>\n'
    )
    groups = [
        ("Trainer path", [("Correct", 1), ("Fewer FS", 1), ("Triplets", 1)]),
        ("SDK path", [("Correct", 2), ("Fewer FS", 2), ("Triplets", 1)]),
    ]
    group_w = plot_w / 2
    for gi, (glabel, bars) in enumerate(groups):
        gx = left + gi * group_w
        parts.append(
            text(gx + group_w / 2, top + plot_h + 44, glabel, bold=True)
        )
        slot = group_w / 3
        for bi, (blabel, value) in enumerate(bars):
            cx = gx + bi * slot + slot / 2
            bh = value / (ymax + 0.4) * plot_h
            parts.append(
                f'  <rect x="{cx - 24}" y="{zero - bh}" width="48" '
                f'height="{bh}" fill="{TEAL}"/>\n'
            )
            parts.append(
                text(cx, zero - bh - 8, f"+{value}", size=13, bold=True)
            )
            parts.append(text(cx, zero + 20, blabel, size=12))
    parts.append(
        text(
            320,
            336,
            "FS is false-safe errors. Up is better on all three bars.",
            size=12,
            fill=GRAY,
        )
    )
    parts.append("</svg>\n")
    return "".join(parts)


def main() -> None:
    """Regenerate the three accessible SVG previews in this directory."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "glimmer-toy-parity.svg").write_text(toy_parity())
    (OUT / "glimmer-improvement-trajectory.svg").write_text(trajectory())
    (OUT / "glimmer-deviated-deltas.svg").write_text(deviated_deltas())
    print("wrote three figures to", OUT)


if __name__ == "__main__":
    main()
