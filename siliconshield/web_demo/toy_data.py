"""Small authored workflow fixture. Never claimed as model qualification."""

import json
from pathlib import Path
from typing import Any

from siliconshield.training.laya.question_contracts import (
    NEW_CONTRACT,
    question_for,
)

ASSUMPTIONS = (
    "Two-state synchronous logic. Reset is exempt from the requirement. "
    "All primary inputs are defined and may vary independently. An undriven policy "
    "wire represents an omitted side-effect-free Boolean expression over the shown inputs."
)


def create_dataset(root: Path) -> dict[str, Any]:
    """Write a balanced, authored fixture for operator integration only."""
    root.mkdir(parents=True, exist_ok=False)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": []}
    for i, (width, style) in enumerate(
        [(4, 0), (8, 0), (12, 0), (4, 1), (8, 1), (12, 1), (6, 0), (10, 1)]
    ):
        split = "train" if i < 6 else "dev"
        group = f"web-toy-register-{i:02d}"
        for kind, guard, label in [
            ("guarded", "write_en && privileged", "satisfies_requirement"),
            ("unguarded", "write_en", "violates_requirement"),
            ("omitted", "write_en && policy_allow", "insufficient_context"),
        ]:
            declaration = "  wire policy_allow;\n" if kind == "omitted" else ""
            update = f"    else if ({guard}) cfg <= write_data;"
            if style:
                inner = {
                    "guarded": "privileged",
                    "unguarded": "1'b1",
                    "omitted": "policy_allow",
                }[kind]
                update = f"    else if (write_en) begin\n      if ({inner}) cfg <= write_data;\n    end"
            rtl = (
                f"module topmodule(input clk, rst_n, write_en, privileged,\n"
                f"  input [{width - 1}:0] write_data, output reg [{width - 1}:0] cfg);\n"
                f"{declaration}  always @(posedge clk) begin\n"
                f"    if (!rst_n) cfg <= {width}'d0;\n{update}\n  end\nendmodule"
            )
            requirement = "Outside reset, the configuration register may change only when the writer is privileged."
            splits[split].append(
                {
                    "id": group + "-" + kind,
                    "family_id": group,
                    "split_group_id": group,
                    "split": split,
                    "pair_id": group,
                    "base_design_id": group,
                    "mechanism_id": "toy-write-authorization",
                    "source": {"name": "toy-synthetic"},
                    "state": f"REQUIREMENT\n{requirement}\n\nASSUMPTIONS\n{ASSUMPTIONS}\n\nRTL\n{rtl}",
                    "questions": question_for(NEW_CONTRACT),
                    "expected": {"security": label},
                    "validation": {
                        "accepted": True,
                        "method": "authored-operability-fixture-not-solver-qualified",
                    },
                }
            )
    for name, rows in splits.items():
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    (root / "pools.json").write_text(
        json.dumps(
            {
                "pools": {
                    "toy": {
                        "rows": [r["id"] for r in splits["train"]],
                        "weight": 1.0,
                        "mode": "flat",
                    }
                }
            },
            indent=2,
        )
    )
    return {
        "train": 18,
        "dev": 6,
        "triplets": 8,
        "label_counts": {"train": [6, 6, 6], "dev": [2, 2, 2]},
        "scope": "Authored integration fixture. Family IDs are disjoint, but widths and renderings are related. Not independent model qualification.",
    }
