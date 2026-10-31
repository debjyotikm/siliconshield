"""Per-contract solver bounds and evidence binding."""

import pytest

from siliconshield.training.laya.build_security_data import solver_commands
from siliconshield.training.laya.checker_adapter import load_contracts


def test_solver_commands_honor_declared_bound():
    bounded, induction = solver_commands("assert(1);", "rst_n")
    assert "sat -seq 6" in bounded
    assert "maxsteps 12" in induction
    custom_bounded, custom_induction = solver_commands(
        "assert(1);", "rst_n", bounded_cycles=10, induction_steps=20
    )
    assert "sat -seq 10" in custom_bounded
    assert "maxsteps 20" in custom_induction
    with pytest.raises(ValueError, match="bounded_cycles"):
        solver_commands("assert(1);", "rst_n", bounded_cycles=0)


def test_contract_bounds_default_and_validate(tmp_path):
    registry = tmp_path / "contracts.json"
    import json

    registry.write_text(
        json.dumps(
            {
                "c1": {
                    "requirement": "Hold when locked.",
                    "assertion": "assert(q);",
                    "scope": "sequential",
                    "reset": "rst_n",
                },
                "c2": {
                    "requirement": "Hold when locked deeply.",
                    "assertion": "assert(q);",
                    "scope": "sequential",
                    "reset": "rst_n",
                    "bound_cycles": 10,
                    "induction_steps": 20,
                },
            }
        )
    )
    loaded = load_contracts(registry)
    assert loaded["c1"]["bound_cycles"] == 6
    assert loaded["c2"]["bound_cycles"] == 10
