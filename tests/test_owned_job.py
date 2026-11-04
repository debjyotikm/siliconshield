"""Owned-job manifest validation and receipt tests: synthetic fixtures."""

import json

from siliconshield.research_loop.execution import owned_job


def _manifest(tmp_path, **overrides):
    manifest = {
        "command": ["true"],
        "output": str(tmp_path / "out"),
        "idempotency_key": "k",
        "unit": "u",
        "timeout_seconds": 30,
        "output_quota_bytes": 10**6,
    }
    manifest.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return str(path)


def test_run_rejects_malformed_manifests(tmp_path):
    import pytest

    path = tmp_path / "manifest.json"
    path.write_text("[1, 2]")
    with pytest.raises(ValueError, match="must be an object"):
        owned_job.run(str(path), str(tmp_path / "res"), "train")
    base = {
        "command": ["true"],
        "output": str(tmp_path / "out"),
        "idempotency_key": "k",
        "unit": "u",
        "timeout_seconds": 30,
        "output_quota_bytes": 10**6,
    }
    for key in ("output", "idempotency_key", "unit"):
        bad = {k: v for k, v in base.items() if k != key}
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="needs " + key):
            owned_job.run(str(path), str(tmp_path / "res"), "train")
    for override in (
        {"command": "true"},
        {"command": []},
        {"input_hashes": [1]},
        {"timeout_seconds": "soon"},
        {"output_quota_bytes": -1},
    ):
        bad = dict(base, **override)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            owned_job.run(str(path), str(tmp_path / "res"), "train")


def test_run_completes_trivial_command(tmp_path):
    manifest = _manifest(tmp_path)
    assert owned_job.run(manifest, str(tmp_path / "res"), "train") == 0
    receipt = json.loads((tmp_path / "out" / "execution.json").read_text())
    assert receipt["phase"] == "completed"
    assert receipt["exit_code"] == 0


def test_root_cause_prefers_traceback_summary():
    text = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in <module>\n'
        "ValueError: bad flag\n"
    )
    assert owned_job.root_cause_from_text(text) == "ValueError: bad flag"
    assert owned_job.root_cause_from_text("prog.py: error: pick one\n") == (
        "prog.py: error: pick one"
    )
    assert owned_job.root_cause_from_text("all quiet\n") is None


def test_sandbox_rejects_misconfigured_limits():
    import pytest

    from siliconshield.research_loop.execution import sandbox

    with pytest.raises(ValueError, match="name list"):
        sandbox.restricted_env("PATH")
    for timeout, address in (
        (None, 16),
        ("5", 16),
        (-1, 16),
        (5, "x"),
        (5, 0),
        (float("inf"), 16),
    ):
        with pytest.raises(ValueError, match="Sandbox limit needs"):
            sandbox.limited_preexec(timeout, address_gb=address)
    assert callable(sandbox.limited_preexec(5))
