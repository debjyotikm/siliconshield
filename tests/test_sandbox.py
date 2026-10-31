"""Restricted subprocess helpers: allowlisted env, bounded resources."""

import os

from siliconshield.research_loop.execution import sandbox


def test_restricted_env_drops_unlisted_secrets(monkeypatch):
    monkeypatch.setenv("SILICONSHIELD_SOME_SECRET", "shh")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = sandbox.restricted_env()
    assert "SILICONSHIELD_SOME_SECRET" not in env
    assert env["PATH"] == "/usr/bin:/bin"
    assert "PATH" in sandbox.BASE_ENV_KEEP


def test_restricted_env_honors_explicit_extras(monkeypatch):
    monkeypatch.setenv("LAYA_ROOT", "/pinned/laya")
    assert "LAYA_ROOT" not in sandbox.restricted_env()
    assert (
        sandbox.restricted_env(extra=["LAYA_ROOT"])["LAYA_ROOT"]
        == "/pinned/laya"
    )


def test_limited_preexec_bounds_a_child_process():
    preexec = sandbox.limited_preexec(5)
    if os.name != "posix":
        assert preexec is None
        return
    import subprocess

    run = subprocess.run(
        ["python3", "-c", "print('ok')"],
        capture_output=True,
        text=True,
        timeout=30,
        env=sandbox.restricted_env(),
        preexec_fn=preexec,
    )
    assert run.returncode == 0
    assert run.stdout.strip() == "ok"
