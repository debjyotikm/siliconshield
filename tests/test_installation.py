"""Build a wheel from an isolated copy and use the package from outside the checkout."""

import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("setuptools")
pytest.importorskip("wheel")

ROOT = Path(__file__).resolve().parents[1]


def read_project_version():
    """The single package-version source: pyproject.toml."""
    text = (ROOT / "pyproject.toml").read_text()
    marker = 'version = "'
    return text.split(marker)[1].split('"')[0]


def run(*args, **kwargs):
    return subprocess.run(
        *args, check=True, capture_output=True, text=True, **kwargs
    )


def test_wheel_installs_and_runs_outside_checkout(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    dist = tmp_path / "dist"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(dist),
            str(source),
        ]
    )
    wheels = list(dist.glob("siliconshield-*.whl"))
    assert len(wheels) == 1
    target = tmp_path / "installed"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target),
            str(wheels[0]),
        ]
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    env = {**os.environ, "PYTHONPATH": str(target), "PYTHONNOUSERSITE": "1"}
    probe = run(
        [
            sys.executable,
            "-c",
            "import siliconshield, siliconshield.research_loop.execution.security_jobs as j; "
            "print(siliconshield.__file__); print(j.command_for.__name__); "
            "print(siliconshield.__version__)",
        ],
        cwd=outside,
        env=env,
    )
    assert str(target) in probe.stdout
    # The installed distribution version is the single package-version source;
    # the package initializer reports it from installed metadata.
    expected = read_project_version()
    assert wheels[0].name.startswith("siliconshield-" + expected)
    assert probe.stdout.strip().splitlines()[-1] == expected
    cli = run(
        [
            sys.executable,
            "-m",
            "siliconshield.research_loop.execution.security_jobs",
            "--help",
        ],
        cwd=outside,
        env=env,
    )
    assert "--experiment-id" in cli.stdout
    for dotted in (
        "siliconshield.training.laya.augment_security_policies",
        "siliconshield.training.laya.augment_security_guards",
        "siliconshield.training.laya.expand_security_data",
        "siliconshield.training.laya.build_security_external",
        "siliconshield.research_loop.controller.serve_fast",
        "siliconshield.research_loop.procedures.muse_harness2",
    ):
        entry = run(
            [sys.executable, "-m", dotted, "--help"], cwd=outside, env=env
        )
        assert "usage:" in entry.stdout.lower()


def test_package_version_comes_from_installed_metadata_or_source_fallback():
    import siliconshield

    try:
        dist_version = importlib.metadata.version("siliconshield")
    except importlib.metadata.PackageNotFoundError:
        dist_version = None
    if dist_version is None:
        assert siliconshield.__version__ == "0.0.0+source"
    else:
        assert siliconshield.__version__ == dist_version
    expected = read_project_version()
    assert expected and all(part.isdigit() for part in expected.split("."))
