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
    pytest.importorskip("pip", reason="wheel installation requires pip")
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
            ".venv*",
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
        "siliconshield.web_demo",
        "siliconshield.web_demo.toy_runner",
    ):
        entry = run(
            [sys.executable, "-m", dotted, "--help"], cwd=outside, env=env
        )
        assert "usage:" in entry.stdout.lower()
    assets = run(
        [
            sys.executable,
            "-c",
            "from importlib.resources import files; "
            "p = files('siliconshield.web_demo'); "
            "assert 'Decision Model Lab' in p.joinpath('static/index.html').read_text(); "
            "assert 'fifo-simultaneous-guarded' in p.joinpath('examples.example.json').read_text(); "
            "import siliconshield.web_demo.server",
        ],
        cwd=outside,
        env=env,
    )
    assert assets.returncode == 0


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


def test_pinned_download_checks_bytes_before_publish(tmp_path, monkeypatch):
    """A failed download must not become a usable model file."""
    import hashlib
    import io
    import json

    import pytest

    from tools.download_models import prepare_model

    body = b"synthetic model payload"
    files = {"model.safetensors": hashlib.sha256(body).hexdigest()}
    record = {
        "repo_id": "example/synthetic",
        "revision": "a" * 40,
        "files": files,
        "export_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    target = tmp_path / "model"
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"damaged")
    )
    with pytest.raises(ValueError, match="Downloaded file checksum"):
        prepare_model(record, target)
    assert not list(target.iterdir())
    urls = []

    def download(url, **kwargs):
        urls.append(url)
        return io.BytesIO(body)

    monkeypatch.setattr("urllib.request.urlopen", download)
    assert prepare_model(record, target) == record["export_sha256"]
    assert record["revision"] in urls[0]
    assert prepare_model(record, target) == record["export_sha256"]
    assert len(urls) == 1  # Reuse requires matching bytes, not a new request.
    (target / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Existing file checksum"):
        prepare_model(record, target)
    assert len(urls) == 1


def test_release_fingerprints_match_worker_identity_contract():
    """Public payload pins use the same aggregate identity as live workers."""
    import tomllib

    from siliconshield.experts.identity import digest
    from tools.download_models import MANIFEST

    manifest = tomllib.loads(MANIFEST.read_text())
    for record in manifest["models"].values():
        assert record["export_sha256"] == digest(record["files"])
