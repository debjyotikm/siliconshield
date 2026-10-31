"""Exercise the actual pinned Harness2 recursion with local fixture model responses."""

import json

import pytest

pytest.importorskip("harness2")
from harness2.core.proposer import Reply
from harness2.core.schedule import run_domain

from siliconshield.research_loop.procedures.harness2_adapter import (
    GlimmerProposer,
    LayaResearchAdapter,
)
from tests.test_glimmer_contracts import ScriptedClient, episode


class FixtureProposer(GlimmerProposer):
    def call(self, mission, context_dir, label, **kwargs):
        from pathlib import Path

        workspace = (
            Path(context_dir) / "harness" if kwargs.get("harvest") else None
        )
        if workspace:
            prompt = workspace / "systemprompt.md"
            prompt.write_text(
                prompt.read_text()
                + "\nCalculate metric changes with the comparison tool."
            )
            text = "<notes>Use the available calculation tool.</notes>"
        else:
            text = "NO_ENTRY"
        reply = Reply(
            label=label, text=text, ok=True, attempts=1, workspace=workspace
        )
        self._calls.append(reply)
        return reply


def test_real_recursion_needs_no_cloud_or_gpu(tmp_path):
    client = ScriptedClient()
    row = episode()
    adapter = LayaResearchAdapter(tmp_path, [row], client)
    proposer = FixtureProposer(client, tmp_path / "audit")
    report = run_domain(
        "checkpoint_review",
        adapter.tasks(),
        1,
        adapter=adapter,
        prop=proposer,
        runs_root=adapter.spec.runs_root,
        library_root=tmp_path / "library",
        modes=("seq",),
    )
    assert report["done"] == 1 and report["failed"] == 0
    outputs = list((tmp_path / "executions").rglob("deliverable.json"))
    assert len(outputs) == 4
    assert all(
        adapter.score(p.parent, row["id"])["analysis_correct"] for p in outputs
    )
    # No oracle actions leak into task source files.
    source = json.loads(
        (adapter.task_files(row["id"]) / "development.json").read_text()
    )
    assert "target" not in source and "action" not in source


def test_model_change_cannot_reuse_old_namespace(tmp_path):
    client = ScriptedClient()
    LayaResearchAdapter(tmp_path, [episode()], client)
    client.model = "fixture-v2"
    with pytest.raises(ValueError, match="new output directory"):
        LayaResearchAdapter(tmp_path, [episode()], client)


def test_missing_or_incomplete_adapter_path_is_rejected(tmp_path):
    from siliconshield.research_loop.procedures.harness2_adapter import (
        checkpoint_fingerprint,
    )

    with pytest.raises(ValueError, match="does not exist"):
        checkpoint_fingerprint(tmp_path / "typo-adapter")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint_fingerprint(incomplete)


def test_checkpoint_change_with_same_name_cannot_reuse_old_namespace(tmp_path):
    from siliconshield.research_loop.procedures.harness2_adapter import (
        checkpoint_fingerprint,
    )

    client = ScriptedClient()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"r": 16}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights-v1")
    client.checkpoint_fingerprint = checkpoint_fingerprint(adapter_dir)
    LayaResearchAdapter(tmp_path / "replay", [episode()], client)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights-v2")
    client.checkpoint_fingerprint = checkpoint_fingerprint(adapter_dir)
    assert client.model == "fixture-v1"  # name unchanged; weights changed
    with pytest.raises(ValueError, match="new output directory"):
        LayaResearchAdapter(tmp_path / "replay", [episode()], client)
