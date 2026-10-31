"""Use Harness2 and a local Glimmer endpoint to test checkpoint-review prompts."""

import argparse
import json
import shutil
from pathlib import Path

from harness2.core import tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.schedule import run_task
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.substrate.opencode import OpencodeSubstrate

from siliconshield.research_loop.adapters.glimmer_training import load_rows
from siliconshield.research_loop.controller.client import ChatClient
from siliconshield.research_loop.controller.replay import execute, grade
from siliconshield.research_loop.episodes.contracts import (
    SYSTEM,
    atomic_json,
    digest,
    read_json,
)
from siliconshield.training.glimmer.identity import (
    MODEL_REVISION,
    checkpoint_fingerprint,
)


def client_fingerprint(client):
    fingerprint = getattr(client, "checkpoint_fingerprint", None)
    if fingerprint:
        return fingerprint
    adapter_path = getattr(client, "adapter_path", None)
    if adapter_path:
        return checkpoint_fingerprint(adapter_path)
    return checkpoint_fingerprint(None)


def execution_identity(client):
    """Use provider identity for CLI clients and weight identity for Glimmer."""
    explicit = getattr(client, "execution_identity", None)
    if explicit is not None:
        return dict(explicit)
    return {
        "model": client.model,
        "endpoint": client.endpoint,
        "base_revision": MODEL_REVISION,
        "checkpoint_fingerprint": client_fingerprint(client),
    }


class LayaResearchAdapter:
    """Apply four prompt components to reviews using the existing Python tools."""

    def __init__(self, root, rows, client):
        self.root = Path(root)
        self.rows = {r["id"]: r for r in rows}
        self.client = client
        self.substrate = OpencodeSubstrate()
        channel = ChannelSpec("opencode", "local/" + client.model)
        self.spec = BenchSpec(
            name="laya_research",
            runs_root=self.root / "runs",
            install_mode="workspace_config",
            prompts=BenchPrompts(
                work="inspect learning-run evidence and record a sound checkpoint decision",
                domain_word="research task",
                inv_levers="clear evidence reading, complete tool use, precise arithmetic",
                dom_levers="distinguish smoke execution from development quality",
                selection_checks="require a supported decision and an explicit evidence trail",
            ),
            channels={"proposer": channel, "selector": channel},
        )
        self.root.mkdir(parents=True, exist_ok=True)
        identity = {
            **execution_identity(client),
            "episodes": digest(rows),
            "integration_schema": 2,
        }
        identity_path = self.root / "identity.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError(
                "Use a new output directory when the model or review episodes change"
            )
        atomic_json(identity_path, identity)
        initial = {name: "" for name in self.substrate.surface().always_on}
        initial["systemprompt.md"] = SYSTEM
        if not self.base_harness().exists():
            tree.write_tree(
                self.substrate.surface(),
                initial,
                self.base_harness(),
                stubs=False,
            )

    def tasks(self, split="all"):
        return list(self.rows)

    def domain_of(self, task_id):
        return "checkpoint_review"

    def task_prompt(self, task_id):
        return self.rows[task_id]["messages"][1]["content"]

    def task_files(self, task_id):
        dest = self.root / "task_inputs" / task_id
        dest.mkdir(parents=True, exist_ok=True)
        atomic_json(dest / "development.json", self.rows[task_id]["summary"])
        return dest

    def base_harness(self):
        return self.root / "base_harness"

    def validate_tree(self, proposed):
        extra = set(proposed) - set(self.substrate.surface().always_on)
        if extra:
            raise ValueError(
                "Only prompt text files are supported; remove: "
                + ", ".join(sorted(extra))
            )
        if not proposed.get("systemprompt.md", "").strip():
            raise ValueError("A system prompt is required")

    def execute(self, task_id, harness_dir, run_id, *, timeout=None):
        root = self.root / "executions" / run_id
        chosen = tree.read_tree(
            self.substrate.surface(), harness_dir or self.base_harness()
        )
        self.validate_tree(chosen)
        identity = {
            "task": task_id,
            **execution_identity(self.client),
            "harness": digest(chosen),
        }
        receipt = root / "identity.json"
        if receipt.exists():
            if read_json(receipt) != identity:
                raise ValueError(
                    "Execution ID belongs to different model/task/harness"
                )
            if (root / "deliverable.json").exists():
                result = read_json(root / "deliverable.json")
                if not result.get("error") and result.get("decision"):
                    return root
        system = "\n\n".join(
            chosen[name]
            for name in self.substrate.surface().always_on
            if chosen.get(name)
        )
        execute(self.rows[task_id], self.client, root, system=system)
        atomic_json(receipt, identity)
        return root

    def delivered(self, run_dir):
        path = Path(run_dir) / "deliverable.json"
        if not path.exists():
            return False
        result = read_json(path)
        return not result.get("error") and bool(result.get("decision"))

    def trajectory(self, run_dir, *, obs_trunc):
        # Return the full trace so review includes complete tool arguments.
        path = Path(run_dir) / "trajectory.jsonl"
        return path.read_text() if path.exists() else "(no trajectory)"

    def deliverable_files(self, run_dir):
        return {"deliverable.json": self.deliverable_text(run_dir)}

    def deliverable_text(self, run_dir):
        path = Path(run_dir) / "deliverable.json"
        return path.read_text() if path.exists() else ""

    def score(self, run_dir, task_id):
        return grade(
            self.rows[task_id], read_json(Path(run_dir) / "deliverable.json")
        )


class GlimmerProposer(Proposer):
    """Ask the local Glimmer endpoint to propose prompt edits for Harness2."""

    def __init__(self, client, audit_dir):
        super().__init__(
            ChannelSpec("opencode", "local/" + client.model),
            audit_dir=Path(audit_dir),
        )
        self.client = client

    def call(self, mission, context_dir, label, **kwargs):
        try:
            return self._call(mission, context_dir, label, **kwargs)
        except (ValueError, KeyError, OSError) as exc:
            raise ProposerError(
                f"Local Muse harness edit failed: {exc}", attempts=1
            ) from exc

    def _call(self, mission, context_dir, label, **kwargs):
        context = Path(context_dir)
        files = {}
        for path in sorted(context.rglob("*")):
            if path.is_file() and path.suffix in (
                ".md",
                ".json",
                ".jsonl",
                ".txt",
            ):
                files[str(path.relative_to(context))] = path.read_text()
        instruction = "Follow the mission using the supplied evidence. Do not claim an outcome without evidence."
        if kwargs.get("harvest"):
            instruction += (
                ' Return a JSON object with "files" mapping editable harness filenames to complete '
                'replacement text and "notes" explaining edits. Editable paths are systemprompt.md, '
                "guardrails.md, LongTermMEMORY.md, tool_descriptions/tool_guidance.md. "
                "Preserve existing useful instructions. No scripts or other paths are supported."
            )
        message = self.client.complete(
            [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": mission + "\n\n" + json.dumps(files),
                },
            ]
        )
        text = message.get("content", "")
        workspace = None
        if kwargs.get("harvest"):
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
            value = json.loads(stripped)
            workspace = Path(self.audit_dir) / (label + ".harness")
            if workspace.exists():
                shutil.rmtree(workspace)
            shutil.copytree(context / "harness", workspace)
            allowed = set(OpencodeSubstrate().surface().always_on)
            for name, body in value["files"].items():
                if name not in allowed or not isinstance(body, str):
                    raise ValueError("Unsupported harness edit")
                dest = workspace / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(body)
            text = "<notes>" + str(value.get("notes", "")) + "</notes>"
        Path(self.audit_dir).mkdir(parents=True, exist_ok=True)
        atomic_json(
            Path(self.audit_dir) / (label + ".json"),
            {
                "model": self.client.model,
                "mission": mission,
                "response": message,
            },
        )
        reply = Reply(
            label=label, text=text, ok=True, attempts=1, workspace=workspace
        )
        self._calls.append(reply)
        return reply


def run_verified(adapter, proposer, root):
    """Run prompt trials and retain lessons whose review decisions pass grading."""
    root = Path(root)
    library = Library(
        root=root / "library",
        domain="checkpoint_review",
        surface=adapter.substrate.surface(),
    )
    records = []
    for task_id in adapter.tasks():
        result = run_task(
            task_id,
            1,
            adapter=adapter,
            prop=proposer,
            runs_root=adapter.spec.runs_root,
            library=library,
            modes=("seq",),
        )
        selection = result.selections.get("seq", {})
        chosen = selection.get("chosen_run_dir")
        verified = False
        if chosen and adapter.delivered(chosen):
            score = adapter.score(chosen, task_id)
            verified = score.get("verified", False)
        if not verified:
            library.forget([task_id])
        records.append(
            {
                "task": task_id,
                "status": result.status,
                "verified_analysis": verified,
                "retained": verified and library.entry_dir(task_id).exists(),
                "errors": result.errors,
            }
        )
        atomic_json(root / "verified_tasks.json", records)
    return {
        "tasks": records,
        "done": sum(r["status"] in ("ok", "skipped") for r in records),
        "verified": sum(r["verified_analysis"] for r in records),
        "acceptance": "external development replay checks; no final-test feedback",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        required=True,
        help="JSONL file containing review episodes",
    )
    parser.add_argument(
        "--endpoint", required=True, help="Chat-completions endpoint URL"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name expected by the serving endpoint",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Fresh directory for prompt trials and review results",
    )
    parser.add_argument(
        "--limit", type=int, default=3, help="Maximum review episodes to run"
    )
    parser.add_argument(
        "--adapter",
        help="Adapter checkpoint served at the endpoint; fingerprinted into identity",
    )
    args = parser.parse_args()
    rows = load_rows(args.episodes)[: args.limit]
    client = ChatClient(
        args.endpoint,
        args.model,
        max_tokens=4096,
        adapter_path=args.adapter,
        checkpoint_fingerprint=checkpoint_fingerprint(args.adapter)
        if args.adapter
        else None,
    )
    root = Path(args.output)
    adapter = LayaResearchAdapter(root, rows, client)
    proposer = GlimmerProposer(client, root / "proposer")
    result = run_verified(adapter, proposer, root)
    # The selector has already chosen; external grades are reported separately.
    grades = {}
    for path in (root / "executions").rglob("deliverable.json"):
        report = read_json(path)
        row = adapter.rows[report["episode_id"]]
        grades[str(path.parent.relative_to(root / "executions"))] = grade(
            row, report
        )
    atomic_json(root / "external_grades.json", grades)
    atomic_json(root / "report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
