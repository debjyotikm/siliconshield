"""Qualify Muse-authored procedures using the pinned Harness2 recursion."""

import hashlib
from pathlib import Path
from typing import Any

import harness2
from harness2.core import tree
from harness2.core.library import Library
from harness2.core.schedule import Namespace, run_task

from siliconshield.research_loop.controller.replay import (
    execute,
    safe_grade,
)
from siliconshield.research_loop.episodes.contracts import (
    atomic_json,
    digest,
    read_json,
    validate_episode,
)
from siliconshield.research_loop.procedures.bundles import (
    publish_bundle,
    qualifies,
    read_bundle,
)
from siliconshield.research_loop.procedures.harness2_adapter import (
    GlimmerProposer,
    LayaResearchAdapter,
)

GUIDANCE_FILE = "LongTermMEMORY.md"

# The replay corpus assumed a trained operator already knew these contracts.
# A coding-agent backend must receive the same task definitions explicitly.
# These rules contain no episode-specific labels or measured values.
CHECKPOINT_POLICY = (
    "Fixed checkpoint-review policy, in priority order. Let a be the agreement "
    "change in percentage points and n the NLL change from compare_metrics. "
    "If phase is smoke OR completed_updates <= 2, choose smoke_only. Otherwise: "
    "if a < -1e-5 and n > 1e-7, choose inspect_regression; "
    "if (a < -1e-5 and n < -1e-7) or (a > 1e-5 and n > 1e-7), "
    "choose inspect_tradeoff; if (a > 1e-5 and n <= 1e-7) or "
    "(n < -1e-7 and a >= -1e-5), choose retain_reviewed; otherwise "
    "choose retain_initial. This action classifies a development review. "
    "It does not grant campaign promotion or deployment acceptance. Report "
    "failed security gates separately; do not replace this task's policy "
    "with a different action policy. These are reused development results, "
    "not evidence of unseen-case performance."
)
OPERATOR_ASSESSMENT_CONTRACT = (
    "Assessment format: compare the last two development_history entries only "
    "when metrics_comparable is true and their population_id values match. "
    "The aggregate entries' metrics pair is named development in comparisons; "
    "additional pairs use the exact names common to both entries' views. "
    "Do not use population_id as a view name. Include only pairs with equal n. "
    "For each pair report accuracy_delta_pp and false_safe_delta_pp as "
    "100*(after-before), and nll_delta as after-before; unavailable values "
    "are null. Empty/mismatched histories have no comparisons. Copy observed "
    "execution failure and selected step; acceptance is passed/failed/unknown "
    "from acceptance.accepted; plateau_reductions counts the recorded "
    "schedule.plateau_lr_reductions, or is null if unavailable. "
    "When checkpoint_candidates are provided, include checkpoint_review with "
    "feasible_ids, selected_feasible, failed_constraints, and "
    "unknown_constraints; otherwise checkpoint_review is null. In prose, "
    "distinguish absolute metrics from their deltas and keep observed numeric "
    "precision. Use the structured assessment for numerical deltas. "
    "Development gains do not establish generalization."
)


def implementation_identity() -> dict[str, str]:
    """Hash the installed integration and Harness2 code used for this trial."""
    package = Path(__file__).resolve().parents[2]
    roots = {"siliconshield": package, "harness2": Path(harness2.__path__[0])}
    return {
        name: digest(
            {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(root.rglob("*.py"))
            }
        )
        for name, root in roots.items()
    }


def validate_populations(
    rows: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> None:
    """Require separate development situations and complete replay contracts."""
    if not rows or not checks:
        raise ValueError(
            "Learning and qualification populations must be nonempty"
        )
    for population in (rows, checks):
        identifiers = set()
        for row in population:
            validate_episode(row)
            identifier = row["id"]
            if (
                identifier in identifiers
                or not isinstance(identifier, str)
                or not identifier
                or any(
                    c
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                    for c in identifier
                )
            ):
                raise ValueError(
                    "Episode identifiers must be unique safe names"
                )
            identifiers.add(identifier)
            summary = row.get("summary")
            if not isinstance(summary, dict):
                raise ValueError("Episode %r has no summary" % (identifier,))
            scope = summary.get("split_role")
            operator = any(
                tool["function"]["name"] == "operator.decide"
                for tool in row["tools"]
            )
            # Operator episodes label the dataset partition on the row. Their
            # observed state has no split_role; test_touched can legitimately
            # be true in a dev exercise about refusing an invalid resume.
            allowed_operator = (
                operator
                and row.get("split") in ("train", "dev")
                and scope in (None, "open_development")
            )
            if scope != "open_development" and not allowed_operator:
                raise ValueError(
                    "Only open development observations are allowed"
                )
            if row.get("split") not in (None, "train", "dev"):
                raise ValueError("Final-test episodes cannot refine procedures")
    for field in ("id", "group_id"):
        if {r[field] for r in rows} & {r[field] for r in checks}:
            raise ValueError(
                "Learning and qualification episodes overlap: " + field
            )


class MuseResearchAdapter(LayaResearchAdapter):
    """Restrict proposals to advice; immutable replay tools define authority."""

    def validate_tree(self, proposed: dict[str, str]) -> None:
        """Permit changes only to the procedure advice component."""
        super().validate_tree(proposed)
        original = tree.read_tree(self.substrate.surface(), self.base_harness())
        for name in set(original) | set(proposed):
            if name != GUIDANCE_FILE and proposed.get(name, "") != original.get(
                name, ""
            ):
                raise ValueError("Only LongTermMEMORY.md may change")

    def task_files(self, task_id: str) -> Path:
        """Expose observed evidence, never the authored target or future trace."""
        dest = super().task_files(task_id)
        if "resources" in self.rows[task_id]:
            atomic_json(
                dest / "resources.json", self.rows[task_id]["resources"]
            )
        return Path(dest)

    def execute(
        self,
        task_id: str,
        harness_dir: Path | None,
        run_id: str,
        *,
        timeout: Any = None,
    ) -> Path:
        """Replay with the episode's own contract plus the proposed advice."""
        chosen = tree.read_tree(
            self.substrate.surface(), harness_dir or self.base_harness()
        )
        self.validate_tree(chosen)
        root = (Path(self.root) / "executions" / run_id).resolve()
        if not root.is_relative_to((self.root / "executions").resolve()):
            raise ValueError("Execution directory escapes the trial")
        identity = {
            "episode": digest(self.rows[task_id]),
            "harness": digest(chosen),
            "client": self.client.execution_identity,
        }
        if root.exists():
            if read_json(root / "identity.json") != identity:
                raise ValueError("Execution identity changed")
            if (root / "deliverable.json").is_file():
                return root
            raise ValueError("Incomplete replay; start a fresh trial")
        atomic_json(root / "identity.json", identity)
        row = self.rows[task_id]
        execute(
            row,
            self.client,
            root,
            system=procedure_prompt(row, chosen.get(GUIDANCE_FILE, "")),
        )
        return root


class MuseProposer(GlimmerProposer):
    """Reuse Harness2's proposer protocol with a tool-free Muse CLI client."""

    def call(
        self, mission: str, context_dir: Path, label: str, **kwargs: Any
    ) -> Any:
        """Harvest portable advice without changing the task or tool contract."""
        mission += (
            "\nFor edits, change ONLY LongTermMEMORY.md. Preserve every other "
            "harness file exactly. Write concise reusable evidence-analysis "
            "and recovery advice, not task answers, run identifiers, paths, "
            "threshold overrides, scripts, or new permissions. Advice cannot "
            "override the current campaign's policy."
        )
        return super().call(mission, context_dir, label, **kwargs)


def procedure_prompt(row: dict[str, Any], guidance: str) -> str:
    """Keep the task contract authoritative over learned advice."""
    operator = any(
        tool["function"]["name"] == "operator.decide" for tool in row["tools"]
    )
    contract = OPERATOR_ASSESSMENT_CONTRACT if operator else CHECKPOINT_POLICY
    return (
        str(row["messages"][0]["content"])
        + "\n\n"
        + contract
        + (
            "\n\nProcedure advice (does not change tools or policy):\n"
            + guidance
            if guidance
            else ""
        )
    )


def score_checks(
    rows: list[dict[str, Any]], client: Any, root: Path, guidance: str
) -> dict[str, Any]:
    """Evaluate all checks; only the external grader reads the targets."""
    results: dict[str, Any] = {}
    for row in rows:
        try:
            prompt = procedure_prompt(row, guidance)
        except (KeyError, TypeError, IndexError) as exc:
            results[row["id"]] = {
                "execution_complete": 0,
                "analysis_correct": 0,
                "verified": False,
                "grade_error": "ValueError: row has no usable prompt (%s)"
                % exc,
            }
            continue
        delivered = execute(
            row,
            client,
            root / row["id"],
            system=prompt,
        )
        results[row["id"]] = safe_grade(row, delivered)
    atomic_json(root / "grades.json", results)
    return results


def learn(
    rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    client: Any,
    root: Path,
    bundle_path: Path,
    proposer_factory: Any = MuseProposer,
) -> dict[str, Any]:
    """Run actual Harness2 trials and publish only qualified procedure advice.

    A failed proposal leaves any existing published bundle unchanged. These
    checks measure replay behavior, not the benefit of a training intervention.
    """
    validate_populations(rows, checks)
    prior = (
        read_bundle(bundle_path, client.model) if bundle_path.exists() else None
    )
    prior_guidance = prior["guidance"] if prior else ""
    root.mkdir(parents=True, exist_ok=False)
    identity = {
        **client.execution_identity,
        "implementation": implementation_identity(),
        "learning_population": digest(rows),
        "qualification_population": digest(checks),
        "prior_bundle_digest": prior["digest"] if prior else None,
    }
    client.execution_identity = identity
    adapter = MuseResearchAdapter(root / "trials", rows, client)
    initial = tree.read_tree(
        adapter.substrate.surface(), adapter.base_harness()
    )
    initial[GUIDANCE_FILE] = prior_guidance
    tree.write_tree(
        adapter.substrate.surface(),
        initial,
        adapter.base_harness(),
        stubs=False,
    )
    proposer = proposer_factory(client, root / "proposer")
    library = Library(
        root=root / "library",
        domain="checkpoint_review",
        surface=adapter.substrate.surface(),
    )
    baseline = score_checks(checks, client, root / "baseline", prior_guidance)
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
        label = selection.get("chosen_label")
        record: dict[str, Any] = {
            "task": task_id,
            "status": result.status,
            "qualified": False,
            "errors": result.errors,
        }
        if (
            chosen
            and label
            and adapter.delivered(chosen)
            and adapter.score(chosen, task_id).get("verified")
        ):
            namespace = Namespace(
                adapter.spec.name, adapter.substrate.name, "seq", 1
            )
            directory = (
                namespace.task_dir(adapter.spec.runs_root, task_id)
                / "harnesses"
                / label
            )
            selected_tree = tree.read_tree(
                adapter.substrate.surface(), directory
            )
            adapter.validate_tree(selected_tree)
            guidance = selected_tree.get(GUIDANCE_FILE, "")
            candidate = score_checks(
                checks, client, root / "qualification" / task_id, guidance
            )
            record["qualified"] = bool(guidance.strip()) and qualifies(
                baseline, candidate
            )
            if record["qualified"]:
                bundle = publish_bundle(
                    bundle_path, guidance, identity, baseline, candidate
                )
                record["bundle_digest"] = bundle["digest"]
        if not record["qualified"]:
            library.forget([task_id])
        records.append(record)
        atomic_json(
            root / "report.json",
            {"identity": identity, "tasks": records, "training_started": False},
        )
    return {"identity": identity, "tasks": records, "training_started": False}
