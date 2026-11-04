"""Compare the base Glimmer model and a LoRA adapter on the same review tasks."""

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

from siliconshield.research_loop.adapters.glimmer_training import (
    dev_subset,
    load_rows,
)
from siliconshield.research_loop.controller.replay import (
    execute,
    safe_episode_dir,
    safe_grade,
)
from siliconshield.research_loop.controller.serve import local_completion
from siliconshield.research_loop.episodes.contracts import atomic_json
from siliconshield.training.glimmer.identity import checkpoint_fingerprint


class InProcessClient:
    def __init__(
        self,
        network,
        tokenizer,
        name,
        checkpoint_fingerprint=None,
        adapter_path=None,
        record_timing=False,
    ):
        self.network, self.tokenizer, self.model = network, tokenizer, name
        self.endpoint = "in-process"
        self.max_tokens = 2048
        self.record_timing = record_timing
        self.last_completion_metrics = None
        self.checkpoint_fingerprint, self.adapter_path = (
            checkpoint_fingerprint,
            adapter_path,
        )

    def complete(self, messages, tools=()):
        result = local_completion(
            self.network,
            self.tokenizer,
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "max_tokens": self.max_tokens,
                "record_timing": self.record_timing,
            },
            name=self.model,
        )
        choice = result["choices"][0]
        self.last_completion_metrics = {
            "usage": result.get("usage"),
            "timing": result.get("timing"),
        }
        if choice["finish_reason"] == "length":
            raise RuntimeError(
                "Generation reached the token limit before the response finished"
            )
        return choice["message"]


def require_episodes(rows: list) -> list:
    """Refuse a vacuous evaluation that would report a hollow pass.

    Zero selected episodes verify nothing; adapter-vs-reference
    counts of 0 >= 0 must never read as non-regression evidence.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Evaluation selected no episodes; refusing a vacuous report."
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Path to the local base-model checkpoint"
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="LoRA checkpoint to compare with the base model",
    )
    parser.add_argument(
        "--reference-adapter",
        help="Compare against an earlier adapter instead of the bare base",
    )
    parser.add_argument(
        "--episodes",
        required=True,
        help="JSONL file containing review episodes",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for checkpoints, reports, and logs",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Maximum development episodes to evaluate",
    )
    parser.add_argument(
        "--harness2-output",
        help="Also run Harness2 trials and save them in this directory",
    )
    parser.add_argument(
        "--record-timing",
        action="store_true",
        help="Record generated-token timings; synchronization adds overhead",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Consume sealed test once; disables procedure refinement",
    )
    args = parser.parse_args()
    if args.final and args.harness2_output:
        raise ValueError(
            "Final evaluation cannot propose or retain Harness2 lessons"
        )
    if args.harness2_output:
        # Harness2 is optional; check it before loading any model weights.
        from siliconshield.research_loop.procedures.harness2_adapter import (
            GlimmerProposer,
            LayaResearchAdapter,
            run_verified,
        )
    from siliconshield.research_loop.execution.gpu_lease import acquire_gpu

    gpu_lease = acquire_gpu()
    if gpu_lease.closed:
        raise RuntimeError(
            "Glimmer GPU lease is not held; release the other holder "
            "(training, serving, or evaluation) and retry"
        )
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    output = Path(args.output)
    if args.final and not os.environ.get("SILICONSHIELD_DATASET_KEY"):
        raise ValueError(
            "Final evaluation requires an authenticated dataset manifest"
        )
    rows = load_rows(
        args.episodes, allowed_splits=("test",) if args.final else ("dev",)
    )
    if not args.final:
        rows = dev_subset(rows, args.examples)
    require_episodes(rows)
    from siliconshield.training.glimmer.identity import (
        MODEL_REVISION,
        code_identity,
        verify_snapshot,
    )

    atomic_json(
        output / "status.json", {"phase": "verifying_snapshot", "n": len(rows)}
    )
    snapshot = verify_snapshot(
        args.model,
        os.environ.get("SILICONSHIELD_MODEL_FINGERPRINT"),
        revision=MODEL_REVISION,
    )
    if args.final:
        seal = Path(args.episodes).parent / ".final-evaluation"
        with seal.open("x") as stream:
            json.dump(
                {
                    "base": snapshot["fingerprint"],
                    "adapter": str(Path(args.adapter).resolve()),
                    "code": code_identity(),
                    "output": str(output.resolve()),
                },
                stream,
            )
    atomic_json(output / "status.json", {"phase": "loading", "n": len(rows)})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = MuseGlimmerForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    atomic_json(
        output / "status.json", {"phase": "loading_adapters", "n": len(rows)}
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    reference_arm = "reference" if args.reference_adapter else "base"
    if args.reference_adapter:
        model.load_adapter(
            args.reference_adapter, adapter_name="reference", is_trainable=False
        )
    model.eval()
    fingerprints = {
        "base": snapshot["fingerprint"],
        "adapter": checkpoint_fingerprint(
            args.adapter, base_revision=snapshot["fingerprint"]
        ),
    }
    if args.reference_adapter:
        fingerprints["reference"] = checkpoint_fingerprint(
            args.reference_adapter, base_revision=snapshot["fingerprint"]
        )
    results = {}
    for arm in (reference_arm, "adapter"):
        model.set_adapter("reference" if arm == "reference" else "default")
        scores = []
        client = InProcessClient(
            model,
            tokenizer,
            "muse-glimmer-" + arm,
            checkpoint_fingerprint=fingerprints[arm],
            adapter_path=args.adapter
            if arm == "adapter"
            else args.reference_adapter,
            record_timing=args.record_timing,
        )
        atomic_json(
            output / "status.json",
            {
                "phase": "evaluating",
                "arm": arm,
                "completed": 0,
                "n": len(rows),
            },
        )
        with model.disable_adapter() if arm == "base" else nullcontext():
            for row in rows:
                path = safe_episode_dir(output / arm, row["id"])
                delivered = execute(row, client, path)
                scored = safe_grade(row, delivered)
                result = {
                    "episode_id": row["id"],
                    "group_id": row["group_id"],
                    **scored,
                }
                atomic_json(path / "grade.json", result)
                scores.append(result)
                atomic_json(
                    output / "status.json",
                    {
                        "phase": "evaluating",
                        "arm": arm,
                        "completed": len(scores),
                        "n": len(rows),
                        "last": result,
                    },
                )
        results[arm] = {
            "n": len(scores),
            "execution_complete": sum(s["execution_complete"] for s in scores),
            "analysis_correct": sum(s["analysis_correct"] for s in scores),
            **{
                k: sum(bool(s.get(k)) for s in scores)
                for k in (
                    "action_correct",
                    "evidence_correct",
                    "explanation_correct",
                    "final_consistent",
                )
            },
            "by_group": {
                group: {
                    "n": sum(s["group_id"] == group for s in scores),
                    "verified": sum(
                        bool(s["verified"])
                        for s in scores
                        if s["group_id"] == group
                    ),
                }
                for group in sorted({s["group_id"] for s in scores})
            },
            "verified": sum(s["verified"] for s in scores),
            "results": scores,
        }
    report = {
        "phase": "complete",
        "episodes": [r["id"] for r in rows],
        "arms": results,
        "base_revision": MODEL_REVISION,
        "checkpoint_fingerprints": fingerprints,
        "reference_arm": reference_arm,
        "adapter_nonregressing_on_replay": results["adapter"]["verified"]
        >= results[reference_arm]["verified"],
        "code_identity": code_identity(),
        "final_evaluation": args.final,
        "episodes_sha256": hashlib.sha256(
            Path(args.episodes).read_bytes()
        ).hexdigest(),
        "decoding": {
            "do_sample": False,
            "max_tokens": 2048,
            "precision": "bfloat16",
            "record_timing": args.record_timing,
        },
        "image_digest": os.environ.get("SILICONSHIELD_IMAGE_DIGEST"),
        "limitations": "Small development replay; does not establish autonomous research quality.",
        "automatic_laya_training_started": False,
    }
    atomic_json(output / "report.json", report)
    if args.harness2_output:
        root = Path(args.harness2_output)
        client = InProcessClient(
            model,
            tokenizer,
            "muse-glimmer-adapter",
            checkpoint_fingerprint=fingerprints["adapter"],
            adapter_path=args.adapter,
        )
        client.max_tokens = 4096
        adapter = LayaResearchAdapter(root, rows[:2], client)
        proposer = GlimmerProposer(client, root / "proposer")
        atomic_json(output / "status.json", {"phase": "harness2_replay"})
        harness_result = run_verified(adapter, proposer, root)
        atomic_json(root / "report.json", harness_result)
    atomic_json(
        output / "status.json",
        {"phase": "complete", "report": str(output / "report.json")},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
