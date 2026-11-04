"""Persistent LAYA inference worker for the router and browser demo.

Runs in the LAYA environment as an owned subprocess of the calling
application. Speaks newline-delimited JSON on stdio: stdin carries
``load``/``analyze``/``shutdown`` commands, stdout carries only
protocol lines (``progress``/``ready``/``result``/``error``), and
diagnostics go to stderr. One model stays resident across many
analyses until shutdown; unloading terminates this process to
release its CUDA context.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from siliconshield.app.laya_assessor import state_text
from siliconshield.experts.contracts import validate_result
from siliconshield.experts.identity import digest, runtime_digest, text_digest
from siliconshield.training.laya.question_contracts import (
    LABELS as CONTRACT_LABELS,
)
from siliconshield.training.laya.question_contracts import (
    question_for,
    read_export_record,
)

LABELS = list(CONTRACT_LABELS)

PROTOCOL_VERSION = 1


def questions_for_staged(staged: Path) -> tuple[dict[str, Any], str]:
    """Return versioned questions for a staged checkpoint.

    Single-file checkpoints read the record from their parent
    export directory. A missing record resolves to the legacy
    omitted-definitions-only contract; a present-but-invalid
    record raises instead of silently downgrading, since wrong
    questions would mislabel every inference.
    """
    staged = Path(staged)
    record_dir = staged.parent if staged.is_file() else staged
    record = read_export_record(record_dir)
    questions = copy.deepcopy(question_for(record["version"]))
    questions["security"]["option_order"] = list(range(len(CONTRACT_LABELS)))
    return questions, record["version"]


def _log(detail: str) -> None:
    print("worker: %s" % detail, file=sys.stderr, flush=True)


def _emit(message: dict[str, Any], session_id: str) -> None:
    payload = dict(message)
    payload.setdefault("protocol", PROTOCOL_VERSION)
    payload.setdefault("session_id", session_id)
    sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
    sys.stdout.flush()


def _progress(
    session_id: str, request_id: str, stage: str, detail: str
) -> None:
    _emit(
        {
            "type": "progress",
            "request_id": request_id,
            "stage": stage,
            "detail": detail,
        },
        session_id,
    )


def _checked_model(command: dict[str, Any]) -> dict[str, Any]:
    model = command.get("model")
    if not isinstance(model, dict):
        raise ValueError("Load command is missing its model entry.")
    for key in ("id", "checkpoint", "device"):
        value = model.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError("Model entry is missing %r." % key)
    device = str(model["device"])
    if device != "cpu" and not device.startswith("cuda:"):
        raise ValueError("Worker supports cpu or cuda:N devices only.")
    return model


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_expected(
    checkpoint: Path, expected: dict[str, Any], emit_progress: Any
) -> None:
    for name, wanted in expected.items():
        if not isinstance(wanted, str):
            raise ValueError("Expected digest for %r is not a string." % name)
        target = checkpoint / name if checkpoint.is_dir() else checkpoint
        # Single-file checkpoints expose the file name directly.
        if checkpoint.is_file():
            if Path(name).name != checkpoint.name:
                raise ValueError(
                    "Expected file %r is not the checkpoint file." % name
                )
            target = checkpoint
        if not target.is_file():
            raise ValueError("Checkpoint is missing expected file %r." % name)
        emit_progress("Checking model files", "hashing %s" % name)
        actual = _hash_file(target)
        if actual.lower() != wanted.lower():
            raise ValueError(
                "Checkpoint file %r hash mismatch; refusing load." % name
            )
    if checkpoint.is_dir():
        actual_names = {
            str(item.relative_to(checkpoint))
            for item in checkpoint.rglob("*")
            if item.is_file()
        }
        unexpected = sorted(actual_names - set(expected))
        if unexpected:
            raise ValueError(
                "Checkpoint has unregistered files %r; refusing load."
                % (unexpected[:3],)
            )


def _tree_hashes(root: Path) -> dict[str, str]:
    """Return SHA-256 digests for every file under a directory."""
    return {
        str(item.relative_to(root)): _hash_file(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def _stage_checkpoint(checkpoint: Path, emit_progress: Any) -> tuple[Path, str]:
    """Copy the sealed export to a private staged generation.

    Returns the staged path and its manifest fingerprint. The source
    export is never modified; tokenizer compatibility fixes run only
    against the private copy. A cached copy is reused only when its
    content hashes match the source, so an altered or interrupted
    copy is always recopied. Copies land under a temporary name and
    are renamed into place atomically.
    """
    emit_progress("Checking model files", "staging private copy")
    fingerprint = hashlib.sha256(
        str(checkpoint.resolve()).encode()
    ).hexdigest()[:16]
    staged_root = Path(tempfile.gettempdir()) / (
        "siliconshield-gui-staging-%s" % fingerprint
    )
    try:
        staged_root.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        pass
    # The staging path is predictable under a shared /tmp: a planted
    # symlink or foreign directory would redirect model loading, so
    # refuse anything that is not our own real directory.
    if staged_root.is_symlink() or not staged_root.is_dir():
        raise ValueError("Staging path is not a directory: %s" % staged_root)
    if staged_root.stat().st_uid != os.geteuid():
        raise ValueError(
            "Staging path is not owned by this user: %s" % staged_root
        )
    if checkpoint.is_file():
        staged = staged_root / checkpoint.name
        if staged.is_file() and _hash_file(staged) == _hash_file(checkpoint):
            return staged, fingerprint
        tmp = staged_root / (checkpoint.name + ".tmp-%d" % os.getpid())
        shutil.copy2(checkpoint, tmp)
        os.replace(tmp, staged)
        return staged, fingerprint
    staged = staged_root / "model"
    if staged.is_dir() and _staged_matches(staged, checkpoint):
        return staged, fingerprint
    if staged.is_dir():
        shutil.rmtree(staged)
    tmp = staged_root / ("model.tmp-%d" % os.getpid())
    if tmp.is_dir():
        shutil.rmtree(tmp)
    shutil.copytree(checkpoint, tmp, symlinks=False)
    os.replace(tmp, staged)
    return staged, fingerprint


def _staged_matches(staged: Path, checkpoint: Path) -> bool:
    """Return True when a cached staged copy matches the source.

    Compares relative file lists and content hashes, so an altered
    or interrupted copy is never trusted. Hash verification of the
    source export itself still happens before staging.
    """
    try:
        source_files = _tree_hashes(checkpoint)
        staged_files = _tree_hashes(staged)
    except OSError:
        return False
    return bool(source_files) and source_files == staged_files


def _check_laya_runtime(laya_root: str | None) -> str:
    import laya  # Local import: Torch/CUDA live only in the worker.

    runtime_file = str(getattr(laya, "__file__", ""))
    if laya_root:
        expected = str(Path(laya_root).resolve())
        actual = str(Path(runtime_file).resolve())
        if not Path(actual).is_relative_to(Path(expected)):
            raise ValueError(
                "LAYA runtime %r is not the configured %r." % (actual, expected)
            )
    return runtime_file


def _load_agent(
    staged: Path, device: str, laya_root: str | None, emit_progress: Any
) -> Any:
    _check_laya_runtime(laya_root)
    emit_progress("Loading model", "importing pinned LAYA runtime")
    # Local imports keep the GUI package importable without Torch.
    import torch  # noqa: E402
    from laya.agent import load  # noqa: E402

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; refusing CPU fallback.")
    emit_progress("Loading model", "loading weights on %s" % device)
    agent = load(str(staged), device=device)
    return agent


def _verify_placement(agent: Any, device: str) -> None:
    actual = str(getattr(agent, "device", ""))
    if actual != device:
        # Some runtimes expose torch.device objects; compare strings.
        if actual.replace("cuda", "cuda") != device:
            raise ValueError(
                "Model is on %r, not the requested %r." % (actual, device)
            )
    if device.startswith("cuda"):
        fallback = int(getattr(agent, "cpu_fallback_count", 0) or 0)
        if fallback > 0:
            raise ValueError("Worker fell back to CPU; refusing load.")


def _answer_block(answer: Any, *keys: str) -> dict[str, Any]:
    """Descend one model answer, refusing malformed blocks."""
    block: Any = answer
    for key in keys:
        if not isinstance(block, dict):
            raise ValueError("Model returned a malformed answer.")
        block = block.get(key, {})
    if not isinstance(block, dict):
        raise ValueError("Model returned a malformed answer.")
    return block


def _warm_up(agent: Any, emit_progress: Any, questions: dict[str, Any]) -> None:
    """Run one authored warm-up inference to prove device placement."""
    emit_progress("Warming up", "running authored warm-up inference")
    state = state_text(
        "Warm-up: the module must hold reset values.",
        "",
        "module warmup; endmodule",
    )
    answers = agent.predict_batch([state], questions, sort_by_length=False)
    if not isinstance(answers, list) or not answers:
        raise ValueError("Warm-up produced no answer.")
    answer = answers[0]
    values = _answer_block(answer, "answers", "security").get(
        "probabilities", {}
    )
    if not isinstance(values, dict):
        raise ValueError("Model returned a malformed answer.")
    for label in LABELS:
        number = values.get(label)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError("Warm-up returned a non-numeric score.")
    if str(getattr(agent, "device", "")).startswith("cuda"):
        fallback = int(getattr(agent, "cpu_fallback_count", 0) or 0)
        if fallback > 0:
            raise ValueError("Warm-up fell back to CPU; refusing load.")


def _run_analyze(
    agent: Any, state: str, questions: dict[str, Any]
) -> dict[str, Any]:
    """Run one inference and validate labels, scores, and coverage."""
    before = int(getattr(agent, "cpu_fallback_count", 0) or 0)
    started = time.monotonic()
    answers = agent.predict_batch([state], questions, sort_by_length=False)
    elapsed = time.monotonic() - started
    after = int(getattr(agent, "cpu_fallback_count", 0) or 0)
    on_cuda = str(getattr(agent, "device", "")).startswith("cuda")
    if on_cuda and after > before:
        raise ValueError("Inference fell back to CPU; no verdict produced.")
    if not isinstance(answers, list) or not answers:
        raise ValueError("Model produced no answer.")
    answer = answers[0]
    if not isinstance(answer, dict):
        raise ValueError("Model returned a malformed answer.")
    usage = answer.get("usage")
    if not isinstance(usage, dict) or any(
        key not in usage
        for key in ("truncated", "truncated_questions", "state_tokens_dropped")
    ):
        raise ValueError("Runtime did not provide complete coverage evidence")
    if usage.get("truncated"):
        raise ValueError("Model input was truncated; no verdict produced.")
    if usage.get("truncated_questions"):
        raise ValueError("Model questions were truncated.")
    dropped = usage["state_tokens_dropped"]
    if isinstance(dropped, (int, float)) and dropped > 0:
        raise ValueError("Model input was truncated; no verdict produced.")
    values = _answer_block(answer, "answers", "security").get(
        "probabilities", {}
    )
    if not isinstance(values, dict):
        raise ValueError("Model returned a malformed answer.")
    probabilities: dict[str, float] = {}
    for label in LABELS:
        number = values.get(label)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError("Model returned a non-finite score.")
        if not 0.0 <= float(number) <= 1.0:
            raise ValueError("Model score is outside [0, 1].")
        probabilities[label] = float(number)
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-3:
        raise ValueError("Model scores do not sum to one.")
    label = max(probabilities, key=lambda name: probabilities[name])
    coverage = {
        "input_tokens": usage.get("input_tokens"),
        "state_tokens": usage.get("state_tokens"),
        "state_tokens_dropped": dropped,
        "truncated": usage["truncated"],
        "truncated_questions": usage["truncated_questions"],
    }
    validate_result(
        {
            "label": label,
            "probabilities": dict(probabilities),
            "usage": usage,
            "coverage": coverage,
        }
    )
    return {
        "label": label,
        "probabilities": probabilities,
        "usage": usage,
        "coverage": coverage,
        "analysis_time_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the worker parser (also used for --help checks)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="Optional default checkpoint; load commands carry the entry.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--laya-root", default=None)
    return parser


def main() -> None:
    """Run the persistent JSON-lines worker until shutdown."""
    args = build_parser().parse_args()
    session_id = uuid.uuid4().hex
    agent: Any = None
    loaded_id: str | None = None
    loaded_device: str | None = None
    loaded_questions: dict[str, Any] | None = None
    loaded_contract: str | None = None
    loaded_identity: dict[str, Any] | None = None
    default_checkpoint = args.checkpoint
    default_device = args.device
    default_laya = args.laya_root or os.environ.get("LAYA_ROOT")

    def progress(request_id: str, stage: str, detail: str) -> None:
        _progress(session_id, request_id, stage, detail)

    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            command = json.loads(text)
        except ValueError as exc:
            _emit(
                {
                    "type": "error",
                    "request_id": "",
                    "stage": "protocol",
                    "detail": "Worker received non-JSON input: %s" % exc,
                },
                session_id,
            )
            continue
        if not isinstance(command, dict):
            _emit(
                {
                    "type": "error",
                    "request_id": "",
                    "stage": "protocol",
                    "detail": "Worker command must be an object.",
                },
                session_id,
            )
            continue
        kind = command.get("type")
        request_id = command.get("request_id")
        if not isinstance(request_id, str) and kind != "shutdown":
            _emit(
                {
                    "type": "error",
                    "request_id": "",
                    "stage": "protocol",
                    "detail": "Worker command is missing its request id.",
                },
                session_id,
            )
            continue
        try:
            req_id = request_id if isinstance(request_id, str) else ""
            if kind == "load":
                # A failed replacement must not leave the prior agent usable.
                agent = None
                loaded_questions = None
                loaded_identity = None
                model = command.get("model")
                if model is None:
                    if default_checkpoint is None:
                        raise ValueError("Load has no model entry.")
                    model = {
                        "id": "default",
                        "checkpoint": default_checkpoint,
                        "device": default_device,
                        "expected_files": {},
                    }
                if not isinstance(model, dict):
                    raise ValueError("Load model entry must be an object.")
                checked = _checked_model({"model": model})
                checkpoint = Path(str(checked["checkpoint"]))
                device = str(checked["device"])
                expected = checked.get("expected_files", {})
                if not isinstance(expected, dict):
                    raise ValueError(
                        "Model entry expected_files must be an object."
                    )
                laya_root = checked.get("laya_root") or default_laya
                if isinstance(laya_root, Path):
                    laya_root = str(laya_root)
                if not checkpoint.exists():
                    raise ValueError(
                        "Checkpoint does not exist: %s" % checkpoint
                    )
                if expected:
                    _verify_expected(
                        checkpoint,
                        expected,
                        lambda s, d, _r=req_id: progress(_r, s, d),
                    )
                staged, _fp = _stage_checkpoint(
                    checkpoint,
                    lambda s, d, _r=req_id: progress(_r, s, d),
                )
                if expected:
                    _verify_expected(
                        staged,
                        expected,
                        lambda s, d, _r=req_id: progress(_r, s, d),
                    )
                sealed_files = expected or _tree_hashes(staged)
                started = time.monotonic()
                laya_arg = (
                    str(laya_root)
                    if isinstance(laya_root, (str, Path))
                    else None
                )
                agent = _load_agent(
                    staged,
                    device,
                    laya_arg,
                    lambda s, d, _r=req_id: progress(_r, s, d),
                )
                _verify_placement(agent, device)
                staged_questions, staged_contract = questions_for_staged(staged)
                _warm_up(
                    agent,
                    lambda s, d, _r=req_id: progress(_r, s, d),
                    staged_questions,
                )
                _verify_placement(agent, device)
                load_time = time.monotonic() - started
                loaded_id = str(checked["id"])
                loaded_device = device
                loaded_questions = staged_questions
                loaded_contract = staged_contract
                if (
                    checked.get("question_contract", staged_contract)
                    != staged_contract
                ):
                    loaded_questions = None
                    raise ValueError(
                        "Loaded question contract differs from registry"
                    )
                import torch

                runtime_root = laya_arg or str(
                    Path(_check_laya_runtime(None)).parent.parent
                )
                loaded_identity = {
                    "checkpoint_sha256": sealed_files.get("model.safetensors"),
                    "export_sha256": digest(sealed_files),
                    "question_contract": staged_contract,
                    "question_sha256": digest(staged_questions),
                    "runtime_sha256": runtime_digest(runtime_root),
                    "inference": {
                        "backend": "laya.predict_batch",
                        "batch_size": 1,
                        "sort_by_length": False,
                        "option_order": [0, 1, 2],
                        "precision": str(agent.dtype),
                        "amp_enabled": bool(agent.amp_enabled),
                        "torch_version": str(torch.__version__),
                        "device_type": device.split(":")[0],
                        "max_len": int(agent.cfg.get("max_len", 512)),
                        "head_max_len": int(agent.cfg.get("head_max_len", 192)),
                        "fast_backend": getattr(agent, "_fast", None)
                        is not None,
                    },
                }
                _log(
                    "loaded %s on %s contract %s"
                    % (loaded_id, device, loaded_contract)
                )
                _emit(
                    {
                        "type": "ready",
                        "request_id": req_id,
                        "model_id": loaded_id,
                        "device": device,
                        "question_contract": loaded_contract,
                        "execution_identity": loaded_identity,
                        "cpu_fallback_count": int(
                            getattr(agent, "cpu_fallback_count", 0) or 0
                        ),
                        "load_time_seconds": load_time,
                    },
                    session_id,
                )
            elif kind == "analyze":
                if agent is None or loaded_questions is None:
                    raise ValueError("No model is loaded.")
                state = command.get("state")
                if not isinstance(state, str) or not state:
                    raise ValueError("Analyze command is missing state.")
                outcome = _run_analyze(agent, state, loaded_questions)
                _emit(
                    {
                        "type": "result",
                        "request_id": req_id,
                        "label": outcome["label"],
                        "probabilities": outcome["probabilities"],
                        "usage": outcome["usage"],
                        "coverage": outcome["coverage"],
                        "device": loaded_device,
                        "model_id": loaded_id,
                        "execution_identity": loaded_identity,
                        "state_sha256": text_digest(state),
                    },
                    session_id,
                )
            elif kind == "shutdown":
                break
            else:
                raise ValueError("Unknown worker command: %r." % (kind,))
        except (
            Exception
        ) as exc:  # Endpoint isolation: never crash without a protocol error.
            _log("request failed: %s: %s" % (type(exc).__name__, exc))
            _emit(
                {
                    "type": "error",
                    "request_id": request_id
                    if isinstance(request_id, str)
                    else "",
                    "stage": "load" if kind == "load" else "analyze",
                    "detail": str(exc),
                },
                session_id,
            )
    _log("worker exiting")


if __name__ == "__main__":
    main()
