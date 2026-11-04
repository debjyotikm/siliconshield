"""Consult fine-tuned scoring adapters behind the router flow.

This is the same router flow as the chat path, with one difference:
each consultation scores the state through a served adapter and the
verdict comes from normalized answer-token scores, never from parsed
prose. The skill document is still digest-checked on every call, and
the serving fingerprint must match the registry and profile binding
exactly.
"""

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from siliconshield.experts.contracts import SCORE_SUM_TOLERANCE
from siliconshield.experts.identity import digest, text_digest
from siliconshield.training.laya.question_contracts import (
    ADAPTER_CONTRACT,
    digest_for,
)

LABELS = (
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
)


class AdapterTransport:
    """Consult registry experts through a persistent scoring shim."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        fingerprint: str,
        entries: dict[str, Any],
        timeout: int = 300,
        progress: Any = None,
    ) -> None:
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError("Adapter endpoint must be a non-empty string")
        if not endpoint.endswith("/score"):
            raise ValueError("Adapter endpoint must end in /score")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A served adapter model name is required")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise ValueError("Adapter fingerprint must be a non-empty string")
        self._endpoint = endpoint
        self._model = model
        self._fingerprint = fingerprint
        self._entries = entries
        self._timeout = timeout
        self._progress = progress
        self._question_sha256 = digest_for(ADAPTER_CONTRACT)
        self._runtime_sha256 = digest({"endpoint": endpoint, "model": model})
        self.calls: list[dict[str, Any]] = []

    def _say(self, text: str) -> None:
        if self._progress is not None:
            self._progress(text)

    def _skill(self, entry: dict[str, Any]) -> tuple[str, str]:
        """Re-check the pinned skill document on every call."""
        try:
            skill = (Path(entry["checkpoint"]) / "SKILL.md").read_text()
        except OSError as exc:
            raise ValueError(
                "Refusing to score: cannot read the skill: %s" % (exc,)
            )
        skill_digest = hashlib.sha256(skill.encode("utf-8")).hexdigest()
        if skill_digest != entry["expected_files"]["SKILL.md"]:
            raise ValueError(
                "Refusing to score: the skill changed under a pinned registry."
            )
        return skill, skill_digest

    def __call__(self, expert_id: str, state: str) -> dict[str, Any]:
        entry = self._entries[expert_id]
        skill, skill_digest = self._skill(entry)
        number = len(self.calls) + 1
        self._say(
            "Call %d: POST %s model %s skill %s. Waiting on the adapter."
            % (number, self._endpoint, self._model, skill_digest[:12])
        )
        started = time.monotonic()
        try:
            reply = self._post(state)
        except Exception as exc:
            seconds = time.monotonic() - started
            self.calls.append(
                {
                    "endpoint": self._endpoint,
                    "model": self._model,
                    "seconds": seconds,
                    "error": str(exc),
                }
            )
            self._say("Call %d failed after %.1fs: %s" % (number, seconds, exc))
            raise
        seconds = time.monotonic() - started
        label, scores = self._checked_scores(reply)
        record = {
            "endpoint": self._endpoint,
            "model": self._model,
            "seconds": seconds,
            "kind": "score",
            "request_messages": [{"role": "user", "chars": len(state)}],
            "skill_sha256": skill_digest,
            "response_chars": len(json.dumps(reply)),
            "response": json.dumps(reply, indent=2),
        }
        self.calls.append(record)
        self._say("Call %d finished in %.1fs: %s." % (number, seconds, label))
        hardware = reply["hardware"]
        device_type = (
            "%dx %s" % (len(hardware), hardware[0])
            if len(set(hardware)) == 1
            else "; ".join(hardware)
        )
        return {
            "label": label,
            "probabilities": scores,
            "coverage": {
                "truncated": False,
                "state_tokens_dropped": 0,
                "truncated_questions": [],
            },
            "model_id": expert_id,
            "execution_identity": {
                "checkpoint_sha256": self._fingerprint,
                "export_sha256": digest(
                    {"model": self._model, "skill_sha256": skill_digest}
                ),
                "question_sha256": self._question_sha256,
                "runtime_sha256": self._runtime_sha256,
                "question_contract": ADAPTER_CONTRACT,
                "inference": {
                    "backend": "adapter.score",
                    "precision": reply["precision"],
                    "device_type": device_type,
                    "torch_version": "remote",
                    "batch_size": 1,
                    "max_len": 2048,
                    "head_max_len": 1,
                    "sort_by_length": False,
                    "option_order": [0, 1, 2],
                },
            },
            "state_sha256": text_digest(state),
        }

    def _post(self, state: str) -> dict[str, Any]:
        """Score one state through the shim without truncation."""
        body = json.dumps({"state": state}).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._timeout
            ) as reply:
                payload = json.loads(reply.read())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", "")
            except (ValueError, TypeError):
                detail = ""
            raise ValueError(
                "Scoring shim refused the input: %s" % (detail or exc,)
            )
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError("Scoring shim is unreachable: %s" % (exc,))
        if not isinstance(payload, dict):
            raise ValueError("Scoring shim returned a malformed reply")
        return payload

    def _checked_scores(
        self, reply: dict[str, Any]
    ) -> tuple[str, dict[str, float]]:
        """Accept only normalized finite scores with a top label."""
        label = reply.get("label")
        scores = reply.get("scores")
        if label not in LABELS:
            raise ValueError("Scoring shim returned an unknown label")
        if not isinstance(scores, dict) or set(scores) != set(LABELS):
            raise ValueError("Scoring shim returned malformed scores")
        numbers: dict[str, float] = {}
        for name, value in scores.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= value <= 1.0
                or value != value
            ):
                raise ValueError("Scoring shim returned a bad score")
            numbers[name] = float(value)
        if abs(sum(numbers.values()) - 1.0) > SCORE_SUM_TOLERANCE:
            raise ValueError("Scoring shim scores do not sum to one")
        best = max(LABELS, key=lambda name: numbers[name])
        if label != best:
            raise ValueError("Scoring shim label is not the top score")
        for field in ("precision", "hardware"):
            if field not in reply:
                raise ValueError("Scoring shim omitted %s" % field)
        if not isinstance(reply["precision"], str) or not reply["precision"]:
            raise ValueError("Scoring shim reported a bad precision")
        hardware = reply["hardware"]
        if (
            not isinstance(hardware, list)
            or not hardware
            or any(not isinstance(item, str) or not item for item in hardware)
        ):
            raise ValueError("Scoring shim reported bad hardware")
        return label, numbers
