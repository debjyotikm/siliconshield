"""Shared multi-expert inference service with no Torch import.

One interface serves the CLI, Muse, and the browser demo over
persistent worker processes. The service owns registry loading,
capability profiles, routing, per-request identity, timeouts, and
worker failures; model execution stays in the worker back end.
Transports are injected so CPU tests use fakes while production
uses the subprocess worker pool.
"""

import copy
import time
import uuid
from collections.abc import Mapping
from typing import TypedDict

from siliconshield.experts import registry
from siliconshield.experts.capabilities import (
    CapabilityProfile,
    load_capability_profile,
    validate_capability_profile,
)
from siliconshield.experts.identity import digest, text_digest
from siliconshield.experts.responses import (
    ExpertError,
    ExpertResponse,
    Transport,
    consult_experts,
)
from siliconshield.experts.router import RoutingDecision, route_scope
from siliconshield.schemas import EXPERT_CAPABILITY_SCHEMA


class Assessment(TypedDict):
    """One scoped multi-expert assessment with identity."""

    request_id: str
    scope: list[str]
    state_sha256: str
    scope_sha256: str
    planned_routing: RoutingDecision
    routing: RoutingDecision
    responses: list[ExpertResponse]
    errors: list[ExpertError]
    elapsed_seconds: float


class InferenceService:
    """Registry-backed expert consultation over an injected transport."""

    def __init__(
        self,
        registry_path: str,
        profiles: Mapping[str, CapabilityProfile] | None = None,
    ) -> None:
        """Load models and capability profiles; no workers spawn here."""
        self._models = {
            entry["id"]: entry
            for entry in registry.load_registry(registry_path)
        }
        resolved: dict[str, CapabilityProfile] = {}
        if profiles is not None:
            resolved.update(profiles)
        for model_id, entry in self._models.items():
            profile_path = entry.get("profile")
            if profile_path and model_id not in resolved:
                resolved[model_id] = load_capability_profile(profile_path)
        for model_id, raw in resolved.items():
            if model_id not in self._models:
                raise ValueError("Profile names an unregistered expert")
            profile = validate_capability_profile(
                {**raw, "schema": EXPERT_CAPABILITY_SCHEMA}
            )
            entry = self._models[model_id]
            if (
                profile["id"] != model_id
                or profile["checkpoint_sha256"]
                != entry["expected_files"].get("model.safetensors")
                or profile["question_contract"]
                != entry.get("question_contract")
                or profile["release"] != entry.get("release")
            ):
                raise ValueError("Profile identity differs from registry")
            resolved[model_id] = copy.deepcopy(profile)
        self._profiles = resolved

    @property
    def expert_ids(self) -> list[str]:
        """Ids of models with a validated capability profile."""
        return [
            model_id for model_id in self._models if model_id in self._profiles
        ]

    def model_entry(self, expert_id: str) -> registry.ModelEntry:
        """Return the validated registry entry for one expert id."""
        return copy.deepcopy(self._models[expert_id])

    def profiles_for(
        self, expert_ids: list[str]
    ) -> dict[str, CapabilityProfile]:
        """Return capability profiles for the given expert ids."""
        return {
            expert_id: copy.deepcopy(self._profiles[expert_id])
            for expert_id in expert_ids
            if expert_id in self._profiles
        }

    def route(self, scope: list[str]) -> RoutingDecision:
        """Plan consultation without loading any workers."""
        return route_scope(
            scope, [self._profiles[key] for key in self.expert_ids]
        )

    def assess(
        self,
        scope: list[str],
        state: str,
        transport: Transport,
    ) -> Assessment:
        """Route one scope and consult every qualified expert.

        The same request id and state go to each expert; responses
        and failures return side by side. Raises ValueError for
        empty state text.
        """
        started = time.monotonic()
        decision = self.route(scope)
        request_id = uuid.uuid4().hex
        scope_hash = digest(decision["scope"])
        responses, errors = consult_experts(
            decision["experts"], self._profiles, state, transport
        )
        for response in responses:
            response["assessment_id"] = request_id
            response["scope_sha256"] = scope_hash
        effective = route_scope(
            decision["scope"],
            [self._profiles[item["expert_id"]] for item in responses],
            {
                item["expert_id"]: item["execution_identity"]
                for item in responses
            },
        )
        return {
            "request_id": request_id,
            "state_sha256": text_digest(state),
            "scope_sha256": scope_hash,
            "planned_routing": decision,
            "scope": decision["scope"],
            "routing": effective,
            "responses": responses,
            "errors": errors,
            "elapsed_seconds": time.monotonic() - started,
        }
