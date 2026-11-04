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
        """Load models and capability profiles; no workers spawn here.

        Caller-supplied profiles fail fast on any defect. Stored
        profile files that fail to load, validate, or bind are
        recorded in broken_experts instead: one corrupt registry
        entry never blocks scoping a healthy expert.
        """
        self._models = {
            entry["id"]: entry
            for entry in registry.load_registry(registry_path)
        }
        self._broken: dict[str, str] = {}
        injected = set(profiles) if profiles is not None else set()
        resolved: dict[str, CapabilityProfile] = {}
        if profiles is not None:
            for model_id, raw in profiles.items():
                if not isinstance(raw, Mapping):
                    raise ValueError(
                        "Injected profile %r must be an object" % (model_id,)
                    )
                entry = self._models.get(model_id)
                if (
                    entry is not None
                    and entry.get("profile")
                    and model_id not in resolved
                ):
                    raise ValueError(
                        "Injected profile %r collides with the registry "
                        "profile file; remove one source." % (model_id,)
                    )
                resolved[model_id] = raw
        for model_id, entry in self._models.items():
            profile_path = entry.get("profile")
            if profile_path and model_id not in resolved:
                try:
                    resolved[model_id] = load_capability_profile(profile_path)
                except ValueError as exc:
                    self._broken[model_id] = "%s" % exc
        bound: dict[str, CapabilityProfile] = {}
        for model_id, raw in resolved.items():
            if model_id not in self._models:
                raise ValueError("Profile names an unregistered expert")
            try:
                bound[model_id] = self._bind_profile(model_id, raw)
            except ValueError as exc:
                if model_id in injected:
                    raise
                self._broken[model_id] = "%s" % exc
        self._profiles = bound

    def _bind_profile(
        self, model_id: str, raw: CapabilityProfile
    ) -> CapabilityProfile:
        """Validate one profile and bind it to its registry entry."""
        incoming = dict(raw)
        if (
            incoming.get("schema", EXPERT_CAPABILITY_SCHEMA)
            != EXPERT_CAPABILITY_SCHEMA
        ):
            raise ValueError(
                "Profile %r has an unsupported schema" % (model_id,)
            )
        profile = validate_capability_profile(
            {**incoming, "schema": EXPERT_CAPABILITY_SCHEMA}
        )
        entry = self._models[model_id]
        stated_weights = entry["expected_files"].get("model.safetensors")
        if stated_weights is None:
            raise ValueError(
                "Registry must state model.safetensors to bind "
                "a profile for %r" % (model_id,)
            )
        stated_contract = entry.get("question_contract")
        stated_release = entry.get("release")
        stated_qualification = entry.get("qualification")
        mismatches = []
        if profile["id"] != model_id:
            mismatches.append(
                "id: profile %r != registry %r" % (profile["id"], model_id)
            )
        if profile["checkpoint_sha256"] != stated_weights:
            mismatches.append(
                "checkpoint_sha256: profile %r != registry "
                "model.safetensors %r"
                % (profile["checkpoint_sha256"], stated_weights)
            )
        for field, stated, expected in (
            (
                "question_contract",
                stated_contract,
                profile["question_contract"],
            ),
            ("release", stated_release, profile["release"]),
            (
                "qualification",
                stated_qualification,
                profile["qualification_ref"],
            ),
        ):
            if stated is not None and stated != expected:
                mismatches.append(
                    "%s: registry %r != profile %r" % (field, stated, expected)
                )
        if mismatches:
            raise ValueError(
                "Profile identity differs from registry for %r: %s. "
                "Copy the profile values into the registry entry, or "
                "remove the registry fields to backfill them from the "
                "profile." % (model_id, ", ".join(mismatches))
            )
        # Backfill unstated registry fields from the bound profile
        # so worker loading and consult-time checks use the same
        # contract, release, and qualification the profile states.
        if stated_contract is None:
            entry["question_contract"] = profile["question_contract"]
        if stated_release is None:
            entry["release"] = profile["release"]
        if stated_qualification is None:
            entry["qualification"] = profile["qualification_ref"]
        return copy.deepcopy(profile)

    @property
    def expert_ids(self) -> list[str]:
        """Ids of models with a validated capability profile."""
        return [
            model_id for model_id in self._models if model_id in self._profiles
        ]

    @property
    def broken_experts(self) -> dict[str, str]:
        """Stored-profile failures by model id, with reasons.

        These experts are excluded from routing. Callers must
        surface this mapping wherever they show the roster: a
        silently missing expert is worse than a loud one.
        """
        return dict(self._broken)

    @property
    def unprofiled_experts(self) -> list[str]:
        """Registered model ids with no profile from any source."""
        return [
            model_id
            for model_id in self._models
            if model_id not in self._profiles and model_id not in self._broken
        ]

    def model_entry(self, expert_id: str) -> registry.ModelEntry:
        """Return the validated registry entry for one expert id."""
        return copy.deepcopy(self._models[expert_id])

    def profiles_for(
        self, expert_ids: list[str]
    ) -> dict[str, CapabilityProfile]:
        """Return capability profiles for the given expert ids.

        Unknown ids are skipped rather than raising, unlike
        model_entry: roster views show None for profile-less
        models instead of failing the whole listing.
        """
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
        # Failures narrow who answered, never what the plan covered:
        # keep the planned scope picture so split-scope guidance
        # survives exactly when callers need it most.
        for key in ("unsupported", "partial", "coverage"):
            effective[key] = decision[key]
        effective["rationale"] = list(decision["rationale"]) + [
            "%s failed during consult (%s); planned coverage above "
            "is unchanged." % (item["expert_id"], item["stage"])
            for item in errors
        ]
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
