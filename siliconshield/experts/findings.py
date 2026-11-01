"""Traceable multi-expert findings (Qt-free).

A finding binds the suspected issue to its functional or security
relevance, the requirement judged, RTL locations, consulted
experts, the arbitration verdict, and unresolved questions. Every
conclusion is labeled a model assessment: simulation and formal
verification stay deferred, explicit next steps rather than
implied guarantees.
"""

from typing import Any, TypedDict

from siliconshield.experts.identity import text_digest


class Finding(TypedDict):
    """One traceable assessment finding."""

    issue: str
    relevance: str
    requirement: str
    rtl_locations: list[str]
    experts_consulted: list[str]
    expert_labels: dict[str, str]
    conclusion: str
    explanation: str
    open_questions: list[str]
    assessment_kind: str


def build_finding(
    issue: str,
    relevance: str,
    requirement: str,
    rtl_locations: list[str],
    assessment: dict[str, Any],
    arbitration: dict[str, Any],
) -> Finding:
    """Assemble one finding from assessment and arbitration outputs.

    Raises ValueError when the issue, relevance, requirement, or
    arbitration conclusion is missing: findings must never paper
    over absent evidence with a confident label.
    """
    if not issue.strip():
        raise ValueError("Finding needs a suspected issue.")
    if not relevance.strip():
        raise ValueError("Finding needs functional/security relevance.")
    if not requirement.strip():
        raise ValueError("Finding needs the requirement judged.")
    conclusion = arbitration.get("conclusion")
    if not conclusion or not isinstance(conclusion, str):
        raise ValueError("Finding needs an arbitration conclusion.")
    for key in ("request_id", "state_sha256", "scope_sha256"):
        if not assessment.get(key) or arbitration.get(key) != assessment[key]:
            raise ValueError(
                "Finding arbitration belongs to a different assessment"
            )
    if arbitration.get("requirement_sha256") != text_digest(
        " ".join(requirement.split())
    ):
        raise ValueError("Finding requirement differs from arbitration")
    responses = assessment.get("responses", [])
    labels = {
        response["expert_id"]: response["label"] for response in responses
    }
    consulted = [response["expert_id"] for response in responses]
    for error in assessment.get("errors", []):
        consulted.append(
            "%s (failed: %s)" % (error["expert_id"], error["stage"])
        )
    questions = list(arbitration.get("open_questions", []))
    questions.append(
        "Confirm by simulation or formal verification; "
        "this conclusion is a model assessment only."
    )
    return {
        "issue": issue,
        "relevance": relevance,
        "requirement": requirement,
        "rtl_locations": list(rtl_locations),
        "experts_consulted": consulted,
        "expert_labels": labels,
        "conclusion": conclusion,
        "explanation": str(arbitration.get("explanation", "")),
        "open_questions": questions,
        "assessment_kind": "model_assessment",
    }
