"""Traceable multi-expert findings (Qt-free).

A finding binds the suspected issue to its functional or security
relevance, the requirement judged, RTL locations, consulted
experts, the arbitration verdict, and unresolved questions. Every
conclusion is labeled a model assessment: simulation and formal
verification stay deferred, explicit next steps rather than
implied guarantees.
"""

from typing import Any, NotRequired, TypedDict

from siliconshield.experts.identity import text_digest


class FailedExpert(TypedDict):
    """One consulted expert that produced no usable response."""

    expert_id: str
    stage: str
    detail: NotRequired[str]


class Finding(TypedDict):
    """One traceable assessment finding."""

    issue: str
    relevance: str
    requirement: str
    rtl_locations: list[str]
    experts_consulted: list[str]
    failed_experts: list[FailedExpert]
    expert_labels: dict[str, str]
    conclusion: str
    explanation: str
    open_questions: list[str]
    agreement: bool
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

    Raises ValueError when the issue, relevance, requirement,
    arbitration conclusion, explanation, agreement flag, or usable
    open questions are missing: findings must never paper over
    absent evidence with a confident label.
    """
    if not isinstance(issue, str) or not issue.strip():
        raise ValueError("Finding needs a suspected issue.")
    if not isinstance(relevance, str) or not relevance.strip():
        raise ValueError("Finding needs functional/security relevance.")
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("Finding needs the requirement judged.")
    if not isinstance(assessment, dict) or not isinstance(arbitration, dict):
        raise ValueError("Finding needs assessment and arbitration objects.")
    if not isinstance(rtl_locations, list) or any(
        not isinstance(item, str) for item in rtl_locations
    ):
        raise ValueError("Finding RTL locations must be strings.")
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
    if not isinstance(responses, list):
        raise ValueError("Finding assessment responses must be a list.")
    labels: dict[str, str] = {}
    consulted: list[str] = []
    for response in responses:
        if not isinstance(response, dict):
            raise ValueError("Finding assessment response must be an object.")
        expert_id = response.get("expert_id")
        label = response.get("label")
        if not isinstance(expert_id, str) or not expert_id:
            raise ValueError("Finding response names no expert.")
        if not isinstance(label, str) or not label:
            raise ValueError("Finding response has no label.")
        if expert_id in labels:
            raise ValueError(
                "Finding has duplicate responses from %r: a conflict "
                "must be arbitrated, never silently last-wins." % (expert_id,)
            )
        labels[expert_id] = label
        consulted.append(expert_id)
    raw_errors = assessment.get("errors", [])
    if not isinstance(raw_errors, list):
        raise ValueError("Finding assessment errors must be a list.")
    failed: list[FailedExpert] = []
    for error in raw_errors:
        if not isinstance(error, dict):
            raise ValueError("Finding assessment error must be an object.")
        failed_id = error.get("expert_id")
        stage = error.get("stage")
        if not isinstance(failed_id, str) or not failed_id:
            raise ValueError("Finding error names no expert.")
        if not isinstance(stage, str) or not stage:
            raise ValueError("Finding error has no stage.")
        failed_entry: FailedExpert = {
            "expert_id": failed_id,
            "stage": stage,
        }
        detail = error.get("detail")
        if isinstance(detail, str) and detail:
            failed_entry["detail"] = detail
        failed.append(failed_entry)
    explanation = arbitration.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Finding needs an arbitration explanation.")
    raw_questions = arbitration.get("open_questions", [])
    if not isinstance(raw_questions, list) or any(
        not isinstance(item, str) or not item.strip() for item in raw_questions
    ):
        raise ValueError("Finding open questions must be strings.")
    questions = list(raw_questions)
    questions.append(
        "Confirm by simulation or formal verification; "
        "this conclusion is a model assessment only."
    )
    agreement = arbitration.get("agreement")
    if not isinstance(agreement, bool):
        raise ValueError("Finding needs an arbitration agreement flag.")
    return {
        "issue": issue,
        "relevance": relevance,
        "requirement": requirement,
        "rtl_locations": list(rtl_locations),
        "experts_consulted": consulted,
        "failed_experts": failed,
        "expert_labels": labels,
        "conclusion": conclusion,
        "explanation": explanation,
        "open_questions": questions,
        "agreement": agreement,
        "assessment_kind": "model_assessment",
    }
