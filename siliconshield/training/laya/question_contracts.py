"""Versioned question contracts for security decisions.

Each contract pins the exact model-visible question bytes
(instructions + criteria order) behind a version id. Training
runs validate every row against the pinned contract, exports
record it, and inference selects questions from the export's
recorded contract. Legacy exports without a record keep the
original omitted-definitions-only question.
"""

import hashlib
import json
from pathlib import Path

LABELS = [
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
]

LEGACY_CONTRACT = "siliconshield-requirement-context-v0"
NEW_CONTRACT = "siliconshield-requirement-context-v1"

_CONTRACTS = {
    LEGACY_CONTRACT: {
        "instructions": (
            "Assess whether the supplied RTL satisfies the stated "
            "requirement under the assumptions. A reachable "
            "counterexample means violates_requirement. Use "
            "insufficient_context only when omitted definitions can "
            "change the answer. A conditional path alone is not "
            "missing context."
        ),
        "criteria": list(LABELS),
    },
    NEW_CONTRACT: {
        "instructions": (
            "Assess the RTL against the requirement under the "
            "supplied assumptions. Use satisfies_requirement when "
            "the property holds and violates_requirement when a "
            "reachable counterexample exists. Use "
            "insufficient_context when missing implementation, "
            "missing policy information, or unresolved requirement "
            "meaning permits compatible interpretations with "
            "different verdicts. If all compatible interpretations "
            "give the same verdict, use that verdict. A conditional "
            "path, irrelevant omission, or difficulty alone is not "
            "insufficient context."
        ),
        "criteria": list(LABELS),
    },
}


def known_contracts():
    """Sorted version ids in this registry."""
    return sorted(_CONTRACTS)


def question_for(version):
    """Model-visible security question for a contract version."""
    try:
        contract = _CONTRACTS[version]
    except KeyError:
        raise ValueError("unknown question contract: %r" % (version,))
    return {
        "security": {
            "type": "choice",
            "instructions": contract["instructions"],
            "criteria": list(contract["criteria"]),
        }
    }


def digest_for(version):
    """SHA-256 over canonical contract bytes."""
    try:
        contract = _CONTRACTS[version]
    except KeyError:
        raise ValueError("unknown question contract: %r" % (version,))
    canonical = json.dumps(
        {
            "version": version,
            "instructions": contract["instructions"],
            "criteria": contract["criteria"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def match_version(questions):
    """Identify the contract version of row questions, or refuse."""
    try:
        security = questions["security"]
        instructions = security["instructions"]
        criteria = list(security["criteria"])
    except (KeyError, TypeError):
        raise ValueError("row questions lack security contract")
    for version, contract in _CONTRACTS.items():
        if (
            instructions == contract["instructions"]
            and criteria == contract["criteria"]
        ):
            return version
    raise ValueError("row questions match no known contract")


def validate_rows(rows, version):
    """Fail closed unless every row carries the pinned contract."""
    if version not in _CONTRACTS:
        raise ValueError("unknown question contract: %r" % (version,))
    for row in rows:
        found = match_version(row["questions"])
        if found != version:
            raise ValueError(
                "row %s carries %s, run pins %s"
                % (row.get("id"), found, version)
            )
    return version


def write_export_record(export_dir, version):
    """Persist the contract record alongside exported weights."""
    record = {
        "version": version,
        "digest": digest_for(version),
        "instructions": _CONTRACTS[version]["instructions"],
        "criteria": list(_CONTRACTS[version]["criteria"]),
    }
    path = Path(export_dir) / "question_contract.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def read_export_record(export_dir):
    """Contract version for an export; legacy when unrecorded."""
    path = Path(export_dir) / "question_contract.json"
    if not path.exists():
        return {
            "version": LEGACY_CONTRACT,
            "digest": None,
            "legacy_default": True,
        }
    record = json.loads(path.read_text())
    version = record.get("version")
    if version not in _CONTRACTS:
        raise ValueError("export records unknown contract: %r" % (version,))
    if record.get("digest") != digest_for(version):
        raise ValueError("export contract digest mismatch")
    if record.get("instructions") != _CONTRACTS[version]["instructions"]:
        raise ValueError("export contract bytes mismatch")
    return {
        "version": version,
        "digest": record["digest"],
        "legacy_default": False,
    }
