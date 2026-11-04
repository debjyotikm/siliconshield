"""Base-Muse patch proposer for the RTL security workflow.

Receives only the model-facing investigation context (RTL, requirement,
assumptions, security-model assessment, relevant statements, and check
evidence) — never an authored patch or expected verdict. Returns exact
text replacements, which the workflow validates against the RTL before
applying.
"""

import json

from siliconshield.research_loop.controller.client import ChatClient

SYSTEM = (
    "You investigate RTL security findings. Given the RTL, the security requirement, "
    "and executable check evidence of a demonstrated violation, propose the smallest "
    'corrective change. Reply with one JSON object: {"replacements": [{"old": '
    '"exact existing RTL text", "new": "replacement text"}]}. Copy old text '
    "exactly from the RTL. Do not claim checks pass; they will be rerun."
)


def parse_proposal(text):
    """Parse exact replacements; every malformed shape is a ValueError."""
    if not isinstance(text, str):
        raise ValueError("Muse proposal must be text")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n", 1)
        if len(lines) != 2 or "```" not in lines[1]:
            raise ValueError("Muse proposal has an unterminated code fence")
        stripped = lines[1].rsplit("```", 1)[0]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Muse proposal must be a JSON object")
    replacements = value.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("Muse proposal contains no replacements")
    parsed = []
    for item in replacements:
        if not isinstance(item, dict):
            raise ValueError("Muse replacement must be an object")
        old, new = item.get("old"), item.get("new")
        if not isinstance(old, str) or not old:
            raise ValueError("Muse replacement needs exact old text")
        if not isinstance(new, str):
            raise ValueError("Muse replacement needs new text")
        parsed.append((old, new))
    return parsed


def make_muse_proposer(endpoint, model, client=None, max_tokens=4096):
    client = client or ChatClient(endpoint, model, max_tokens=max_tokens)

    def proposer(context):
        message = client.complete(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(context, indent=2)},
            ]
        )
        return parse_proposal(message.get("content", ""))

    proposer.__name__ = "muse:%s" % model
    return proposer
