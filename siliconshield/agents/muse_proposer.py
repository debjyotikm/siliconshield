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
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    value = json.loads(stripped)
    replacements = value.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("Muse proposal contains no replacements")
    return [(item["old"], item["new"]) for item in replacements]


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
