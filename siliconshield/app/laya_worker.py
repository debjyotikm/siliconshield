"""SiliconShield-RTL assessment worker on the LAYA runtime.

Run with the Python that owns LAYA + torch. Reads one JSON object on
stdin ({state, checkpoint}) and prints one JSON object ({label,
probabilities, usage}) on stdout. Kept dependency-light so the main
workflow never imports LAYA core.
"""

import copy
import json
import os
import sys

from siliconshield.training.laya.question_contracts import (
    LABELS as CONTRACT_LABELS,
)
from siliconshield.training.laya.question_contracts import (
    LEGACY_CONTRACT,
    question_for,
    read_export_record,
)

LABELS = list(CONTRACT_LABELS)
QUESTION = question_for(LEGACY_CONTRACT)


def questions_for_checkpoint(checkpoint):
    """Return versioned questions for a checkpoint export.

    Reads the export's recorded question contract and returns
    ``(questions, version)`` with a deterministic option order.
    Single-file checkpoints read the record from their parent
    export directory; without that, a v1 single file would
    silently fall back to the legacy questions its weights never
    trained on. Legacy exports without a record resolve to the
    original omitted-definitions-only contract.
    """
    record_dir = checkpoint
    if os.path.isfile(checkpoint):
        record_dir = os.path.dirname(os.path.abspath(checkpoint))
    record = read_export_record(record_dir)
    questions = copy.deepcopy(question_for(record["version"]))
    questions["security"]["option_order"] = list(range(len(CONTRACT_LABELS)))
    return questions, record["version"]


def main():
    request = json.loads(sys.stdin.read())
    from laya.agent import load

    questions, _ = questions_for_checkpoint(request["checkpoint"])
    agent = load(request["checkpoint"])
    answers = agent.predict_batch(
        [request["state"]], questions, sort_by_length=False
    )
    answer = answers[0]
    values = answer["answers"]["security"]["probabilities"]
    probabilities = {label: float(values[label]) for label in LABELS}
    label = max(probabilities, key=probabilities.get)
    print(
        json.dumps(
            {
                "label": label,
                "probabilities": probabilities,
                "usage": answer.get("usage", {}),
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
