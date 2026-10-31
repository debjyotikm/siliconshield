"""Format Muse conversations and mask loss to assistant responses."""

import re

from siliconshield.research_loop.episodes.contracts import validate_episode

TURN = re.compile(
    r"<\|start\|>([^<]*)<\|message\|>(.*?)(<\|eo[mt]\|>)", re.DOTALL
)


def encode_episode(tokenizer, row, max_length):
    validate_episode(row)
    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=False,
        reasoning_strength="high",
    )
    encoded = tokenizer(
        rendered, add_special_tokens=False, return_offsets_mapping=True
    )
    ids, offsets = encoded["input_ids"], encoded.pop("offset_mapping")
    if len(ids) > max_length:
        raise ValueError(
            f"Episode {row['id']} needs {len(ids)} tokens, exceeds {max_length}; truncation disabled"
        )
    assistant_spans = [
        (m.start(), m.end())
        for m in TURN.finditer(rendered)
        if m.group(1).split()[0] == "assistant"
    ]
    labels = [
        token
        if end > start
        and any(start >= a and end <= b for a, b in assistant_spans)
        else -100
        for token, (start, end) in zip(ids, offsets)
    ]
    if not any(x != -100 for x in labels):
        raise ValueError(
            "Native chat template produced no assistant loss tokens"
        )
    return {
        "input_ids": ids,
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }
