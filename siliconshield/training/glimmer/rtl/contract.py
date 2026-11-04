"""Native chat prompt and answer-token contract."""

LABELS = (
    "satisfies_requirement",
    "violates_requirement",
    "insufficient_context",
)


def encode(tokenizer, state, contract):
    """Encode the prompt up to the final answer, rejecting clipped inputs."""
    sentinel = "SENTINEL_ANSWER"
    if sentinel in state:
        raise ValueError("Input contains reserved answer sentinel")
    spec = contract["template"]
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": spec["system"]},
            {"role": "user", "content": spec["user_prefix"] + state},
            {"role": "assistant", "content": sentinel},
        ],
        tokenize=False,
        add_generation_prompt=False,
        reasoning_strength=spec["reasoning_strength"],
        date_string=spec["date_string"],
    )
    if text.count(sentinel) != 1:
        raise ValueError("Chat template did not preserve the answer sentinel")
    ids = tokenizer.encode(text.split(sentinel)[0], add_special_tokens=False)
    if not ids or len(ids) > contract["max_length"]:
        raise ValueError("Empty or over-budget prompt; truncation is forbidden")
    return ids


def validate_tokens(tokenizer, contract):
    """Require the declared three-class ordering and native single tokens."""
    if contract["label_order"] != list(LABELS):
        raise ValueError("Unsupported label order")
    tokens = [tokenizer.encode(x, add_special_tokens=False) for x in "ABC"]
    if any(len(x) != 1 for x in tokens):
        raise ValueError("Answers must each encode to exactly one token")
    ids = [x[0] for x in tokens]
    if len(set(ids)) != 3 or ids != contract["label_token_ids"]:
        raise ValueError("Tokenizer does not match the decision contract")
    return ids
