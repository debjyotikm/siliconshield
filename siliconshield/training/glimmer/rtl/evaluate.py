"""Development metrics for three-way decisions and linked whole triplets."""

import collections
import math


def summarize(rows):
    """Report per-domain confusion, recalls, false-safes and triplets."""
    if not rows:
        raise ValueError("No evaluation rows")
    result = {}
    for domain in sorted({r["domain"] for r in rows}):
        subset = [r for r in rows if r["domain"] == domain]
        cm = [[0] * 3 for _ in range(3)]
        groups = collections.defaultdict(list)
        for row in subset:
            cm[row["label"]][row["pred"]] += 1
            groups[row["pair_id"]].append(row)
        if any(
            len(g) != 3 or {r["label"] for r in g} != {0, 1, 2}
            for g in groups.values()
        ):
            raise ValueError("Evaluation must contain complete triplets")
        result[domain] = {
            "rows": len(subset),
            "correct": sum(cm[k][k] for k in range(3)),
            "confusion": cm,
            "class_recall": [cm[k][k] / sum(cm[k]) for k in range(3)],
            "false_safe": cm[1][0],
            "violation_count": sum(cm[1]),
            "triplets": len(groups),
            "triplets_correct": sum(
                all(r["label"] == r["pred"] for r in g) for g in groups.values()
            ),
            "nll": sum(
                -math.log(max(r["probabilities"][r["label"]], 1e-30))
                for r in subset
            )
            / len(subset),
        }
    return result


def batch_inputs(rows, device, pad):
    """Left-pad prompts and explicitly derive positions for final-token loss."""
    import torch

    size = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), size), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[i, -length:] = torch.tensor(row["input_ids"], device=device)
        mask[i, -length:] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return dict(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        logits_to_keep=1,
    )


def evaluate(model, rows, device, pad, label_ids):
    """Score one row at a time with conditional answer probabilities."""
    import torch

    model.eval()
    results = []
    with torch.inference_mode():
        for row in rows:
            logits = (
                model(**batch_inputs([row], device, pad)).logits[0, -1].float()
            )
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite evaluation logits")
            options = logits[label_ids]
            probs = options.softmax(-1).cpu().tolist()
            results.append(
                {
                    **{k: row[k] for k in ("id", "domain", "pair_id", "label")},
                    "pred": max(range(3), key=lambda k: probs[k]),
                    "probabilities": probs,
                    "label_mass": float(
                        (options.logsumexp(0) - logits.logsumexp(0)).exp()
                    ),
                }
            )
    return {"metrics": summarize(results), "rows": results}
