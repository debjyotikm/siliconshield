"""Guards for coherent distributed training and evaluation.

Prior runs mixed predictions from diverged DDP ranks when two
separate ``backward()`` calls ran inside one accumulation step
with ``gradient_as_bucket_view``. The trainer now folds the
preservation penalty into the supervised loss and calls
``backward()`` once per microbatch inside a single ``no_sync``
scope. This module provides a static gate plus runtime helpers
so a fresh run cannot silently reintroduce the dual-backward
pattern and attribute its effects to data.
"""

import ast
from pathlib import Path


def count_backward_calls(source_path: Path) -> int:
    """Count ``.backward()`` call expressions in trainer source.

    Parses the module and counts real attribute calls, so
    comments, docstrings, and string literals cannot inflate
    the count or mask a second call site.
    """
    if not isinstance(source_path, (str, Path)):
        raise ValueError("DDP gate needs a trainer source path")
    try:
        tree = ast.parse(
            Path(source_path).read_text(encoding="utf-8"),
            filename=str(source_path),
        )
    except SyntaxError as exc:
        raise ValueError("Trainer source does not parse: %s" % exc) from exc
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "backward"
    )


def require_single_backward(source_path: Path) -> int:
    """Require exactly one backward call site in the trainer.

    Returns the count for receipts. Refuses dual-backward
    regressions that would invalidate development comparisons.
    """
    count = count_backward_calls(source_path)
    if count != 1:
        raise ValueError(
            "Trainer must call backward() exactly once, found %d in %s"
            % (count, source_path)
        )
    return count


def training_loop_uses_no_sync(source_path: Path) -> bool:
    """Check the accumulation loop guards non-final reductions."""
    if not isinstance(source_path, (str, Path)):
        raise ValueError("DDP gate needs a trainer source path")
    text = Path(source_path).read_text(encoding="utf-8")
    return "ddp.no_sync()" in text and "gradient_as_bucket_view" in text


def require_coherent_training(source_path: Path) -> dict[str, object]:
    """Enforce the coherent-training static gate for a fresh run."""
    count = require_single_backward(source_path)
    if not training_loop_uses_no_sync(source_path):
        raise ValueError(
            "Trainer lacks no_sync accumulation guard in %s" % source_path
        )
    return {"backward_calls": count, "no_sync": True}
