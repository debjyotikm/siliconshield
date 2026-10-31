"""Optional CPU staging for memory-mapped weights before CUDA transfer."""

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def staged_weights(enabled: bool) -> Iterator[None]:
    """Stage one tensor at a time without changing its values or dtype.

    The pinned Transformers loader exposes this internal hook. Restore it
    even after a failed load, so later operations retain their normal path.
    """
    if not enabled:
        yield
        return
    import torch
    from transformers import core_model_loading

    original = core_model_loading._materialize_copy

    def materialize(tensor, device=None, dtype=None):
        source = tensor[...]
        target = torch.device(device) if device is not None else None
        if source.device.type == "cpu" and target is not None:
            if target.type == "cuda":
                source = source.clone()
        if device is not None or dtype is not None:
            return source.to(device=device, dtype=dtype)
        return source

    core_model_loading._materialize_copy = materialize
    try:
        yield
    finally:
        core_model_loading._materialize_copy = original
