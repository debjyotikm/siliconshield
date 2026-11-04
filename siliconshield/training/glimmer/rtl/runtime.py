"""Explicit, scoped CPU staging for multi-device Glimmer execution."""

from contextlib import contextmanager


@contextmanager
def staged_transfers(enabled):
    """Restore both Accelerate hooks on exit, including exceptional exit."""
    if not enabled:
        yield
        return
    import accelerate.hooks as hooks
    import accelerate.utils.operations as operations
    import torch

    original = operations.send_to_device
    original_hook = hooks.send_to_device

    def send(tensor, device, non_blocking=False, skip_keys=None):
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda":
            target = (
                torch.device("cuda", device)
                if isinstance(device, int)
                else torch.device(device)
            )
            if target.type == "cuda":
                index = (
                    torch.cuda.current_device()
                    if target.index is None
                    else target.index
                )
                if index != tensor.device.index:
                    return tensor.to("cpu").to(target)
        return original(
            tensor, device, non_blocking=non_blocking, skip_keys=skip_keys
        )

    operations.send_to_device = send
    hooks.send_to_device = send
    try:
        yield
    finally:
        operations.send_to_device = original
        hooks.send_to_device = original_hook


def clip_grad_norm(parameters, maximum):
    """Aggregate gradient norms on CPU rather than cross-device CUDA copies."""
    import torch

    params = list(parameters)
    if not params or any(p.grad is None for p in params):
        raise RuntimeError("Missing trainable gradients")
    norm = torch.stack(
        [p.grad.detach().float().norm().cpu() for p in params]
    ).norm()
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("Nonfinite gradient norm")
    coefficient = min(1.0, maximum / (float(norm) + 1e-6))
    for param in params:
        param.grad.mul_(coefficient)
    return float(norm)
