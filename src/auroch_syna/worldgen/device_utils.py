"""Device and dtype resolution utilities.

Single source of truth for picking the right torch.device and torch.dtype
so that every module behaves consistently across CUDA / MPS / CPU.
"""
from __future__ import annotations

import torch


def resolve_device(preferred: str | torch.device | None = None) -> torch.device:
    """Return the best available device, respecting an explicit preference.

    Priority when preferred is None:
        CUDA → MPS (Apple Silicon) → CPU
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """Return the recommended floating-point dtype for the given device.

    - CUDA:  bfloat16 (native, fast on Ampere+)
    - MPS:   float16  (bfloat16 has limited MPS support)
    - CPU:   float32  (no reduced-precision benefit on CPU)
    """
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32
