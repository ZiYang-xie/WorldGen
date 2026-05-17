"""Torch compatibility helpers.

Anything that needs torch and would otherwise litter the codebase with
device branching goes here. Importing this module imports torch.
"""
from __future__ import annotations

import contextlib
import torch


_AUTOCAST_SAFE_DEVICES = {"cuda", "cpu", "mps", "xpu"}


@contextlib.contextmanager
def safe_autocast(device_type: str, dtype: torch.dtype | None = None):
    """torch.autocast that no-ops on unsupported devices.

    The old code path ``torch.autocast(model.device.type)`` raises on
    older torch versions when the device isn't supported. Wrap it.
    """
    if device_type not in _AUTOCAST_SAFE_DEVICES:
        yield
        return
    if dtype is None:
        with torch.autocast(device_type):
            yield
    else:
        with torch.autocast(device_type, dtype=dtype):
            yield


def resolve_torch_device(name: str) -> torch.device:
    return torch.device(name)
