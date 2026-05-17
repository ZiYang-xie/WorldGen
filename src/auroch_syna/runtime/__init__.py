"""auroch_syna.runtime — deterministic core.

This package MUST be importable in environments without torch / open3d /
pytorch3d / diffusers. Anything that needs those belongs in
``auroch_syna.worldgen`` (the ML pipeline) or in optional backends.

Public surface kept intentionally small.
"""
from __future__ import annotations

from .logging import get_logger, configure
from .bus import EventBus, default_bus, Event
from .device import resolve_device, DeviceInfo
from .policy import RuntimePolicy, select_policy

__all__ = [
    "get_logger",
    "configure",
    "EventBus",
    "default_bus",
    "Event",
    "resolve_device",
    "DeviceInfo",
    "RuntimePolicy",
    "select_policy",
]
