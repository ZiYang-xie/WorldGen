"""Events published by the engine and consumed by Winnie / UI / continuity."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import time


@dataclass(frozen=True)
class ProgressEvent:
    """Pipeline progress. Emitted by ``WorldGen`` on the default bus."""

    stage: str  # e.g. "depth", "splats", "inpaint_bg"
    fraction: Optional[float] = None  # 0.0..1.0 if known
    message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SelectionEvent:
    """The user (or an automated agent) has selected something.

    Fields:
      kind="object" → ``object_id`` is set, ``ray`` and ``hit_point`` may
                      be set if the selection came from a ray pick.
      kind="region" → ``aabb`` is set in scene coordinates.
      kind="none"   → selection cleared.
    """

    kind: Literal["object", "region", "none"]
    object_id: Optional[str] = None
    ray: Optional[tuple[tuple[float, float, float], tuple[float, float, float]]] = None
    hit_point: Optional[tuple[float, float, float]] = None
    aabb: Optional[tuple[tuple[float, float, float], tuple[float, float, float]]] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)
