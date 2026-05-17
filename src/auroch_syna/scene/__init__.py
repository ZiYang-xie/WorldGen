"""auroch.scene — the public scene contract.

Imported by every consumer of Syna (Winnie, the command bar, the
viewer, the daemon, the CLI). Contains:

- ``ir``: the deterministic scene IR (snapshot + edit log)
- ``events``: progress/selection events emitted by the pipeline
- ``commands``: actions the orchestrator can invoke

This package has NO ML deps. It is pure Python + dataclasses + JSON.
"""
from __future__ import annotations

from .ir import (
    Affordance,
    Pose,
    Provenance,
    SceneSnapshot,
    SemanticObject,
    SplatRef,
    MeshRef,
)
from .events import ProgressEvent, SelectionEvent
from .commands import Command, EditOp, RegenerationRequest

__all__ = [
    "Affordance",
    "Pose",
    "Provenance",
    "SceneSnapshot",
    "SemanticObject",
    "SplatRef",
    "MeshRef",
    "ProgressEvent",
    "SelectionEvent",
    "Command",
    "EditOp",
    "RegenerationRequest",
]
