"""Commands the engine accepts. Each command is also an edit-log op.

Treating commands and ops as the same type means undo/redo, command-bar
invocation, and cross-device continuity all share one mechanism.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import time


CommandKind = Literal[
    "generate_scene",      # text/image → new scene
    "regenerate_region",   # regenerate a masked region
    "set_affordance",      # set affordance state (open door, turn light on)
    "select",              # change selection
    "move_object",         # transform an object
    "delete_object",       # remove an object
    "tag_object",          # add/remove tags
    "set_pose",            # set object pose
    "import_asset",        # import GLB/USDZ/etc
    "noop",
]


@dataclass(frozen=True)
class Command:
    """A serializable command that the daemon can dispatch.

    ``kind`` selects the handler. ``params`` is kind-specific. The
    ``client_id`` and ``op_id`` together make the command idempotent and
    let the CRDT edit log dedupe replays.
    """

    kind: CommandKind
    params: dict = field(default_factory=dict)
    op_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    client_id: str = "local"
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Command":
        return cls(
            kind=d["kind"],
            params=dict(d.get("params", {})),
            op_id=d.get("op_id", uuid.uuid4().hex),
            client_id=d.get("client_id", "local"),
            timestamp=d.get("timestamp", time.time()),
        )


@dataclass(frozen=True)
class RegenerationRequest:
    """A specific 'regenerate this region' command, sugared.

    Equivalent to ``Command(kind="regenerate_region", params=asdict(self))``.
    """

    object_id: str
    prompt: Optional[str] = None
    mask: Optional[str] = None  # path to a mask PNG in the bundle
    seed: Optional[int] = None

    def as_command(self) -> Command:
        return Command(kind="regenerate_region", params=asdict(self))


# ---- alias for the edit log ----

EditOp = Command
"""An edit-log entry. Commands ARE ops; the log is a stream of them."""
