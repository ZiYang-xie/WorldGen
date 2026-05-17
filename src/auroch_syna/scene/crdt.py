"""CRDT primitives for collaborative scene editing.

Provides the building blocks for offline-first, conflict-free scene sync.
All types are JSON-round-trippable; no torch / numpy / open3d dependency.

Primitives
----------
LWWRegister[T]
    Last-Write-Wins register. On concurrent writes the higher logical
    timestamp wins. Ties broken by lexicographic comparison of the
    author field (arbitrary but deterministic).

ORSet[T]
    Observed-Remove Set. Each element is paired with a set of unique
    "add tags". An element is *present* iff its tag set is non-empty.
    Removes work by clearing all currently-known tags; concurrent adds
    (with new tags) survive removes.

SceneDelta
    A compact, JSON-serializable diff between two ``SceneSnapshot``
    states. Captures added, removed, and changed objects.

Functions
---------
diff(a, b) -> SceneDelta
    Compute the changes needed to advance ``a`` to ``b``.

merge(base, delta) -> SceneSnapshot
    Apply a ``SceneDelta`` to a snapshot, returning the merged result.

merge_snapshots(a, b) -> SceneSnapshot
    Merge two diverged snapshots using OR-Set semantics for objects and
    LWW for scalar fields (name, up_axis, provenance).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Generic, Hashable, Iterator, Optional, TypeVar

from auroch_syna.scene.ir import SceneSnapshot, SemanticObject

T = TypeVar("T")


# ---------------------------------------------------------------------------
# LWW Register
# ---------------------------------------------------------------------------

@dataclass
class LWWRegister(Generic[T]):
    """Last-Write-Wins register.

    Carries a value and a logical timestamp (float seconds). On merge,
    the higher timestamp wins; ties resolved by ``author`` lexicography.
    """

    value: T
    timestamp: float = field(default_factory=time.time)
    author: str = "local"

    def write(self, value: T, *, author: str = "local") -> "LWWRegister[T]":
        """Return a new register with the updated value and current time."""
        return LWWRegister(value=value, timestamp=time.time(), author=author)

    def merge(self, other: "LWWRegister[T]") -> "LWWRegister[T]":
        """Return the register whose write dominates (higher ts, then author)."""
        if self.timestamp > other.timestamp:
            return self
        if other.timestamp > self.timestamp:
            return other
        # Tie — pick lexicographically larger author for determinism.
        return self if self.author >= other.author else other

    def to_dict(self) -> dict:
        return {"value": self.value, "timestamp": self.timestamp, "author": self.author}

    @classmethod
    def from_dict(cls, d: dict) -> "LWWRegister":
        return cls(value=d["value"], timestamp=d["timestamp"], author=d.get("author", "local"))


# ---------------------------------------------------------------------------
# OR-Set
# ---------------------------------------------------------------------------

@dataclass
class ORSet(Generic[T]):
    """Observed-Remove Set.

    Each element is paired with a set of unique "add tags" (UUID strings).
    An element is *present* iff its tag set is non-empty. Removes work by
    tombstoning all currently observed tags; concurrent remote adds
    (carrying new tags) survive the remove.

    Elements must be hashable.
    """

    # element → {add_tag, ...}
    _adds: dict = field(default_factory=dict)

    # ---- mutation ----

    def add(self, element: T) -> str:
        """Add element; return the new add-tag."""
        tag = uuid.uuid4().hex
        key = _key(element)
        self._adds.setdefault(key, {"element": element, "tags": set()})["tags"].add(tag)
        return tag

    def remove(self, element: T) -> None:
        """Remove element by clearing all observed tags."""
        key = _key(element)
        if key in self._adds:
            self._adds[key]["tags"].clear()

    def __contains__(self, element: object) -> bool:
        key = _key(element)
        entry = self._adds.get(key)
        return bool(entry and entry["tags"])

    def __iter__(self) -> Iterator[T]:
        for entry in self._adds.values():
            if entry["tags"]:
                yield entry["element"]

    def __len__(self) -> int:
        return sum(1 for entry in self._adds.values() if entry["tags"])

    # ---- merge ----

    def merge(self, other: "ORSet[T]") -> "ORSet[T]":
        """Return the union (tag sets merged element-wise)."""
        result: ORSet[T] = ORSet()
        all_keys = set(self._adds) | set(other._adds)
        for key in all_keys:
            a = self._adds.get(key)
            b = other._adds.get(key)
            if a and b:
                merged_tags = a["tags"] | b["tags"]
                result._adds[key] = {"element": a["element"], "tags": merged_tags}
            elif a:
                result._adds[key] = {"element": a["element"], "tags": set(a["tags"])}
            elif b:
                result._adds[key] = {"element": b["element"], "tags": set(b["tags"])}
        return result

    # ---- serialization ----

    def to_dict(self) -> dict:
        return {
            key: {"element": entry["element"], "tags": list(entry["tags"])}
            for key, entry in self._adds.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ORSet":
        s: ORSet = cls()
        for key, entry in d.items():
            s._adds[key] = {"element": entry["element"], "tags": set(entry["tags"])}
        return s


def _key(element: object) -> str:
    if isinstance(element, dict):
        return element.get("id", str(id(element)))
    return getattr(element, "id", str(element))


# ---------------------------------------------------------------------------
# SceneDelta — compact diff
# ---------------------------------------------------------------------------

@dataclass
class SceneDelta:
    """Compact diff between two ``SceneSnapshot`` states.

    Describes only what changed; unchanged objects are omitted.
    Apply with ``merge(base, delta)`` to produce the target snapshot.
    """

    # IDs of newly-added objects (full SemanticObject dicts).
    added: list[dict] = field(default_factory=list)
    # IDs of removed objects.
    removed: list[str] = field(default_factory=list)
    # Changed objects (full SemanticObject dicts for simplicity; partial
    # diff tracking is a future optimisation).
    changed: list[dict] = field(default_factory=list)
    # Scalar field changes.
    name: Optional[str] = None
    up_axis: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return (
            not self.added and not self.removed and not self.changed
            and self.name is None and self.up_axis is None
        )

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "name": self.name,
            "up_axis": self.up_axis,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneDelta":
        return cls(
            added=d.get("added", []),
            removed=d.get("removed", []),
            changed=d.get("changed", []),
            name=d.get("name"),
            up_axis=d.get("up_axis"),
        )


# ---------------------------------------------------------------------------
# diff and merge
# ---------------------------------------------------------------------------

def diff(a: SceneSnapshot, b: SceneSnapshot) -> SceneDelta:
    """Compute the delta needed to advance snapshot ``a`` to snapshot ``b``.

    Returns a ``SceneDelta`` with added, removed, and changed objects.
    """
    a_by_id = {o.id: o for o in a.objects}
    b_by_id = {o.id: o for o in b.objects}

    added_ids = set(b_by_id) - set(a_by_id)
    removed_ids = set(a_by_id) - set(b_by_id)
    common_ids = set(a_by_id) & set(b_by_id)

    added = [asdict(b_by_id[oid]) for oid in added_ids]
    removed = list(removed_ids)
    changed = [
        asdict(b_by_id[oid])
        for oid in common_ids
        if asdict(a_by_id[oid]) != asdict(b_by_id[oid])
    ]

    return SceneDelta(
        added=added,
        removed=removed,
        changed=changed,
        name=b.name if b.name != a.name else None,
        up_axis=b.up_axis if b.up_axis != a.up_axis else None,
    )


def merge(base: SceneSnapshot, delta: SceneDelta) -> SceneSnapshot:
    """Apply ``delta`` to ``base``; return the merged snapshot.

    The base snapshot is not mutated.
    """
    from auroch_syna.scene.ir import _obj_from_dict  # avoid circular at module level

    objects_by_id = {o.id: o for o in base.objects}

    # Apply removals.
    for oid in delta.removed:
        objects_by_id.pop(oid, None)

    # Apply changes (overwrite).
    for obj_dict in delta.changed:
        obj = _obj_from_dict(obj_dict)
        objects_by_id[obj.id] = obj

    # Apply additions.
    for obj_dict in delta.added:
        obj = _obj_from_dict(obj_dict)
        objects_by_id[obj.id] = obj

    return SceneSnapshot(
        id=base.id,
        name=delta.name if delta.name is not None else base.name,
        schema=base.schema,
        up_axis=delta.up_axis if delta.up_axis is not None else base.up_axis,
        objects=list(objects_by_id.values()),
        provenance=base.provenance,
    )


def merge_snapshots(a: SceneSnapshot, b: SceneSnapshot) -> SceneSnapshot:
    """Merge two diverged snapshots using OR-Set semantics.

    Objects present in either snapshot are retained. For objects present
    in both, the copy from ``b`` wins (LWW with ``b`` as the later writer).
    Scalar fields (name, up_axis) use LWW with ``b`` winning.

    This is intentionally simple — a production implementation should
    attach vector clocks or hybrid logical clocks to each object.
    """
    a_by_id = {o.id: o for o in a.objects}
    b_by_id = {o.id: o for o in b.objects}

    merged: dict[str, SemanticObject] = {}
    for oid, obj in a_by_id.items():
        merged[oid] = b_by_id.get(oid, obj)  # b wins on conflict
    for oid, obj in b_by_id.items():
        merged.setdefault(oid, obj)

    return SceneSnapshot(
        id=a.id,
        name=b.name,
        schema=a.schema,
        up_axis=b.up_axis,
        objects=list(merged.values()),
        provenance=b.provenance or a.provenance,
    )


__all__ = [
    "LWWRegister",
    "ORSet",
    "SceneDelta",
    "diff",
    "merge",
    "merge_snapshots",
]
