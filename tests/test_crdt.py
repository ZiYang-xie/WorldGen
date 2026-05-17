"""Unit tests for the CRDT primitives — zero ML deps."""
import time

import pytest

from auroch_syna.scene.crdt import (
    LWWRegister,
    ORSet,
    SceneDelta,
    diff,
    merge,
    merge_snapshots,
)
from auroch_syna.scene.ir import Pose, SceneSnapshot, SemanticObject


# ---------------------------------------------------------------------------
# LWWRegister
# ---------------------------------------------------------------------------

def test_lww_write_wins():
    r = LWWRegister(value="a", timestamp=1.0)
    r2 = r.write("b")
    assert r2.value == "b"
    assert r2.timestamp > r.timestamp


def test_lww_merge_higher_ts_wins():
    old = LWWRegister(value="old", timestamp=1.0, author="alice")
    new = LWWRegister(value="new", timestamp=2.0, author="bob")
    assert old.merge(new).value == "new"
    assert new.merge(old).value == "new"


def test_lww_merge_tie_author_wins():
    a = LWWRegister(value="a-val", timestamp=5.0, author="alice")
    b = LWWRegister(value="b-val", timestamp=5.0, author="bob")
    # "bob" > "alice" lexicographically → bob wins
    assert a.merge(b).value == "b-val"
    assert b.merge(a).value == "b-val"


def test_lww_roundtrip():
    r = LWWRegister(value=42, timestamp=3.14, author="x")
    r2 = LWWRegister.from_dict(r.to_dict())
    assert r2.value == 42
    assert r2.author == "x"


# ---------------------------------------------------------------------------
# ORSet
# ---------------------------------------------------------------------------

def test_orset_add_contains():
    s: ORSet[str] = ORSet()
    s.add("apple")
    assert "apple" in s
    assert "banana" not in s


def test_orset_remove():
    s: ORSet[str] = ORSet()
    s.add("apple")
    s.remove("apple")
    assert "apple" not in s


def test_orset_concurrent_add_survives_remove():
    """Add from replica B should survive a concurrent remove on replica A."""
    a: ORSet[str] = ORSet()
    a.add("apple")

    b: ORSet[str] = ORSet()
    b._adds = {k: {"element": v["element"], "tags": set(v["tags"])} for k, v in a._adds.items()}

    # A removes apple; B adds apple with a new tag (concurrent)
    a.remove("apple")
    b.add("apple")

    merged = a.merge(b)
    assert "apple" in merged


def test_orset_merge_union():
    a: ORSet[str] = ORSet()
    a.add("x")
    b: ORSet[str] = ORSet()
    b.add("y")

    m = a.merge(b)
    assert "x" in m
    assert "y" in m
    assert len(m) == 2


def test_orset_len_iter():
    s: ORSet[str] = ORSet()
    s.add("a")
    s.add("b")
    s.add("c")
    s.remove("b")
    assert len(s) == 2
    items = set(s)
    assert items == {"a", "c"}


# ---------------------------------------------------------------------------
# SceneDelta + diff + merge
# ---------------------------------------------------------------------------

def _snap(*labels) -> SceneSnapshot:
    objects = [SemanticObject(id=f"id-{l}", label=l) for l in labels]
    return SceneSnapshot(objects=objects, name="test")


def test_diff_added():
    a = _snap("chair")
    b = _snap("chair", "table")
    d = diff(a, b)
    assert len(d.added) == 1
    assert d.added[0]["label"] == "table"
    assert d.removed == []
    assert d.changed == []


def test_diff_removed():
    a = _snap("chair", "table")
    b = _snap("chair")
    d = diff(a, b)
    assert d.added == []
    assert len(d.removed) == 1
    assert d.removed[0] == "id-table"


def test_diff_changed():
    a = SceneSnapshot(objects=[SemanticObject(id="x", label="old")])
    b = SceneSnapshot(objects=[SemanticObject(id="x", label="new")])
    d = diff(a, b)
    assert d.added == []
    assert d.removed == []
    assert len(d.changed) == 1
    assert d.changed[0]["label"] == "new"


def test_diff_name_change():
    a = SceneSnapshot(name="Before")
    b = SceneSnapshot(name="After")
    d = diff(a, b)
    assert d.name == "After"
    assert d.is_empty is False


def test_diff_no_change():
    a = _snap("chair")
    d = diff(a, a)
    assert d.is_empty


def test_merge_apply_add():
    base = _snap("chair")
    delta = SceneDelta(added=[{"id": "id-table", "label": "table",
                               "pose": {}, "affordances": [], "tags": []}])
    result = merge(base, delta)
    ids = {o.id for o in result.objects}
    assert "id-chair" in ids
    assert "id-table" in ids


def test_merge_apply_remove():
    base = _snap("chair", "table")
    delta = SceneDelta(removed=["id-table"])
    result = merge(base, delta)
    assert all(o.id != "id-table" for o in result.objects)


def test_merge_apply_name():
    base = SceneSnapshot(name="Old")
    delta = SceneDelta(name="New")
    result = merge(base, delta)
    assert result.name == "New"


def test_roundtrip_delta():
    a = _snap("chair")
    b = _snap("chair", "lamp")
    d = diff(a, b)
    d2 = SceneDelta.from_dict(d.to_dict())
    assert len(d2.added) == len(d.added)


def test_merge_snapshots():
    a = SceneSnapshot(objects=[SemanticObject(id="shared", label="old"),
                                SemanticObject(id="only-a", label="a")])
    b = SceneSnapshot(objects=[SemanticObject(id="shared", label="new"),
                                SemanticObject(id="only-b", label="b")])
    m = merge_snapshots(a, b)
    ids = {o.id for o in m.objects}
    assert ids == {"shared", "only-a", "only-b"}
    shared = next(o for o in m.objects if o.id == "shared")
    assert shared.label == "new"  # b wins
