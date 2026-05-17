"""Unit tests for the scene IR and ProjectBundle — zero ML deps."""
import json
import tempfile
from pathlib import Path

import pytest

from auroch_syna.scene.ir import (
    SCHEMA_VERSION,
    Affordance,
    MeshRef,
    Pose,
    Provenance,
    SceneSnapshot,
    SemanticObject,
    SplatRef,
)
from auroch_syna.scene.commands import Command
from auroch_syna.project.bundle import ProjectBundle


# ---------------------------------------------------------------------------
# SceneSnapshot serialization
# ---------------------------------------------------------------------------

def test_snapshot_default_roundtrip():
    snap = SceneSnapshot(name="Test Scene")
    text = snap.to_json()
    data = json.loads(text)
    assert data["name"] == "Test Scene"
    assert data["schema"] == SCHEMA_VERSION
    assert data["objects"] == []

    restored = SceneSnapshot.from_json(text)
    assert restored.id == snap.id
    assert restored.name == snap.name


def test_snapshot_with_objects():
    obj = SemanticObject(
        id="obj-1",
        label="chair",
        pose=Pose(translation=(1.0, 0.0, 2.0)),
        affordances=[Affordance(kind="surface", params={"walkable": False, "sittable": True})],
        provenance=Provenance(prompt="A wooden chair", seed=7),
    )
    snap = SceneSnapshot(name="Room", objects=[obj])
    restored = SceneSnapshot.from_json(snap.to_json())

    assert len(restored.objects) == 1
    r = restored.objects[0]
    assert r.id == "obj-1"
    assert r.label == "chair"
    assert list(r.pose.translation) == [1.0, 0.0, 2.0]
    assert len(r.affordances) == 1
    assert r.affordances[0].kind == "surface"
    assert r.provenance.prompt == "A wooden chair"
    assert r.provenance.seed == 7


def test_snapshot_find():
    o1 = SemanticObject(id="a", label="wall")
    o2 = SemanticObject(id="b", label="floor")
    o3 = SemanticObject(id="c", label="wall")
    snap = SceneSnapshot(objects=[o1, o2, o3])

    walls = snap.find(label="wall")
    assert len(walls) == 2
    assert {o.id for o in walls} == {"a", "c"}

    found = snap.find(id="b")
    assert len(found) == 1
    assert found[0].label == "floor"


def test_splat_ref_roundtrip():
    obj = SemanticObject(
        id="s",
        label="background",
        splats=SplatRef(path="assets/abc.ply", sha256="abc123", count=10_000),
    )
    snap = SceneSnapshot(objects=[obj])
    restored = SceneSnapshot.from_json(snap.to_json())
    assert restored.objects[0].splats.sha256 == "abc123"
    assert restored.objects[0].splats.count == 10_000


def test_mesh_ref_roundtrip():
    obj = SemanticObject(
        id="m",
        label="prop",
        mesh=MeshRef(path="assets/table.glb", sha256="def456", format="glb"),
    )
    snap = SceneSnapshot(objects=[obj])
    restored = SceneSnapshot.from_json(snap.to_json())
    assert restored.objects[0].mesh.format == "glb"


# ---------------------------------------------------------------------------
# Command serialization
# ---------------------------------------------------------------------------

def test_command_roundtrip():
    cmd = Command(kind="generate_scene", params={"prompt": "forest", "seed": 42})
    restored = Command.from_dict(cmd.to_dict())
    assert restored.kind == "generate_scene"
    assert restored.params["prompt"] == "forest"
    assert restored.op_id == cmd.op_id


def test_command_from_dict_defaults():
    cmd = Command.from_dict({"kind": "noop"})
    assert cmd.client_id == "local"
    assert cmd.op_id  # auto-generated


# ---------------------------------------------------------------------------
# ProjectBundle
# ---------------------------------------------------------------------------

def test_bundle_create_and_open():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "my_scene.aurochsyna"
        bundle = ProjectBundle.create(path, name="My Scene")

        assert (path / "scene.json").exists()
        assert (path / "manifest.json").exists()
        assert (path / "assets").is_dir()
        assert (path / "edit_log.jsonl").exists()
        assert bundle.snapshot.name == "My Scene"

        reopened = ProjectBundle.open(path)
        assert reopened.snapshot.name == "My Scene"
        assert reopened.snapshot.id == bundle.snapshot.id


def test_bundle_append_and_iter_ops():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.aurochsyna"
        bundle = ProjectBundle.create(path)

        cmd1 = Command(kind="generate_scene", params={"prompt": "a forest"})
        cmd2 = Command(kind="select", params={"object_id": "obj-1"})
        bundle.append_op(cmd1)
        bundle.append_op(cmd2)

        ops = list(bundle.iter_ops())
        assert len(ops) == 2
        assert ops[0].kind == "generate_scene"
        assert ops[1].kind == "select"


def test_bundle_stage_asset(tmp_path):
    path = tmp_path / "scene.aurochsyna"
    bundle = ProjectBundle.create(path)

    # Write a tiny fake asset
    src = tmp_path / "splat.ply"
    src.write_bytes(b"PLY fake data")

    rel = bundle.stage_asset(src)
    assert rel.startswith("assets/")
    assert rel.endswith(".ply")
    assert bundle.asset_path(rel).exists()

    # Staging the same content again should be idempotent (same rel path)
    rel2 = bundle.stage_asset(src)
    assert rel2 == rel


def test_bundle_create_conflict():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "existing"
        path.mkdir()
        with pytest.raises(FileExistsError):
            ProjectBundle.create(path)


def test_bundle_open_missing():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError):
            ProjectBundle.open(Path(tmp) / "ghost.aurochsyna")
