"""Scene IR — the deterministic, serializable representation of a scene.

Design goals:
  - JSON round-trippable. (No binary blobs inline; splats live alongside
    as PLY/NPZ files referenced by content hash.)
  - Stable across versions. Every top-level object carries ``schema``.
  - Cheap to diff. Each SemanticObject has a stable id; affordances are
    a list keyed by ``kind``.
  - Cross-platform. No torch / open3d / pytorch3d imports anywhere.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


SCHEMA_VERSION = "auroch.scene/0.1"


@dataclass(frozen=True)
class Pose:
    """SE(3) pose in scene-local coordinates."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class Affordance:
    """A typed, declarative behavior attached to a SemanticObject.

    ``kind`` is one of:
      "door"        — door {hinge_axis, swing_range_rad, state}
      "window"      — window {hinge_axis, swing_range_rad, transparent}
      "container"   — container {capacity, contents: list[obj_id]}
      "light"       — light {color_rgb, intensity, on}
      "surface"     — surface {walkable, sittable, friction}
      "portal"      — portal {target_scene_id, target_pose}

    ``params`` holds the kind-specific fields. The schema is intentionally
    open — the runtime validates known kinds and ignores unknown ones.
    """

    kind: Literal["door", "window", "container", "light", "surface", "portal", "custom"]
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SplatRef:
    """Reference to a gaussian-splat blob stored beside the scene."""

    path: str  # relative to the scene bundle
    sha256: str  # content hash for cache/idempotency
    count: int  # number of gaussians


@dataclass(frozen=True)
class MeshRef:
    """Reference to a mesh file (glb / ply / usdz) stored beside the scene."""

    path: str
    sha256: str
    format: Literal["glb", "ply", "usdz"]


@dataclass(frozen=True)
class Provenance:
    """How this object was produced. Required for reproducible regeneration."""

    prompt: Optional[str] = None
    image_ref: Optional[str] = None  # path or URL
    seed: Optional[int] = None
    model_id: Optional[str] = None  # e.g. "FLUX.1-dev"
    lora_id: Optional[str] = None
    sampler: Optional[str] = None
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    created_at: Optional[float] = None


@dataclass(frozen=True)
class SemanticObject:
    """An addressable, semantic-aware piece of the scene.

    Each object can be backed by splats, a mesh, or both. Affordances
    describe behavior; provenance describes origin.
    """

    id: str
    label: str  # ADE20K label, custom tag, or "scene-background"
    pose: Pose = field(default_factory=Pose)
    splats: Optional[SplatRef] = None
    mesh: Optional[MeshRef] = None
    affordances: list[Affordance] = field(default_factory=list)
    provenance: Optional[Provenance] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class SceneSnapshot:
    """A complete, serializable scene.

    Edit log is kept separately in the project bundle so snapshots stay
    small.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Untitled Scene"
    schema: str = SCHEMA_VERSION
    up_axis: Literal["y", "z", "-y", "-z"] = "-y"
    objects: list[SemanticObject] = field(default_factory=list)
    provenance: Optional[Provenance] = None

    # ---- (de)serialization ----

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "SceneSnapshot":
        data = json.loads(text)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "SceneSnapshot":
        objects = [_obj_from_dict(o) for o in data.get("objects", [])]
        prov = data.get("provenance")
        return cls(
            id=data.get("id", uuid.uuid4().hex),
            name=data.get("name", "Untitled Scene"),
            schema=data.get("schema", SCHEMA_VERSION),
            up_axis=data.get("up_axis", "-y"),
            objects=objects,
            provenance=Provenance(**prov) if prov else None,
        )

    # ---- queries ----

    def find(self, *, id: Optional[str] = None, label: Optional[str] = None) -> list[SemanticObject]:
        out = []
        for o in self.objects:
            if id is not None and o.id != id:
                continue
            if label is not None and o.label != label:
                continue
            out.append(o)
        return out


def _obj_from_dict(d: dict) -> SemanticObject:
    pose = Pose(**d.get("pose", {})) if d.get("pose") else Pose()
    splats = SplatRef(**d["splats"]) if d.get("splats") else None
    mesh = MeshRef(**d["mesh"]) if d.get("mesh") else None
    prov = Provenance(**d["provenance"]) if d.get("provenance") else None
    affordances = [Affordance(**a) for a in d.get("affordances", [])]
    return SemanticObject(
        id=d["id"],
        label=d.get("label", ""),
        pose=pose,
        splats=splats,
        mesh=mesh,
        affordances=affordances,
        provenance=prov,
        tags=list(d.get("tags", [])),
    )
