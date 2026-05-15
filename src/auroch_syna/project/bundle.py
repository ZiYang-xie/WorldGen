"""ProjectBundle — open / save .aurochsyna directories."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

from auroch_syna.runtime import get_logger
from auroch_syna.scene.commands import Command
from auroch_syna.scene.ir import SCHEMA_VERSION, SceneSnapshot

log = get_logger(__name__)


MANIFEST_VERSION = "auroch.bundle/0.1"


class ProjectBundle:
    """A read/write handle to a project on disk.

    Bundles are directories. ``ProjectBundle.create()`` makes a fresh
    one; ``ProjectBundle.open()`` loads an existing one. Edits are
    appended to ``edit_log.jsonl``; the scene snapshot is rewritten on
    ``save()``.
    """

    def __init__(self, path: Path, snapshot: SceneSnapshot) -> None:
        self.path = path
        self.snapshot = snapshot

    # ---- factories ----

    @classmethod
    def create(cls, path: Path, *, name: str = "Untitled Scene") -> "ProjectBundle":
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"bundle already exists: {path}")
        path.mkdir(parents=True)
        (path / "assets").mkdir()
        snapshot = SceneSnapshot(name=name)
        bundle = cls(path, snapshot)
        bundle.save()
        (path / "edit_log.jsonl").touch()
        return bundle

    @classmethod
    def open(cls, path: Path) -> "ProjectBundle":
        path = Path(path)
        scene_path = path / "scene.json"
        if not scene_path.exists():
            raise FileNotFoundError(f"missing scene.json in {path}")
        snapshot = SceneSnapshot.from_json(scene_path.read_text())
        return cls(path, snapshot)

    # ---- persistence ----

    def save(self) -> None:
        manifest = {
            "schema": MANIFEST_VERSION,
            "scene_schema": SCHEMA_VERSION,
            "name": self.snapshot.name,
            "id": self.snapshot.id,
        }
        (self.path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (self.path / "scene.json").write_text(self.snapshot.to_json())

    def append_op(self, op: Command) -> None:
        with (self.path / "edit_log.jsonl").open("a") as f:
            f.write(json.dumps(op.to_dict()) + "\n")

    def iter_ops(self) -> Iterator[Command]:
        log_path = self.path / "edit_log.jsonl"
        if not log_path.exists():
            return iter([])
        def _gen():
            with log_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield Command.from_dict(json.loads(line))
        return _gen()

    # ---- assets ----

    def stage_asset(self, source: Path) -> str:
        """Copy an asset into the bundle and return its content-addressed path.

        Returns ``"assets/<sha256>.<ext>"`` relative to the bundle root.
        """
        source = Path(source)
        sha = _sha256(source)
        ext = source.suffix.lstrip(".") or "bin"
        rel = f"assets/{sha}.{ext}"
        dst = self.path / rel
        if not dst.exists():
            shutil.copy2(source, dst)
        return rel

    def asset_path(self, rel: str) -> Path:
        return self.path / rel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
