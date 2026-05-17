"""Project bundle format (.aurochsyna).

A project bundle is a directory (or zip) containing:

    project.aurochsyna/
      manifest.json            — bundle metadata, version
      scene.json               — SceneSnapshot (auroch.scene.ir)
      edit_log.jsonl           — one Command per line, in causal order
      provenance.json          — top-level provenance (model versions etc)
      assets/                  — binary blobs (splats, meshes, thumbnails)
        splat-<sha256>.ply
        mesh-<sha256>.glb
        thumbnail.png

The bundle is content-addressed: assets live under their sha256. The
scene.json references them by relative path, so opening a bundle on
another device only needs to fetch the assets that aren't already
cached.
"""
from __future__ import annotations

from .bundle import ProjectBundle

__all__ = ["ProjectBundle"]
