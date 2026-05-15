"""Auroch Syna — cross-platform world-building environment.

Top-level package. Heavy ML dependencies (torch, diffusers, open3d, pytorch3d)
are only imported lazily on attribute access so that
`import auroch_syna` is cheap and works even in environments where the
optional `[ml]` extra is not installed.
"""
from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["WorldGen", "SplatFile", "__version__"]


def __getattr__(name: str):
    if name == "WorldGen":
        from auroch_syna.worldgen.worldgen import WorldGen
        return WorldGen
    if name == "SplatFile":
        from auroch_syna.worldgen.utils.splat_utils import SplatFile
        return SplatFile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
