"""Auroch Syna compatibility package.

This package acts as a lightweight bridge while the full package rename is
performed. It exposes `__version__` and re-exports the main public symbols.

It also inserts the original `src/worldgen` directory onto the package
`__path__` so `import auroch_syna.<module>` will resolve to the existing
`src/worldgen` sources until we move files.
"""
import os
from importlib import import_module

__version__ = "0.1.0"

# Keep the old `src/worldgen` path available so submodule imports continue
# to work during the migration window. Do not import heavy submodules here.
_here = os.path.abspath(os.path.dirname(__file__))
_worldgen_path = os.path.abspath(os.path.join(_here, "..", "worldgen"))
if os.path.isdir(_worldgen_path) and _worldgen_path not in __path__:
	__path__.insert(0, _worldgen_path)


def __getattr__(name: str):
	"""Lazily import heavy attributes on access to avoid expensive imports at
	module import time (useful for CI smoke tests and lightweight tooling).
	"""
	if name == "WorldGen":
		mod = import_module("auroch_syna.worldgen")
		return getattr(mod, "WorldGen")
	if name == "SplatFile":
		# splat utils might live under worldgen.utils during migration
		try:
			mod = import_module("auroch_syna.utils.splat_utils")
		except Exception:
			mod = import_module("worldgen.utils.splat_utils")
		return getattr(mod, "SplatFile")
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
	return ["WorldGen", "SplatFile", "__version__"]

__all__ = ["WorldGen", "SplatFile", "__version__"]
