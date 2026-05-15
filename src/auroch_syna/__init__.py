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

# Prepend the absolute path to the original `worldgen` package so submodule
# imports like `auroch_syna.pano_gen` resolve to `src/worldgen/*.py`.
_here = os.path.abspath(os.path.dirname(__file__))
_worldgen_path = os.path.abspath(os.path.join(_here, "..", "worldgen"))
if os.path.isdir(_worldgen_path) and _worldgen_path not in __path__:
	__path__.insert(0, _worldgen_path)

# Re-export the main public API for convenience
try:
	WorldGen = import_module("worldgen.worldgen").WorldGen
except Exception:
	# Fallback: try to import via the aliased path
	WorldGen = import_module("auroch_syna.worldgen").WorldGen

try:
	SplatFile = import_module("worldgen.utils.splat_utils").SplatFile
except Exception:
	SplatFile = import_module("auroch_syna.utils.splat_utils").SplatFile

__all__ = ["WorldGen", "SplatFile", "__version__"]
