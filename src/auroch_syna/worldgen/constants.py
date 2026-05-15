"""Package-wide named constants.

Centralising magic numbers here makes them easy to find, adjust, and
reference from tests without re-deriving their meaning from context.
"""

# ── Depth estimation ────────────────────────────────────────────────────────

# DA-2 outputs scale-invariant depth; we rescale so the maximum distance
# across the panorama equals this value (in metres).
MAX_DEPTH_DISTANCE: float = 20.0

# ── Gaussian splats ──────────────────────────────────────────────────────────

# Zeroth-order spherical-harmonics coefficient 1 / (2*sqrt(pi)).
# Used to convert linear RGB [0,1] → SH DC coefficient and back.
SH_C0: float = 0.28209479177387814

# ── Sharp (ML-Sharp cubemap pipeline) ───────────────────────────────────────

# Internal resolution that the Sharp model expects its input upsampled to.
SHARP_INTERNAL_SHAPE: tuple[int, int] = (1536, 1536)

# Default cubemap face size used when decomposing an equirectangular image.
DEFAULT_CUBEMAP_FACE_SIZE: int = 768
