"""MLX-based FLUX inference backend (Apple Silicon).

STUB. This module sketches the contract; the actual MLX-FLUX
integration is intentionally not implemented here because mlx-flux
ports diverge in their pipeline class structure and we want a single
seam to plug whichever one we adopt.

Usage:

    from auroch_syna.runtime.transport import ModelHandle
    handle: ModelHandle = build_mlx_pano_gen_handle(lora_path=...)
    image = handle.infer(prompt="a beach at sunset", height=800, width=1600)

When the real implementation lands, ``RuntimePolicy.backend == "mlx"``
will cause ``ModelClient`` to route ``handle_pano_gen`` here instead of
the torch pipeline.

TODO (priority order):
  1. Vendor or pin a working mlx-flux fork (mlx-examples or community).
  2. Map FLUX-LoRA safetensors → MLX tensor format (one-time conversion
     under tools/convert_flux_lora_to_mlx.py).
  3. Implement ``MlxPanoGenHandle.infer(...)`` with the same kwargs as
     ``gen_pano_image`` so the swap is transparent.
  4. Wire blend_extend / vae_tiling equivalents if needed.
"""
from __future__ import annotations

from typing import Any


class MlxBackendUnavailable(RuntimeError):
    """Raised when MLX or mlx-flux is not installed."""


def is_available() -> bool:
    try:
        import mlx  # noqa: F401
        return True
    except Exception:
        return False


def build_mlx_pano_gen_handle(lora_path: str | None = None, **_: Any):
    """Build an MLX-backed pano-gen ``ModelHandle``.

    Currently raises ``MlxBackendUnavailable`` — call sites should fall
    back to the torch backend (RuntimePolicy.backend == "torch") when
    this is the case.
    """
    if not is_available():
        raise MlxBackendUnavailable(
            "MLX is not installed. Run `pip install '.[ml-mlx]'` and ensure "
            "mlx-flux (or a compatible port) is on PYTHONPATH."
        )
    raise NotImplementedError(
        "MLX FLUX backend is a stub. See module docstring for the integration plan."
    )
