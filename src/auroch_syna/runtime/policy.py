"""Runtime policy — given device + memory budget, decide pipeline modes.

This replaces the ad-hoc ``low_vram: bool`` branching in ``worldgen.py``
and the CUDA-only assumptions in ``pano_gen.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .device import DeviceInfo, resolve_device


DType = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class RuntimePolicy:
    device: str
    dtype: DType
    low_vram: bool
    enable_cpu_offload: bool
    enable_vae_tiling: bool
    backend: Literal["torch", "nunchaku", "mlx"]
    notes: tuple[str, ...]

    def torch_dtype(self):
        """Return the torch.dtype matching ``self.dtype`` (imports torch)."""
        import torch

        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.dtype]


# Apple Silicon configurations where mlx-flux is the right backend.
_MLX_ELIGIBLE_GB = 16.0
# CUDA threshold below which Nunchaku low-vram is required.
_CUDA_LOW_VRAM_GB = 24.0


def select_policy(
    *,
    prefer_device: str | None = None,
    explicit_low_vram: bool | None = None,
) -> RuntimePolicy:
    """Pick a runtime policy for the host.

    Resolution order:
      1. Use the resolved device.
      2. Pick a backend per device (Nunchaku on CUDA-low-vram, MLX on
         Apple Silicon if available, plain torch otherwise).
      3. Pick a dtype favoring on-device support.
    """
    info: DeviceInfo = resolve_device(prefer_device)
    notes: list[str] = []

    if info.name == "cuda":
        # Auto-detect low_vram when not specified
        low_vram = explicit_low_vram if explicit_low_vram is not None else (
            (info.total_memory_gb or 0) < _CUDA_LOW_VRAM_GB
        )
        backend = "nunchaku" if (low_vram and _nunchaku_available()) else "torch"
        if low_vram and backend != "nunchaku":
            notes.append("low_vram requested but nunchaku unavailable; "
                        "falling back to torch with cpu offload")
        dtype: DType = "bf16" if info.supports_bf16 else "fp16"
        return RuntimePolicy(
            device="cuda",
            dtype=dtype,
            low_vram=low_vram,
            enable_cpu_offload=True,
            enable_vae_tiling=True,
            backend=backend,
            notes=tuple(notes),
        )

    if info.name == "mps":
        # On Apple Silicon prefer fp16 unless the user opts into bf16.
        dtype = "bf16" if info.supports_bf16 else "fp16"
        backend = "mlx" if _mlx_available() and (info.total_memory_gb or 0) >= _MLX_ELIGIBLE_GB else "torch"
        if backend == "torch":
            notes.append("MLX backend not selected; using torch+MPS. "
                        "Install mlx-flux for faster on-device inference.")
        return RuntimePolicy(
            device="mps",
            dtype=dtype,
            low_vram=True,  # On unified memory, always treat as low-vram
            enable_cpu_offload=False,  # MPS+offload is buggy in torch today
            enable_vae_tiling=True,
            backend=backend,
            notes=tuple(notes),
        )

    # CPU
    notes.append("Running on CPU — FLUX inference is impractical here. "
                "Consider an out-of-process model server.")
    return RuntimePolicy(
        device="cpu",
        dtype="fp32",
        low_vram=True,
        enable_cpu_offload=False,
        enable_vae_tiling=True,
        backend="torch",
        notes=tuple(notes),
    )


def _nunchaku_available() -> bool:
    try:
        import nunchaku  # noqa: F401
        return True
    except Exception:
        return False


def _mlx_available() -> bool:
    try:
        import mlx  # noqa: F401
        return True
    except Exception:
        return False
