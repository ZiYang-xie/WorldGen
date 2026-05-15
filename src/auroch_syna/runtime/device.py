"""Device resolution — the single source of truth for "what should I run on".

Importing this module does NOT import torch. Torch is imported lazily
inside the functions that need it so the rest of the runtime package
stays light.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Optional, Literal

DeviceName = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceInfo:
    name: DeviceName
    total_memory_gb: Optional[float]  # None if not measurable
    supports_bf16: bool
    supports_fp16: bool
    is_apple_silicon: bool


def resolve_device(prefer: Optional[str] = None) -> DeviceInfo:
    """Pick the best device available.

    Order: explicit ``prefer`` → ``AUROCH_SYNA_DEVICE`` env → CUDA → MPS → CPU.

    Returns a ``DeviceInfo`` describing capabilities. Callers should use
    ``info.name`` as the torch device string and consult the dtype/policy
    helpers for the rest.
    """
    if prefer is None:
        prefer = os.environ.get("AUROCH_SYNA_DEVICE")

    if prefer in ("cuda", "mps", "cpu"):
        return _describe(prefer)
    if prefer == "auto" or prefer is None:
        for cand in _detect_candidates():
            return _describe(cand)
    # Unknown string — fall back to auto
    for cand in _detect_candidates():
        return _describe(cand)
    return _describe("cpu")


def _detect_candidates() -> list[DeviceName]:
    out: list[DeviceName] = []
    try:
        import torch

        if torch.cuda.is_available():
            out.append("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            out.append("mps")
    except Exception:
        pass
    out.append("cpu")
    return out


def _describe(name: str) -> DeviceInfo:
    name = name if name in ("cuda", "mps", "cpu") else "cpu"
    is_apple = (
        platform.system() == "Darwin"
        and platform.processor() in ("arm", "")
        and platform.machine() == "arm64"
    )

    total_gb: Optional[float] = None
    supports_bf16 = False
    supports_fp16 = False

    try:
        import torch

        if name == "cuda" and torch.cuda.is_available():
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                supports_bf16 = torch.cuda.is_bf16_supported()
                supports_fp16 = True
            except Exception:
                pass
        elif name == "mps":
            # MPS does not expose memory size. fp16 is well-supported;
            # bf16 support is partial — prefer fp16 unless caller opts in.
            supports_fp16 = True
            supports_bf16 = _mps_bf16_likely()
            total_gb = _apple_unified_memory_gb()
        else:
            # CPU — bf16 only safe on recent Intel/AMD; fp32 default.
            supports_bf16 = False
            supports_fp16 = False
            total_gb = _system_ram_gb()
    except Exception:
        pass

    return DeviceInfo(
        name=name,  # type: ignore[arg-type]
        total_memory_gb=total_gb,
        supports_bf16=supports_bf16,
        supports_fp16=supports_fp16,
        is_apple_silicon=is_apple,
    )


def _mps_bf16_likely() -> bool:
    # macOS 14+ on Apple Silicon has functional (but incomplete) bf16 in MPS.
    # Conservative default: opt-in via env so we don't tank perf on older
    # systems that silently fall back to fp32.
    return os.environ.get("AUROCH_SYNA_MPS_BF16", "0") == "1"


def _apple_unified_memory_gb() -> Optional[float]:
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=1.0)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        return None


def _system_ram_gb() -> Optional[float]:
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=1.0)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except Exception:
        return None
