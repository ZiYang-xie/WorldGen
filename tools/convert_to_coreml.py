"""Convert DA-2 and LaMa checkpoints to CoreML.

This is a SKELETON. Each conversion has its own tracing pitfalls; this
script is here so the path is named and reachable, and so the audit
items in REPORT.md / AUDIT_FULL.md have a landing place.

Usage:

    python tools/convert_to_coreml.py --model da2 --output models/da2.mlpackage
    python tools/convert_to_coreml.py --model lama --output models/lama.mlpackage

Requirements (install separately, not in core deps):

    pip install coremltools torch torchvision

Status:

    [STUB] da2:  Custom SphereViT layers — needs custom op shims.
    [STUB] lama: TorchScript JIT module — convert via coremltools.convert
                 with the JIT model directly. Should be straightforward.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert_lama(output: Path) -> None:
    """Convert LaMa to CoreML.

    LaMa ships as a TorchScript module already (`iopaint.helper.load_jit_model`),
    so the path is:

      1. Download the .pt via the LAMA_MODEL_URL env or default.
      2. Load with torch.jit.load.
      3. Trace with a representative (image, mask) tuple.
      4. coremltools.convert(...) → output mlpackage.

    Wired as a stub for now.
    """
    raise NotImplementedError(
        "LaMa → CoreML conversion is unimplemented. See module docstring "
        "for the four-step recipe."
    )


def convert_da2(output: Path) -> None:
    """Convert DA-2 (SphereViT) to CoreML.

    Harder than LaMa: DA-2 uses custom spherical attention layers that
    may not have direct MIL equivalents. Likely requires:

      - Replacing custom layers with their unrolled equivalents.
      - Tracing at a fixed (H, W) (no dynamic shape).
      - Possibly two variants (full-res + tiled) for different memory budgets.

    Wired as a stub.
    """
    raise NotImplementedError(
        "DA-2 → CoreML conversion is unimplemented. SphereViT custom "
        "attention needs unrolling first."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["da2", "lama"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.model == "lama":
        convert_lama(args.output)
    else:
        convert_da2(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
