"""Export Auroch Syna models to portable inference formats.

Supported formats
-----------------
onnx         ONNX graph — works on CPU/CUDA/CoreML via onnxruntime.
torchscript  TorchScript JIT module — no Python at inference time.
coreml       Apple CoreML .mlpackage — requires coremltools and macOS.

Supported models
----------------
da2          DA-2 equirectangular depth estimator (SphereViT backbone).
lama         LaMa inpainting model (JIT module).

Usage
-----
    # export DA-2 to ONNX (fixed 512×1024 input):
    python tools/convert_model.py --model da2 --format onnx \\
           --output models/da2_512x1024.onnx --height 512 --width 1024

    # export DA-2 to TorchScript:
    python tools/convert_model.py --model da2 --format torchscript \\
           --output models/da2.pt

    # export LaMa to CoreML (macOS only):
    python tools/convert_model.py --model lama --format coreml \\
           --output models/lama.mlpackage

Install the [convert] extra first:
    pip install '.[convert]'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# DA-2 (SphereViT depth)
# ---------------------------------------------------------------------------

def _load_da2(device: str):
    """Return the DA-2 model ready for tracing."""
    from auroch_syna.worldgen.pano_depth import build_depth_model
    import torch

    model = build_depth_model(device=torch.device(device))
    model.eval()
    return model


def _da2_dummy_input(height: int, width: int, device: str):
    import torch
    return torch.randn(1, 3, height, width, device=device)


def export_da2_onnx(output: Path, height: int, width: int, device: str) -> None:
    """Export DA-2 to ONNX.

    The model is traced at a fixed (height, width). For variable-size
    inference, export multiple variants or enable dynamic axes below.
    """
    import torch
    import torch.onnx

    model = _load_da2(device)
    dummy = _da2_dummy_input(height, width, device)

    print(f"Tracing DA-2 at {height}×{width}  …")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes={
                "image": {0: "batch"},
                "depth": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"Saved ONNX model → {output}")

    # Quick validation
    try:
        import onnx
        onnx.checker.check_model(str(output))
        print("ONNX model validation passed.")
    except ImportError:
        print("(onnx not installed — skipping validation)")


def export_da2_torchscript(output: Path, height: int, width: int, device: str) -> None:
    """Export DA-2 to TorchScript via tracing."""
    import torch

    model = _load_da2(device)
    dummy = _da2_dummy_input(height, width, device)

    print(f"Tracing DA-2 at {height}×{width} to TorchScript …")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(str(output))
    print(f"Saved TorchScript → {output}")


def export_da2_coreml(output: Path, height: int, width: int) -> None:
    """Export DA-2 to CoreML (macOS only).

    Traces via TorchScript first, then converts.
    Note: SphereViT uses custom spherical attention; if custom ops are not
    supported by the MIL backend, you may need to replace them with
    standard ops before conversion.
    """
    import sys
    if sys.platform != "darwin":
        raise RuntimeError("CoreML export requires macOS.")

    import torch
    try:
        import coremltools as ct
    except ImportError as e:
        raise RuntimeError(
            "coremltools is not installed. Run `pip install '.[convert]'`."
        ) from e

    model = _load_da2("cpu")
    dummy = _da2_dummy_input(height, width, "cpu")

    print(f"Tracing DA-2 at {height}×{width} for CoreML …")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy.shape)],
        outputs=[ct.TensorType(name="depth")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(str(output))
    print(f"Saved CoreML package → {output}")


# ---------------------------------------------------------------------------
# LaMa (inpainting)
# ---------------------------------------------------------------------------

def _load_lama_jit(device: str):
    """Return the LaMa JIT module (downloads if needed)."""
    from auroch_syna.worldgen.models.inpaint_model import LaMa
    import torch

    lama = LaMa(device=torch.device(device))
    return lama.model  # the JIT module itself


def export_lama_onnx(output: Path, device: str) -> None:
    """Export LaMa to ONNX (image + mask → inpainted image)."""
    import torch
    import torch.onnx

    jit = _load_lama_jit(device)
    H, W = 512, 512
    dummy_img  = torch.randn(1, 3, H, W, device=device)
    dummy_mask = torch.randn(1, 1, H, W, device=device)

    print("Tracing LaMa to ONNX …")
    with torch.no_grad():
        torch.onnx.export(
            jit,
            (dummy_img, dummy_mask),
            str(output),
            input_names=["image", "mask"],
            output_names=["inpainted"],
            dynamic_axes={
                "image":     {0: "batch", 2: "height", 3: "width"},
                "mask":      {0: "batch", 2: "height", 3: "width"},
                "inpainted": {0: "batch", 2: "height", 3: "width"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"Saved ONNX model → {output}")


def export_lama_torchscript(output: Path, device: str) -> None:
    """Export LaMa to TorchScript (it is already JIT-compiled)."""
    import torch

    jit = _load_lama_jit(device)
    torch.jit.save(jit, str(output))
    print(f"Saved TorchScript → {output}")


def export_lama_coreml(output: Path) -> None:
    """Export LaMa to CoreML."""
    import sys
    if sys.platform != "darwin":
        raise RuntimeError("CoreML export requires macOS.")

    import torch
    try:
        import coremltools as ct
    except ImportError as e:
        raise RuntimeError("Run `pip install '.[convert]'`.") from e

    jit = _load_lama_jit("cpu")
    H, W = 512, 512
    dummy_img  = torch.randn(1, 3, H, W)
    dummy_mask = torch.randn(1, 1, H, W)

    print("Converting LaMa to CoreML …")
    mlmodel = ct.convert(
        jit,
        inputs=[
            ct.TensorType(name="image", shape=dummy_img.shape),
            ct.TensorType(name="mask",  shape=dummy_mask.shape),
        ],
        outputs=[ct.TensorType(name="inpainted")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(str(output))
    print(f"Saved CoreML package → {output}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_EXPORTS = {
    ("da2",  "onnx"):        lambda args: export_da2_onnx(args.output, args.height, args.width, args.device),
    ("da2",  "torchscript"): lambda args: export_da2_torchscript(args.output, args.height, args.width, args.device),
    ("da2",  "coreml"):      lambda args: export_da2_coreml(args.output, args.height, args.width),
    ("lama", "onnx"):        lambda args: export_lama_onnx(args.output, args.device),
    ("lama", "torchscript"): lambda args: export_lama_torchscript(args.output, args.device),
    ("lama", "coreml"):      lambda args: export_lama_coreml(args.output),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model",  required=True, choices=["da2", "lama"])
    parser.add_argument("--format", required=True, choices=["onnx", "torchscript", "coreml"],
                        dest="fmt")
    parser.add_argument("--output", "-o", required=True, type=Path,
                        help="Output file path (.onnx / .pt / .mlpackage)")
    parser.add_argument("--height", type=int, default=512,
                        help="Input height for tracing (DA-2 only, default 512)")
    parser.add_argument("--width",  type=int, default=1024,
                        help="Input width for tracing (DA-2 only, default 1024)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for tracing (default cpu)")
    args = parser.parse_args(argv)

    fn = _EXPORTS.get((args.model, args.fmt))
    if fn is None:
        print(f"No exporter for ({args.model}, {args.fmt})", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
