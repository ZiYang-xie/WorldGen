"""auroch-syna — command-line interface.

Subcommands
-----------
  auroch-syna info              Print system/device info and exit.
  auroch-syna daemon            Start the WebSocket command daemon.
  auroch-syna generate          Generate a scene from a text prompt.
  auroch-syna generate-from     Generate a scene conditioned on an image.

All ML-heavy subcommands (generate*) import torch lazily so that the
CLI itself starts fast on every platform.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="auroch-syna",
        description="Auroch Syna — cross-platform world-building CLI.",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    subs = p.add_subparsers(dest="cmd", metavar="COMMAND")

    # ---- info ----
    subs.add_parser("info", help="Show system and device information.")

    # ---- daemon ----
    d = subs.add_parser("daemon", help="Run the WebSocket command daemon.")
    d.add_argument("--host", default="127.0.0.1", metavar="HOST")
    d.add_argument("--port", default=8765, type=int, metavar="PORT")

    # ---- generate ----
    g = subs.add_parser("generate", help="Generate a 3D scene from a text prompt.")
    g.add_argument("prompt", nargs="?", default="", help="Text prompt.")
    g.add_argument("--out", "-o", metavar="PATH", help="Output file (.ply for splats, .glb for mesh).")
    g.add_argument("--mesh", action="store_true", help="Produce a triangle mesh instead of Gaussian splats.")
    g.add_argument("--resolution", type=int, default=1600, metavar="W", help="Panorama width (height = W/2).")
    g.add_argument("--device", default=None, metavar="DEVICE", help="cuda / mps / cpu (auto-detected if omitted).")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--sharp", action="store_true", help="Use ML-Sharp Gaussian predictor (requires [ml] install).")
    g.add_argument("--inpaint-bg", action="store_true", dest="inpaint_bg", help="Segment + inpaint the background.")

    # ---- generate-from ----
    gf = subs.add_parser("generate-from", help="Generate a scene conditioned on an input image.")
    gf.add_argument("image", help="Path to an input image.")
    gf.add_argument("--prompt", default="", help="Optional conditioning text.")
    gf.add_argument("--out", "-o", metavar="PATH")
    gf.add_argument("--mesh", action="store_true")
    gf.add_argument("--resolution", type=int, default=1600, metavar="W")
    gf.add_argument("--device", default=None, metavar="DEVICE")
    gf.add_argument("--seed", type=int, default=42)
    gf.add_argument("--steps", type=int, default=50)
    gf.add_argument("--sharp", action="store_true")
    gf.add_argument("--inpaint-bg", action="store_true", dest="inpaint_bg")

    return p


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def _cmd_info(_args) -> int:
    from auroch_syna.runtime import resolve_device, DeviceInfo
    import auroch_syna

    print(f"auroch-syna {auroch_syna.__version__}")

    info: DeviceInfo = resolve_device()
    print(f"device   : {info.device}")
    print(f"backend  : {info.backend}")

    try:
        import torch
        print(f"torch    : {torch.__version__}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            print(f"cuda     : {props.name}  {vram:.1f} GB VRAM")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("mps      : available")
    except ImportError:
        print("torch    : not installed (install .[ml] for ML features)")

    return 0


def _cmd_daemon(args) -> int:
    from auroch_syna.daemon.server import run
    print(f"Starting daemon on ws://{args.host}:{args.port}  (Ctrl-C to stop)")
    run(host=args.host, port=args.port)
    return 0


def _cmd_generate(args, *, mode: str = "t2s") -> int:
    from auroch_syna import WorldGen

    wg = WorldGen(
        mode=mode,
        use_sharp=args.sharp,
        inpaint_bg=args.inpaint_bg,
        resolution=args.resolution,
        device=args.device,
    )

    if mode == "t2s":
        image = None
        prompt = args.prompt
    else:
        from PIL import Image as _Image
        image = _Image.open(args.image).convert("RGB")
        prompt = getattr(args, "prompt", "")

    scene = wg.generate_world(prompt=prompt, image=image, return_mesh=args.mesh)
    out = Path(args.out) if args.out else Path("scene.ply" if not args.mesh else "scene.glb")

    _save_scene(scene, out, mesh=args.mesh)
    print(f"Saved to {out}")
    return 0


def _save_scene(scene, out: Path, *, mesh: bool) -> None:
    if mesh:
        import open3d as o3d
        o3d.io.write_triangle_mesh(str(out), scene)
    else:
        scene.save(str(out))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version or args.cmd is None:
        if args.version:
            import auroch_syna
            print(auroch_syna.__version__)
        else:
            parser.print_help()
        sys.exit(0)

    dispatch = {
        "info": _cmd_info,
        "daemon": _cmd_daemon,
        "generate": _cmd_generate,
        "generate-from": lambda a: _cmd_generate(a, mode="i2s"),
    }

    fn = dispatch.get(args.cmd)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    try:
        code = fn(args)
        sys.exit(code or 0)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
