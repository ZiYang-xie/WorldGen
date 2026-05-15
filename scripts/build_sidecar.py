#!/usr/bin/env python3
"""
Build the Python daemon sidecar for the Tauri desktop app.

Usage:
    python scripts/build_sidecar.py [--target <triple>]

Outputs a single binary to apps/desktop/src-tauri/binaries/syna-daemon-<triple>.
Tauri strips the triple suffix at runtime via the externalBin mechanism.
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARIES_DIR = ROOT / "apps" / "desktop" / "src-tauri" / "binaries"
ENTRY = ROOT / "src" / "auroch_syna" / "daemon" / "__main__.py"


def host_triple() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        arch = "aarch64"
    elif machine in ("x86_64", "amd64"):
        arch = "x86_64"
    else:
        arch = machine

    system = platform.system().lower()
    if system == "darwin":
        return f"{arch}-apple-darwin"
    elif system == "linux":
        return f"{arch}-unknown-linux-gnu"
    elif system == "windows":
        return f"{arch}-pc-windows-msvc"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def build(triple: str) -> None:
    BINARIES_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"syna-daemon-{triple}"
    out_path = BINARIES_DIR / out_name

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", out_name,
        "--distpath", str(BINARIES_DIR),
        "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build" / "pyinstaller"),
        "--noconfirm",
        # Hidden imports for lazy-loaded ML modules
        "--hidden-import", "auroch_syna.worldgen",
        "--hidden-import", "auroch_syna.worldgen.worldgen",
        "--hidden-import", "auroch_syna.daemon.handlers",
        "--collect-data", "auroch_syna",
        str(ENTRY),
    ]

    print(f"Building sidecar for {triple}…")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Tauri requires the binary to be executable
    out_path.chmod(0o755)
    print(f"\nSidecar written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Auroch Syna daemon sidecar")
    parser.add_argument(
        "--target",
        default=None,
        help="Rust target triple (default: auto-detect host)",
    )
    args = parser.parse_args()

    triple = args.target or host_triple()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found. Install it with:  pip install pyinstaller")
        sys.exit(1)

    build(triple)


if __name__ == "__main__":
    main()
