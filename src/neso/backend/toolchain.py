"""Offline Metal toolchain discovery and compilation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


class MetalToolchainUnavailable(RuntimeError):
    """Raised when the offline Metal compiler is not installed."""


def _find_tool(name: str) -> str | None:
    try:
        result = subprocess.run(["xcrun", "--find", name], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    roots = (
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin",
        "/Library/Developer/CommandLineTools/usr/bin",
    )
    return next((str(Path(root, name)) for root in roots if Path(root, name).is_file()), None)


def compile_msl(msl_source: str) -> bytes:
    """Compile MSL source to a metallib, or report that the toolchain is absent."""
    metal = _find_tool("metal")
    metallib = _find_tool("metallib")
    if not metal or not metallib:
        raise MetalToolchainUnavailable(
            "the offline Metal compiler requires Xcode with the metal and metallib tools"
        )

    with tempfile.TemporaryDirectory(prefix="neso-") as tmpdir:
        source_path = Path(tmpdir, "kernel.metal")
        air_path = Path(tmpdir, "kernel.air")
        library_path = Path(tmpdir, "kernel.metallib")
        source_path.write_text(msl_source)
        try:
            subprocess.run(
                [
                    metal,
                    "-c",
                    os.fspath(source_path),
                    "-o",
                    os.fspath(air_path),
                    "-std=metal3.1",
                    "-ffast-math",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [metallib, os.fspath(air_path), "-o", os.fspath(library_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            diagnostics = exc.stderr.strip() or exc.stdout.strip()
            raise RuntimeError(f"Metal compilation failed:\n{diagnostics}") from exc
        return library_path.read_bytes()
