#!/usr/bin/env python3
"""Compile the Windows harness's Triton kernels to HLSL and DXIL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from kernels import scale, vector_add
from neso.aot_compile import compile_kernel
from neso.backend.codegen import ttir_to_hlsl_with_metadata


BLOCK_SIZE = 256
KERNELS = {
    "vector_add": (vector_add, "*fp32, *fp32, *fp32, i32, 256"),
    "scale": (scale, "*fp32, *fp32, fp32, i32, 256"),
}


def write_if_changed(path: Path, data: str) -> None:
    if not path.exists() or path.read_text() != data:
        path.write_text(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parent.parent
    dxc = Path(os.environ.get(
        "DXC_PATH", repo.parent / "directxshadercompiler/build-release/bin/dxc"
    )).expanduser().resolve()
    if not dxc.is_file():
        raise SystemExit(f"DXC not found at {dxc}; set DXC_PATH")

    for name, (kernel, signature) in KERNELS.items():
        compiled = compile_kernel(
            fn=kernel,
            signature=signature,
            target="neso:2:32",
            grid=[f"cdiv(n_elements, {BLOCK_SIZE})", "1", "1"],
            emit_metallib=False,
        )
        hlsl, entry, _, threads, _ = ttir_to_hlsl_with_metadata(
            compiled.ttgir_text, block_size=BLOCK_SIZE
        )
        if threads != BLOCK_SIZE:
            raise RuntimeError(f"{name}: expected {BLOCK_SIZE} threads, got {threads}")
        hlsl_path = args.out / f"{name}.hlsl"
        dxil_path = args.out / f"{name}.dxil"
        dxil_tmp = dxil_path.with_suffix(".dxil.tmp")
        write_if_changed(hlsl_path, hlsl)
        subprocess.run([
            os.fspath(dxc), "-T", "cs_6_6", "-enable-16bit-types", "-O3",
            "-Wno-for-redefinition", "-E", entry, "-Fo", os.fspath(dxil_tmp),
            os.fspath(hlsl_path),
        ], check=True)
        if dxil_path.exists() and dxil_path.read_bytes() == dxil_tmp.read_bytes():
            dxil_tmp.unlink()
        else:
            dxil_tmp.replace(dxil_path)
        print(f"{name}: {threads} threads, {dxil_path.stat().st_size} byte DXIL")


if __name__ == "__main__":
    main()
