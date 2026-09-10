#!/usr/bin/env python3
"""Compile the macOS Candle harness's Triton kernels to Metal libraries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from neso.aot_compile import compile_kernel
from neso.backend.codegen import ttir_to_msl_with_metadata
from neso.backend.toolchain import compile_msl
from tests.test_flash_attention import generate_fa2_ttir
from windows.kernels import scale, vector_add


BLOCK_SIZE = 256
KERNELS = {
    "vector_add": (vector_add, "*fp32, *fp32, *fp32, i32, 256"),
    "scale": (scale, "*fp32, *fp32, fp32, i32, 256"),
}
FA2_SIMD_BLOCK_M = 16
FA2_SCALAR_BLOCK_M = 8
FA2_BLOCK_N = 32
FA2_HEAD_DIM = 64


def write_if_changed(path: Path, data: bytes) -> None:
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, (kernel, signature) in KERNELS.items():
        compiled = compile_kernel(
            fn=kernel,
            signature=signature,
            target="neso:2:32",
            grid=[f"cdiv(n_elements, {BLOCK_SIZE})", "1", "1"],
            require_metallib=True,
        )
        if compiled.threadgroup_size != BLOCK_SIZE:
            raise RuntimeError(
                f"{name}: expected {BLOCK_SIZE} threads, "
                f"got {compiled.threadgroup_size}"
            )
        assert compiled.metallib_bytes is not None
        output = args.out / f"{name}.metallib"
        write_if_changed(output, compiled.metallib_bytes)
        print(f"{name}: {compiled.threadgroup_size} threads, "
              f"{output.stat().st_size} byte metallib")

    for variant, use_simdgroup, block_m in (
        ("simd", True, FA2_SIMD_BLOCK_M),
        ("scalar", False, FA2_SCALAR_BLOCK_M),
    ):
        qkv_dtype = "f16" if use_simdgroup else "f32"
        ttir = generate_fa2_ttir(
            block_m, FA2_BLOCK_N, FA2_HEAD_DIM, qkv_dtype=qkv_dtype
        )
        msl, _, _, threads = ttir_to_msl_with_metadata(
            ttir,
            block_size=block_m * FA2_BLOCK_N,
            use_simdgroup=use_simdgroup,
            max_threads=0,
        )
        output = args.out / f"flash_attention_fwd_{variant}.metallib"
        write_if_changed(output, compile_msl(msl))
        print(f"flash_attention_fwd_{variant}: {threads} threads, "
              f"{output.stat().st_size} byte metallib")


if __name__ == "__main__":
    main()
