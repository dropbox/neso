#!/usr/bin/env python3
"""
Neso AOT Compiler

Compiles @triton.jit kernels ahead-of-time to MSL, optional metallib, and metadata
for use from Rust, Swift, C++, or any Metal-capable application.

Usage:
    python aot_compile.py kernels.py \
        --kernel matmul_kernel \
        --signature "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, 128, 128, 32" \
        --num-warps 8 \
        --grid "cdiv(M,128), cdiv(N,128), 1" \
        --output matmul

Produces:
    matmul.metallib   - compiled Metal library, when Xcode's tools are available
    matmul.metal      - MSL source (for inspection/debugging)
    matmul.json       - kernel metadata (buffer bindings, launch config)
    matmul.ttir       - Triton IR (for debugging)
    matmul.ttgir      - Triton GPU IR (for debugging)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .backend.abi import KernelParameter, build_parameters, parse_signature
    from .backend.toolchain import MetalToolchainUnavailable, compile_msl
else:
    # Preserve direct execution from a source checkout.
    from backend.abi import KernelParameter, build_parameters, parse_signature
    from backend.toolchain import MetalToolchainUnavailable, compile_msl


@dataclass
class AOTResult:
    """Result of AOT compilation."""
    kernel_name: str
    msl_source: str
    metallib_bytes: bytes | None
    ttir_text: str
    ttgir_text: str
    parameters: list[KernelParameter]
    constants: dict[str, Any]
    threadgroup_size: int
    num_warps: int
    grid: list[str]

    def save(self, output_prefix: str) -> dict[str, str]:
        """Save all artifacts to files.

        Writes:
            {prefix}.metallib  - compiled Metal library binary, when available
            {prefix}.metal     - MSL source code
            {prefix}.json      - kernel metadata
            {prefix}.ttir      - Triton IR (for debugging)
            {prefix}.ttgir     - Triton GPU IR (for debugging)

        Returns dict of output file paths.
        """
        prefix = Path(output_prefix)
        paths = {}

        if self.metallib_bytes is not None:
            metallib_path = prefix.with_suffix('.metallib')
            with open(metallib_path, 'wb') as f:
                f.write(self.metallib_bytes)
            paths['metallib'] = str(metallib_path)
            print(f"  {metallib_path} ({len(self.metallib_bytes):,} bytes)")

        # Write MSL source
        metal_path = prefix.with_suffix('.metal')
        with open(metal_path, 'w') as f:
            f.write(self.msl_source)
        paths['metal'] = str(metal_path)
        print(f"  {metal_path} ({len(self.msl_source):,} chars)")

        # Write metadata JSON
        json_path = prefix.with_suffix('.json')
        metadata = self.to_metadata()
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        paths['json'] = str(json_path)
        print(f"  {json_path}")

        # Write TTIR
        ttir_path = prefix.with_suffix('.ttir')
        with open(ttir_path, 'w') as f:
            f.write(self.ttir_text)
        paths['ttir'] = str(ttir_path)
        print(f"  {ttir_path}")

        ttgir_path = prefix.with_suffix('.ttgir')
        with open(ttgir_path, 'w') as f:
            f.write(self.ttgir_text)
        paths['ttgir'] = str(ttgir_path)
        print(f"  {ttgir_path}")

        return paths

    def to_metadata(self) -> dict[str, Any]:
        """Return metadata dict suitable for JSON serialization."""
        return {
            "kernel_name": self.kernel_name,
            "params": [parameter.to_metadata() for parameter in self.parameters],
            "constants": self.constants,
            "threadgroup_size": self.threadgroup_size,
            "num_warps": self.num_warps,
            "grid": self.grid,
        }

    @property
    def params(self) -> list[dict[str, Any]]:
        """Compatibility view of parameters as JSON-ready dictionaries."""
        return [parameter.to_metadata() for parameter in self.parameters]


def compile_kernel(
    fn,
    signature: str,
    num_warps: int = 4,
    grid: list[str] | None = None,
    target: str | None = None,
    require_metallib: bool = False,
    emit_metallib: bool = True,
) -> AOTResult:
    """Compile a @triton.jit kernel to Metal AOT artifacts.

    Args:
        fn: A @triton.jit decorated function
        signature: Comma-separated type list, e.g.
            "*fp16, *fp16, *fp16, i32, i32, i32, 128, 128, 32"
            Bare numeric values become constexpr constants.
            Types with ":16" hint get tt.divisibility attribute.
        num_warps: Number of SIMD groups per threadgroup
        grid: Grid dimensions as list of strings, e.g. ["cdiv(M,128)", "cdiv(N,128)", "1"]
        target: Target string "neso:arch:warp_size" (default: auto-detect)
        require_metallib: Fail if the offline Metal toolchain is unavailable
        emit_metallib: Attempt to compile MSL to a metallib when true

    Returns:
        AOTResult with all compilation artifacts
    """
    import triton
    import triton.compiler
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    arg_names = fn.arg_names
    jit_fn = fn.fn if hasattr(fn, "fn") else fn
    parsed_signature = parse_signature(arg_names, signature, jit_fn.__globals__)

    # Determine target
    if target:
        parts = target.split(':')
        if len(parts) != 3:
            raise ValueError("target must have the form 'neso:arch:warp_size'")
        if parts[0] != "neso":
            raise ValueError(f"expected a Neso target, got {parts[0]!r}")
        gpu_target = GPUTarget(
            parts[0],
            int(parts[1]) if parts[1].isdigit() else parts[1],
            int(parts[2]),
        )
    else:
        try:
            gpu_target = triton.runtime.driver.active.get_current_target()
        except RuntimeError:
            gpu_target = GPUTarget("neso", 2, 32)

    # Create compilation source
    # Construct the compiler input directly. JITFunction.create_binder() also
    # initializes a runtime driver, which is unnecessary for ahead-of-time
    # compilation and prevents AOT compilation on build hosts without a GPU.
    src = ASTSource(
        fn=fn,
        constexprs=parsed_signature.constants,
        signature=parsed_signature.triton_signature,
        attrs=parsed_signature.attributes,
    )

    # Compile
    compiled = triton.compile(src, target=gpu_target, options={"num_warps": num_warps})

    # Extract artifacts
    msl_source = compiled.asm.get("msl", "")
    if require_metallib and not emit_metallib:
        raise ValueError("require_metallib=True requires emit_metallib=True")
    metallib_bytes = None
    if emit_metallib:
        try:
            metallib_bytes = compile_msl(msl_source)
        except MetalToolchainUnavailable:
            if require_metallib:
                raise

    # Get TTIR and TTGIR text
    ttir_obj = compiled.asm.get("ttir", "")
    ttir_text = str(ttir_obj)
    ttgir_obj = compiled.asm.get("ttgir", "")
    ttgir_text = str(ttgir_obj) if ttgir_obj else ""

    # Build param list from ttir_param_names (the authoritative source for
    # which args survived optimization and their buffer indices)
    ttir_param_names = getattr(compiled.metadata, 'ttir_param_names', None) or []
    parameters = build_parameters(ttir_param_names, parsed_signature.source_types)

    # If ttir_param_names not available, fall back to non-constexpr args
    if not parameters:
        runtime_names = [name for name in arg_names if name not in parsed_signature.constants]
        parameters = build_parameters(runtime_names, parsed_signature.source_types)

    # Get metadata
    actual_warps = getattr(compiled.metadata, 'num_warps', num_warps)
    threadgroup_size = actual_warps * 32
    kernel_name = getattr(compiled.metadata, 'name', fn.__name__)

    return AOTResult(
        kernel_name=kernel_name,
        msl_source=msl_source,
        metallib_bytes=metallib_bytes,
        ttir_text=ttir_text,
        ttgir_text=ttgir_text,
        parameters=parameters,
        constants=dict(parsed_signature.constants),
        threadgroup_size=threadgroup_size,
        num_warps=actual_warps,
        grid=grid or ["1", "1", "1"],
    )


def compile_many(
    fn,
    configs: list[dict[str, Any]],
    target: str | None = None,
    require_metallib: bool = False,
) -> list[AOTResult]:
    """Compile multiple specializations of the same kernel.

    Args:
        fn: A @triton.jit decorated function
        configs: List of dicts, each with 'signature', 'num_warps', 'grid' keys
        target: Target string (default: auto-detect)
        require_metallib: Fail if the offline Metal toolchain is unavailable

    Returns:
        List of AOTResult, one per config
    """
    results = []
    for cfg in configs:
        result = compile_kernel(
            fn=fn,
            signature=cfg['signature'],
            num_warps=cfg.get('num_warps', 4),
            grid=cfg.get('grid'),
            target=target,
            require_metallib=require_metallib,
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Neso AOT Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simple vector add
    python aot_compile.py kernels.py \\
        --kernel add_kernel \\
        --signature "*fp32, *fp32, *fp32, i32, 1024" \\
        --grid "cdiv(n_elements, 1024), 1, 1" \\
        --output add

    # FP16 matmul
    python aot_compile.py kernels.py \\
        --kernel matmul_kernel \\
        --signature "*fp16:16, *fp16:16, *fp16:16, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 32" \\
        --num-warps 8 \\
        --grid "cdiv(M,128), cdiv(N,128), 1" \\
        --output matmul

Signature format:
    *fp16       pointer to fp16 buffer
    *fp16:16    pointer with 16-byte alignment hint
    i32         32-bit integer scalar
    128         constexpr constant (compiled into the kernel)
        """,
    )
    parser.add_argument("path", help="Path to Python file containing the kernel")
    parser.add_argument("--kernel", "-k", required=True,
                        help="Name of the @triton.jit kernel function")
    parser.add_argument("--signature", "-s", required=True,
                        help="Comma-separated argument types")
    parser.add_argument("--num-warps", "-w", type=int, default=4,
                        help="SIMD groups per threadgroup (default: 4)")
    parser.add_argument("--grid", "-g", default="1,1,1",
                        help="Grid dimensions, e.g. 'cdiv(M,128), cdiv(N,128), 1'")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file prefix (default: kernel name)")
    parser.add_argument("--target", "-t", default=None,
                        help="Target 'neso:arch:warp_size' (default: auto)")
    parser.add_argument("--require-metallib", action="store_true",
                        help="Fail unless Xcode can produce a metallib")

    args = parser.parse_args()

    # Import kernel from Python file
    path = Path(args.path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    kernel = getattr(mod, args.kernel, None)
    if kernel is None:
        available = [name for name in dir(mod)
                     if hasattr(getattr(mod, name), 'arg_names')]
        print(f"Error: kernel '{args.kernel}' not found in {path}", file=sys.stderr)
        if available:
            print(f"Available kernels: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)

    grid = [g.strip() for g in args.grid.split(',')]
    output = args.output or args.kernel

    print(f"Compiling {args.kernel} from {path}...")
    result = compile_kernel(
        fn=kernel,
        signature=args.signature,
        num_warps=args.num_warps,
        grid=grid,
        target=args.target,
        require_metallib=args.require_metallib,
    )

    print(f"\nKernel: {result.kernel_name}")
    print(f"Threadgroup: {result.threadgroup_size} threads ({result.num_warps} SIMD groups)")
    print(f"Constants: {result.constants}")
    print(f"Params ({len(result.params)} buffers):")
    for p in result.params:
        ptr_tag = "buffer" if p['is_pointer'] else "scalar"
        print(f"  [{p['index']}] {p['name']}: {p['metal_type']}  ({ptr_tag})")
    print(f"Grid: ({', '.join(result.grid)})")

    print("\nOutput files:")
    result.save(output)
    print("\nDone!")


if __name__ == "__main__":
    main()
