"""
Triton TTIR -> target code generation package.

Public API:
    ttir_to_msl(mod_or_text, block_size, use_simdgroup) -> (msl_source, kernel_name)
    ttir_to_hlsl(mlir_text, block_size) -> (hlsl_source, kernel_name)
"""
from __future__ import annotations

from .hlsl_emitter import HLSLEmitter
from .ir import parse_type
from .lowering import TritonLowering
from .mlir_walker import walk_module, walk_module_from_text
from .msl_emitter import MSLEmitter


def _extract_ir(mod_or_text):
    """Extract IR from either a module object or MLIR text string."""
    if isinstance(mod_or_text, str):
        return walk_module_from_text(mod_or_text)
    else:
        return walk_module(mod_or_text)


def ttir_to_msl(mod_or_text, block_size: int = 256,
                use_simdgroup: bool = True) -> tuple[str, str]:
    """Convert Triton TTIR to Metal Shading Language source.

    Args:
        mod_or_text: Either a triton._C.libtriton.ir.module or MLIR text string.
        block_size: Threads per threadgroup.
        use_simdgroup: Whether to use simdgroup_matrix hardware MMA.
            True = Apple Silicon (default), False = Intel/older GPUs.

    Returns (msl_source, kernel_name).
    """

    func_name, func_args, ops = _extract_ir(mod_or_text)

    if func_name is None:
        raise ValueError("Could not find tt.func in MLIR input")

    emitter = MSLEmitter(use_simdgroup=use_simdgroup)
    lowering = TritonLowering(emitter, block_size=block_size)
    msl_source = lowering.generate(func_name, func_args, ops)
    lowering._actual_block_size = lowering.block_size
    return msl_source, func_name


def ttir_to_msl_with_metadata(mod_or_text, block_size: int = 256,
                               use_simdgroup: bool = True,
                               max_threads: int = 0) -> tuple[str, str, int, int]:
    """Like ttir_to_msl but also returns lowering metadata.

    Args:
        max_threads: Override threadgroup size (0 = auto).
            Set to num_warps*32 to enable correct grid-stride codegen
            for element-wise kernels where threads < block_size.

    Returns (msl_source, kernel_name, actual_block_size, recommended_threads).
    """

    func_name, func_args, ops = _extract_ir(mod_or_text)

    if func_name is None:
        raise ValueError("Could not find tt.func in MLIR input")

    emitter = MSLEmitter(use_simdgroup=use_simdgroup)
    lowering = TritonLowering(emitter, block_size=block_size,
                               max_threads=max_threads)
    msl_source = lowering.generate(func_name, func_args, ops)
    return msl_source, func_name, lowering.block_size, lowering.recommended_threads


def ttir_to_hlsl(mod_or_text,
                 block_size: int = 256,
                 wave_size: int | None = None) -> tuple[str, str]:
    """Convert Triton TTIR to HLSL compute shader source.

    Targets Shader Model 6.6 with native half (float16_t) support. ``wave_size``
    can fix the wave width for a known target; by default the driver selects
    its preferred width. Compile with DXC: dxc -T cs_6_6 -enable-16bit-types

    Returns (hlsl_source, kernel_name).
    """
    func_name, func_args, ops = _extract_ir(mod_or_text)

    if func_name is None:
        raise ValueError("Could not find tt.func in MLIR input")

    emitter = HLSLEmitter(wave_size=wave_size)
    lowering = TritonLowering(emitter, block_size=block_size)
    hlsl_source = lowering.generate(func_name, func_args, ops)
    hlsl_source = hlsl_source.replace(
        HLSLEmitter.NUMTHREADS_PLACEHOLDER, str(lowering.recommended_threads))
    return hlsl_source, func_name


def ttir_to_hlsl_with_metadata(mod_or_text,
                                block_size: int = 256,
                                force_acc_fp16: bool = False,
                                max_threads: int = 0,
                                wave_size: int | None = None) -> tuple[str, str, int, int, set[int]]:
    """Like ttir_to_hlsl but also returns lowering metadata.

    Args:
        force_acc_fp16: Force fp16 accumulation in matmul inner loops.
        max_threads: Override threadgroup size (0 = auto).
        wave_size: Fix the HLSL wave width, or let the driver choose when None.

    Returns (hlsl_source, kernel_name, actual_block_size, recommended_threads, half4_args).
    """
    func_name, func_args, ops = _extract_ir(mod_or_text)

    if func_name is None:
        raise ValueError("Could not find tt.func in MLIR input")

    emitter = HLSLEmitter(wave_size=wave_size)
    lowering = TritonLowering(emitter, block_size=block_size,
                               force_acc_fp16=force_acc_fp16,
                               max_threads=max_threads)
    hlsl_source = lowering.generate(func_name, func_args, ops)
    hlsl_source = hlsl_source.replace(
        HLSLEmitter.NUMTHREADS_PLACEHOLDER, str(lowering.recommended_threads))
    return hlsl_source, func_name, lowering.block_size, lowering.recommended_threads, set()
