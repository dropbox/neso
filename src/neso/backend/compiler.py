"""
Neso backend compiler for Triton.

Compilation pipeline:
  Triton Python AST -> TTIR (Triton IR) -> MSL (Metal Shading Language)

The backend's final ``metalbin`` payload is UTF-8 MSL for the optional
development runtime. Offline metallib creation belongs to the AOT toolchain.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from triton.backends.compiler import BaseBackend, GPUTarget, Language


@dataclass(frozen=True)
class NesoOptions:
    num_warps: int = 4  # maps to SIMD groups
    num_stages: int = 1
    num_ctas: int = 1
    # Apple GPU SIMD width is 32 on all current Apple Silicon
    warp_size: int = 32
    # Threads per threadgroup (block size)
    threads_per_threadgroup: int = 256
    enable_fp_fusion: bool = True
    debug: bool = False
    backend_name: str = 'neso'
    sanitize_overflow: bool = True
    supported_fp8_dtypes: tuple = ()
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: tuple = ("ieee",)
    max_num_imprecise_acc_default: int = 0
    extern_libs: tuple[tuple[str, str], ...] | dict[str, str] = ()
    instrumentation_mode: str = ""

    def __post_init__(self):
        if isinstance(self.extern_libs, dict):
            object.__setattr__(self, 'extern_libs', tuple(self.extern_libs.items()))

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in sorted(self.__dict__.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class NesoBackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'neso'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        _enable_jit_function_conditionals()
        self.binary_ext = "metalbin"

    def parse_options(self, opts) -> Any:
        args = {}
        args.update({k: opts[k] for k in NesoOptions.__dataclass_fields__ if k in opts if opts[k] is not None})
        return NesoOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.get('shared', 0) if isinstance(metadata, dict) else getattr(metadata, 'shared', 0),
        )

    def get_codegen_implementation(self, options):
        codegen_fns = {
            "min_dot_size": lambda lhs, rhs: (1, 1, 1),
        }
        return codegen_fns

    def get_module_map(self) -> dict[str, ModuleType]:
        return {}

    def load_dialects(self, ctx):
        # No additional MLIR dialects needed for the Neso backend
        # The core Triton and TritonGPU dialects are loaded by the framework
        pass

    @staticmethod
    def make_ttir(mod, metadata, opt):
        """Apply standard TTIR optimization passes."""
        from triton._C.libtriton import ir, passes
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt):
        """Convert TTIR to TritonGPU IR with layout annotations."""
        from triton._C.libtriton import ir, passes

        # Phase 1: Convert TTIR → TTGIR (backend-agnostic)
        target_str = f"neso:{opt.num_warps}"
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(
            pm, target_str, opt.num_warps, opt.warp_size, opt.num_ctas)
        pm.run(mod, 'make_ttgir_convert')

        # Phase 2: Backend-agnostic TTGIR optimization passes
        pm2 = ir.pass_manager(mod.context)
        pm2.enable_debug()
        passes.ttgpuir.add_coalesce(pm2)
        passes.ttgpuir.add_remove_layout_conversions(pm2)
        passes.ttgpuir.add_optimize_thread_locality(pm2)
        passes.ttgpuir.add_optimize_dot_operands(pm2, False)
        passes.ttgpuir.add_remove_layout_conversions(pm2)
        passes.common.add_canonicalizer(pm2)
        passes.common.add_cse(pm2)
        passes.common.add_symbol_dce(pm2)
        pm2.run(mod, 'make_ttgir_opt')

        return mod

    @staticmethod
    def make_msl(mod, metadata, opt):
        """Convert optimized TTGIR to Metal Shading Language source."""
        from .codegen import ttir_to_msl_with_metadata

        block_size = opt.threads_per_threadgroup

        if os.environ.get('NESO_DUMP_TTGIR'):
            print("=== TTGIR ===")
            print(mod.str_nodebug())

        msl_source, kernel_name, _actual_block_size, recommended_threads = \
            ttir_to_msl_with_metadata(mod, block_size=block_size)

        if os.environ.get('NESO_DUMP_MSL'):
            print("=== TTIR ===")
            print(mod.str_nodebug())
            print("\n=== MSL ===")
            print(msl_source)

        metadata["name"] = kernel_name
        metadata["shared"] = 0
        total_threads = recommended_threads
        metadata["num_warps"] = (total_threads + 31) // 32
        metadata["num_ctas"] = opt.num_ctas

        # Extract param names for driver arg mapping.
        # str(mod) preserves original Python names via MLIR locations:
        #   %x_ptr: !tt.ptr<f32> loc("x_ptr"(...))
        # mod.str_nodebug() strips to %arg0, %arg1 which don't match Python names
        ttir_param_names = _extract_ttir_param_names(mod)
        metadata["ttir_param_names"] = ttir_param_names

        return msl_source

    @staticmethod
    def make_metalbin(src, metadata, opt):
        """Encode MSL for the optional development runtime."""
        metadata["metal_binary_format"] = "msl-source"
        return src.encode('utf-8')

    def add_stages(self, stages, options, language=Language.TRITON):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["msl"] = lambda src, metadata: self.make_msl(src, metadata, options)
        stages["metalbin"] = lambda src, metadata: self.make_metalbin(src, metadata, options)

    @staticmethod
    def hash():
        # Version identifier for the Neso backend
        return "neso-backend-v0.3"


def _enable_jit_function_conditionals() -> None:
    """Preserve ``if ACTIVATION:`` without patching upstream Triton 3.8."""
    from triton.compiler.code_generator import _condition_types
    from triton.runtime.jit import JITFunction

    _condition_types.add(JITFunction)


def _extract_ttir_param_names(mod) -> list[str]:
    """Extract source parameter names from typed TTIR function arguments.

    The TTIR uses named parameters like %a_ptr, %stride_bk, etc.
    Triton's optimizer may eliminate unused params (e.g., stride=1 gets
    constant-folded), so the TTIR function has fewer params than the
    original Python function. We need these names to correctly map
    driver args to kernel buffer indices.
    """
    func = mod.get_function(mod.get_entry_func_name())
    names = []
    for index in range(func.get_num_args()):
        location = str(func.args(index).get_loc())
        match = re.search(r'loc\("([^"]+)"', location)
        names.append(match.group(1) if match else f"arg{index}")
    return names
