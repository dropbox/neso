"""
Metal Shading Language (MSL) code emitter.

Implements the CodeEmitter interface for Apple Metal GPU targets.
"""
from __future__ import annotations

from .emitter_base import CodeEmitter
from .ir import FuncArg

TRITON_TO_MSL_DTYPE = {
    "fp16": "half",
    "f16": "half",
    "fp32": "float",
    "f32": "float",
    "fp64": "double",
    "f64": "double",
    "bf16": "bfloat",  # Metal 3.1+ on Apple Silicon
    "i1": "bool",
    "i8": "char",
    "i16": "short",
    "i32": "int",
    "i64": "long",
    "u8": "uchar",
    "u16": "ushort",
    "u32": "uint",
    "u64": "ulong",
}

TRITON_TO_MSL_UNSIGNED = {
    "i8": "uchar",
    "i16": "ushort",
    "i32": "uint",
    "i64": "ulong",
}


def msl_type_for_triton(triton_type: str) -> str:
    """Return the MSL ABI spelling for a Triton signature type."""
    if triton_type.startswith("*"):
        element_type = TRITON_TO_MSL_DTYPE.get(triton_type[1:], triton_type[1:])
        return f"device {element_type}*"
    return TRITON_TO_MSL_DTYPE.get(triton_type, triton_type)


class MSLEmitter(CodeEmitter):
    """Metal Shading Language code emitter."""

    def __init__(self, use_simdgroup: bool = True):
        self.use_simdgroup = use_simdgroup

    # --- Type mapping ---

    def map_dtype(self, triton_dtype: str) -> str:
        return TRITON_TO_MSL_DTYPE.get(triton_dtype, triton_dtype)

    def map_dtype_unsigned(self, triton_dtype: str) -> str:
        return TRITON_TO_MSL_UNSIGNED.get(triton_dtype, self.map_dtype(triton_dtype))

    # --- Kernel structure ---

    def file_header(self) -> str:
        return (
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "\n"
            "// Polynomial erf approximation (Abramowitz & Stegun 7.1.26, max err ~1.5e-7)\n"
            "static inline float _erf_approx(float x) {\n"
            "    float ax = abs(x);\n"
            "    float t = 1.0f / (1.0f + 0.3275911f * ax);\n"
            "    float t2 = t * t;\n"
            "    float t3 = t2 * t;\n"
            "    float t4 = t3 * t;\n"
            "    float t5 = t4 * t;\n"
            "    float p = 0.254829592f * t - 0.284496736f * t2\n"
            "            + 1.421413741f * t3 - 1.453152027f * t4\n"
            "            + 1.061405429f * t5;\n"
            "    float r = 1.0f - p * exp(-ax * ax);\n"
            "    return x >= 0.0f ? r : -r;\n"
            "}\n"
            "static inline float _erfc_approx(float x) { return 1.0f - _erf_approx(x); }\n"
        )

    def kernel_signature(self, name: str, func_args: list[FuncArg]) -> str:
        params = []
        for arg in func_args:
            msl_name = f"arg{arg.index}"
            if arg.ttype.is_ptr:
                metal_type = self.map_dtype(arg.ttype.dtype)
                params.append(f"    device {metal_type}* {msl_name} [[buffer({arg.index})]]")
            else:
                metal_type = self.map_dtype(arg.ttype.dtype)
                params.append(f"    constant {metal_type}& {msl_name} [[buffer({arg.index})]]")
        params.append("    uint3 _tgid [[threadgroup_position_in_grid]]")
        params.append("    uint3 _tid_in_tg [[thread_position_in_threadgroup]]")
        params.append("    uint3 _tg_size [[threads_per_threadgroup]]")
        params.append("    uint3 _grid_size [[threadgroups_per_grid]]")
        params_str = ',\n'.join(params)
        return f"kernel void {name}(\n{params_str}\n) {{"

    # --- Thread indexing ---

    def thread_id_expr(self) -> str:
        return "_tid_in_tg.x"

    def threadgroup_id_expr(self, axis: str) -> str:
        return f"_tgid.{axis}"

    def grid_dim_expr(self, axis: str) -> str:
        return f"_grid_size.{axis}"

    def threads_per_group_expr(self) -> str:
        return "_tg_size.x"

    # --- Memory ---

    def shared_memory_decl(self, name: str, dtype: str, count: int) -> str:
        metal_type = self.map_dtype(dtype)
        return f"threadgroup {metal_type} {name}[{count}];"

    def barrier(self) -> str:
        return "threadgroup_barrier(mem_flags::mem_threadgroup);"

    # --- SIMD / wave operations ---

    def supports_simd_matrix(self) -> bool:
        return self.use_simdgroup

    def simd_matrix_type(self, dtype: str, M: int, N: int) -> str:
        metal_type = self.map_dtype(dtype)
        return f"simdgroup_matrix<{metal_type}, {M}, {N}>"

    def simd_matrix_init(self, var: str, dtype: str, M: int, N: int, value: str) -> str:
        mat_type = self.simd_matrix_type(dtype, M, N)
        return f"{mat_type} {var}({value});"

    def simd_load(self, var: str, src_ptr: str, stride: str,
                  transpose: bool = False) -> str:
        if transpose:
            return f"simdgroup_load({var}, {src_ptr}, {stride}, ulong2(0, 0), true);"
        return f"simdgroup_load({var}, {src_ptr}, {stride});"

    def simd_store(self, var: str, dst_ptr: str, stride: str) -> str:
        return f"simdgroup_store({var}, {dst_ptr}, {stride});"

    def simd_multiply(self, c: str, a: str, b: str) -> str:
        return f"simdgroup_multiply({c}, {a}, {b});"

    def simd_multiply_accumulate(self, c: str, a: str, b: str, acc: str) -> str:
        return f"simdgroup_multiply_accumulate({c}, {a}, {b}, {acc});"

    def simd_reduce(self, op: str, expr: str) -> str:
        func_map = {'add': 'simd_sum', 'sum': 'simd_sum',
                     'max': 'simd_max', 'min': 'simd_min'}
        func = func_map.get(op, 'simd_sum')
        return f"{func}({expr})"

    # --- Math ---

    def math_func(self, name: str) -> str:
        # MSL uses 'rint' for rounding to nearest integer
        if name == 'round':
            return 'rint'
        return name

    # --- Casts ---

    def cast_expr(self, target_type: str, expr: str) -> str:
        return f"({target_type})({expr})"

    def bitcast_expr(self, target_type: str, expr: str) -> str:
        return f"as_type<{target_type}>({expr})"

    # --- Extended methods ---

    def supports_ptr_cast(self) -> bool:
        return True

    # --- Wave / subgroup intrinsics ---

    def supports_wave_reduce(self) -> bool:
        return self.use_simdgroup

    def wave_lane_count_expr(self) -> str:
        return "32u"

    def wave_lane_index_expr(self) -> str:
        return "((uint)_tid_in_tg.x % 32u)"
