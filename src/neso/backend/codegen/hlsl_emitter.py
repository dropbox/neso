"""
HLSL compute shader code emitter targeting Shader Model 6.6 with native half.

Implements the CodeEmitter interface for DirectX 12 / D3D12 compute shaders.
Shader Model 6.6 permits an optional fixed wave size; otherwise the driver
selects its preferred size for the target GPU.
"""
from __future__ import annotations

from .emitter_base import CodeEmitter
from .ir import FuncArg

TRITON_TO_HLSL_DTYPE = {
    "f16": "half",       # SM 6.2+ native float16_t / half
    "f32": "float",
    "f64": "double",
    "bf16": "uint16_t",  # No native bf16; store as raw bits
    "i1": "bool",
    "i8": "int",         # No 8-bit type in HLSL; widen to int
    "i16": "int16_t",    # SM 6.2+ with -enable-16bit-types
    "i32": "int",
    "i64": "int64_t",
    "u8": "uint",        # Widen to uint
    "u16": "uint16_t",
    "u32": "uint",
    "u64": "uint64_t",
}

TRITON_TO_HLSL_UNSIGNED = {
    "i8": "uint",
    "i16": "uint16_t",
    "i32": "uint",
    "i64": "uint64_t",
}


class HLSLEmitter(CodeEmitter):
    """HLSL compute shader emitter for Shader Model 6.6."""

    # Placeholder replaced after lowering determines actual threadgroup size.
    NUMTHREADS_PLACEHOLDER = "/*NUMTHREADS*/"
    VALID_WAVE_SIZES = frozenset({4, 8, 16, 32, 64, 128})

    def __init__(self, wave_size: int | None = None):
        if wave_size is not None and wave_size not in self.VALID_WAVE_SIZES:
            raise ValueError(
                f"wave_size must be one of {sorted(self.VALID_WAVE_SIZES)}, got {wave_size}"
            )
        self.wave_size = wave_size

    # --- Type mapping ---

    def map_dtype(self, triton_dtype: str) -> str:
        return TRITON_TO_HLSL_DTYPE.get(triton_dtype, triton_dtype)

    def map_dtype_unsigned(self, triton_dtype: str) -> str:
        return TRITON_TO_HLSL_UNSIGNED.get(triton_dtype, self.map_dtype(triton_dtype))

    # --- Kernel structure ---

    def file_header(self) -> str:
        wave_helpers = ""
        if self.wave_size is None:
            wave_helpers = (
                "uint _wave_lane_shift() {\n"
                "    return (uint)firstbitlow(WaveGetLaneCount());\n"
                "}\n\n"
            )
        return (
            "// Auto-generated HLSL compute shader — SM 6.6, DXC\n"
            "// Requires: -enable-16bit-types -T cs_6_6\n"
            "\n"
            "// MSL compatibility defines\n"
            "#define HUGE_VALF asfloat(0x7F800000u)\n"
            "#define NAN asfloat(0x7FC00000u)\n"
            "\n"
            + wave_helpers
            + "// Polynomial erf approximation (Abramowitz & Stegun 7.1.26)\n"
            "float _erf_approx(float x) {\n"
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
            "float _erfc_approx(float x) { return 1.0f - _erf_approx(x); }\n"
            "float _copysign_f(float x, float y) { return abs(x) * sign(y); }\n"
        )

    def kernel_signature(self, name: str, func_args: list[FuncArg]) -> str:
        lines = []

        # Separate pointer args (→ RWStructuredBuffer) from scalar args (→ cbuffer)
        ptr_args = [a for a in func_args if a.ttype.is_ptr]
        scalar_args = [a for a in func_args if not a.ttype.is_ptr]

        # Global resource declarations: all pointer args as UAVs
        for i, arg in enumerate(ptr_args):
            hlsl_type = self.map_dtype(arg.ttype.dtype)
            lines.append(
                f"RWStructuredBuffer<{hlsl_type}> arg{arg.index} : register(u{i});")

        lines.append("")

        # cbuffer for scalar args + grid dimensions
        lines.append("cbuffer Params : register(b0) {")
        for arg in scalar_args:
            hlsl_type = self.map_dtype(arg.ttype.dtype)
            lines.append(f"    {hlsl_type} arg{arg.index};")
        # Grid dimensions (needed for num_programs)
        lines.append("    uint _grid_dim_x;")
        lines.append("    uint _grid_dim_y;")
        lines.append("    uint _grid_dim_z;")
        lines.append("};")
        lines.append("")

        # Compile-time threadgroup size (placeholder replaced post-lowering)
        ph = self.NUMTHREADS_PLACEHOLDER
        lines.append(f"static const uint3 _tg_size = uint3({ph}, 1, 1);")
        lines.append("")

        # Entry point
        if self.wave_size is not None:
            lines.append(f"[WaveSize({self.wave_size})]")
        lines.append(f"[numthreads({ph}, 1, 1)]")
        lines.append(f"void {name}(")
        lines.append("    uint3 _tgid : SV_GroupID,")
        lines.append("    uint3 _tid_in_tg : SV_GroupThreadID")
        lines.append(") {")

        return '\n'.join(lines)

    # --- Thread indexing ---

    def thread_id_expr(self) -> str:
        return "_tid_in_tg.x"

    def threadgroup_id_expr(self, axis: str) -> str:
        return f"_tgid.{axis}"

    def grid_dim_expr(self, axis: str) -> str:
        return f"_grid_dim_{axis}"

    def threads_per_group_expr(self) -> str:
        return "_tg_size.x"

    # --- Memory ---

    def shared_memory_decl(self, name: str, dtype: str, count: int) -> str:
        hlsl_type = self.map_dtype(dtype)
        return f"groupshared {hlsl_type} {name}[{count}];"

    def barrier(self) -> str:
        return "GroupMemoryBarrierWithGroupSync();"

    # --- SIMD / wave operations ---

    def supports_simd_matrix(self) -> bool:
        return False  # No wave matrix support initially

    def simd_matrix_type(self, dtype: str, M: int, N: int) -> str:
        # Not supported — placeholder
        return f"/* unsupported simd_matrix<{dtype},{M},{N}> */"

    def simd_matrix_init(self, var: str, dtype: str, M: int, N: int, value: str) -> str:
        return f"/* unsupported simd_matrix_init {var} */"

    def simd_load(self, var: str, src_ptr: str, stride: str,
                  transpose: bool = False) -> str:
        return f"/* unsupported simd_load {var} */"

    def simd_store(self, var: str, dst_ptr: str, stride: str) -> str:
        return f"/* unsupported simd_store {var} */"

    def simd_multiply(self, c: str, a: str, b: str) -> str:
        return "/* unsupported simd_multiply */"

    def simd_multiply_accumulate(self, c: str, a: str, b: str, acc: str) -> str:
        return "/* unsupported simd_multiply_accumulate */"

    def simd_reduce(self, op: str, expr: str) -> str:
        func_map = {
            'add': 'WaveActiveSum', 'sum': 'WaveActiveSum',
            'max': 'WaveActiveMax', 'min': 'WaveActiveMin',
        }
        func = func_map.get(op, 'WaveActiveSum')
        return f"{func}({expr})"

    # --- Math ---

    def math_func(self, name: str) -> str:
        hlsl_map = {
            'rint': 'round',       # HLSL round = round-half-to-even
            'rsqrt': 'rsqrt',
            'copysign': '_copysign_f',
            'fma': 'mad',          # HLSL fma() is double-only; mad() is float
        }
        return hlsl_map.get(name, name)

    # --- Casts ---

    def cast_expr(self, target_type: str, expr: str) -> str:
        return f"({target_type})({expr})"

    def bitcast_expr(self, target_type: str, expr: str) -> str:
        # HLSL bitcast depends on source/target types
        bitcast_map = {
            'float': 'asfloat',
            'int': 'asint',
            'uint': 'asuint',
            'half': 'asfloat16',
        }
        func = bitcast_map.get(target_type)
        if func:
            return f"{func}({expr})"
        # Fallback: not all conversions have HLSL intrinsics
        return f"/* bitcast to {target_type} */ ({target_type})({expr})"

    # --- Extended methods ---

    def infinity_expr(self, negative: bool = False) -> str:
        if negative:
            return "asfloat(0xFF800000u)"  # -inf
        return "asfloat(0x7F800000u)"      # +inf

    def supports_ptr_cast(self) -> bool:
        return False

    def requires_global_shared_memory(self) -> bool:
        return True

    # --- Vec4 operations (constructor/component syntax for HLSL) ---

    def vec4_load_shared(self, dtype: str, arr: str, idx: str) -> str:
        return f"{dtype}4({arr}[{idx}], {arr}[({idx}) + 1u], {arr}[({idx}) + 2u], {arr}[({idx}) + 3u])"

    def vec4_store_shared(self, dtype: str, arr: str, idx: str, val: str) -> str:
        return (f"{arr}[{idx}] = {val}.x; {arr}[({idx}) + 1u] = {val}.y; "
                f"{arr}[({idx}) + 2u] = {val}.z; {arr}[({idx}) + 3u] = {val}.w;")

    def vec4_load_device(self, dtype: str, buf: str, idx: str) -> str:
        return f"{dtype}4({buf}[{idx}], {buf}[({idx}) + 1], {buf}[({idx}) + 2], {buf}[({idx}) + 3])"

    def vec4_store_device(self, dtype: str, buf: str, idx: str, components: list) -> str:
        stmts = [f"{buf}[({idx}) + {i}] = ({dtype}){c};" for i, c in enumerate(components)]
        return " ".join(stmts)

    def vec4_copy_device_to_shared(self, dtype: str, s_arr: str, s_idx: str,
                                   d_buf: str, d_idx: str) -> str:
        return (f"{s_arr}[{s_idx}] = {d_buf}[{d_idx}]; "
                f"{s_arr}[({s_idx}) + 1u] = {d_buf}[({d_idx}) + 1]; "
                f"{s_arr}[({s_idx}) + 2u] = {d_buf}[({d_idx}) + 2]; "
                f"{s_arr}[({s_idx}) + 3u] = {d_buf}[({d_idx}) + 3];")

    def _shared_qualifier(self) -> str:
        return "groupshared"

    # --- Wave intrinsics (SM 6.0+) ---

    def supports_wave_reduce(self) -> bool:
        return True

    def wave_lane_count_expr(self) -> str:
        if self.wave_size is not None:
            return f"{self.wave_size}u"
        return "WaveGetLaneCount()"

    def wave_id_expr(self, thread_id: str) -> str:
        shift = (self.wave_size.bit_length() - 1
                 if self.wave_size is not None else "_wave_lane_shift()")
        return f"((uint){thread_id} >> {shift})"

    def wave_count_expr(self, thread_count: str) -> str:
        if self.wave_size is not None:
            shift = self.wave_size.bit_length() - 1
            return f"(({thread_count} + {self.wave_size - 1}u) >> {shift})"
        return (f"(({thread_count} + WaveGetLaneCount() - 1u) "
                ">> _wave_lane_shift())")

    def wave_lane_index_expr(self) -> str:
        return "WaveGetLaneIndex()"
