"""
Abstract code emitter interface for target-specific code generation.

Each backend (MSL, HLSL, etc.) implements this interface to provide
target-specific syntax while sharing the common TTIR lowering logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .ir import FuncArg, TType


class CodeEmitter(ABC):
    """Abstract base for target-specific code emission."""

    # --- Type mapping ---

    @abstractmethod
    def map_dtype(self, triton_dtype: str) -> str:
        """Map Triton dtype (f32, f16, i32, ...) to target type name."""
        ...

    def map_dtype_unsigned(self, triton_dtype: str) -> str:
        """Map Triton integer dtype to unsigned target type. Default: same as map_dtype."""
        return self.map_dtype(triton_dtype)

    def scalar_type(self, ttype: TType) -> str:
        """Get the scalar type name for a TType."""
        return self.map_dtype(ttype.dtype)

    # --- Kernel structure ---

    @abstractmethod
    def file_header(self) -> str:
        """Includes/preamble at top of file."""
        ...

    @abstractmethod
    def kernel_signature(self, name: str, func_args: list[FuncArg]) -> str:
        """Complete kernel function signature including opening brace."""
        ...

    # --- Thread indexing ---

    @abstractmethod
    def thread_id_expr(self) -> str:
        """Expression for linear thread ID within threadgroup (e.g. 'tid.x')."""
        ...

    @abstractmethod
    def threadgroup_id_expr(self, axis: str) -> str:
        """Expression for threadgroup ID along axis 'x'/'y'/'z'."""
        ...

    @abstractmethod
    def grid_dim_expr(self, axis: str) -> str:
        """Expression for grid dimension along axis 'x'/'y'/'z' (num threadgroups)."""
        ...

    @abstractmethod
    def threads_per_group_expr(self) -> str:
        """Expression for threadgroup size (e.g. '_tg_size.x')."""
        ...

    # --- Memory ---

    @abstractmethod
    def shared_memory_decl(self, name: str, dtype: str, count: int) -> str:
        """Declaration for shared/threadgroup memory array."""
        ...

    @abstractmethod
    def barrier(self) -> str:
        """Threadgroup memory barrier statement."""
        ...

    # --- SIMD / wave operations ---

    @abstractmethod
    def supports_simd_matrix(self) -> bool:
        """Whether this target supports hardware matrix multiply (simdgroup_matrix, WaveMatrix)."""
        ...

    @abstractmethod
    def simd_matrix_type(self, dtype: str, M: int, N: int) -> str:
        """Type name for SIMD matrix (e.g. 'simdgroup_matrix<float, 8, 8>')."""
        ...

    @abstractmethod
    def simd_matrix_init(self, var: str, dtype: str, M: int, N: int, value: str) -> str:
        """Statement to declare and initialize a SIMD matrix."""
        ...

    @abstractmethod
    def simd_load(self, var: str, src_ptr: str, stride: str,
                  transpose: bool = False) -> str:
        """Statement to load data into a SIMD matrix from shared memory."""
        ...

    @abstractmethod
    def simd_store(self, var: str, dst_ptr: str, stride: str) -> str:
        """Statement to store a SIMD matrix to shared memory."""
        ...

    @abstractmethod
    def simd_multiply(self, c: str, a: str, b: str) -> str:
        """Statement for C = A * B (no accumulate)."""
        ...

    @abstractmethod
    def simd_multiply_accumulate(self, c: str, a: str, b: str, acc: str) -> str:
        """Statement for C = A * B + acc."""
        ...

    @abstractmethod
    def simd_reduce(self, op: str, expr: str) -> str:
        """Expression for SIMD-width reduction (sum, max, min)."""
        ...

    # --- Math ---

    def math_func(self, name: str) -> str:
        """Map math function name. Most are the same across targets."""
        return name

    # --- Casts ---

    @abstractmethod
    def cast_expr(self, target_type: str, expr: str) -> str:
        """Cast expression to target type."""
        ...

    @abstractmethod
    def bitcast_expr(self, target_type: str, expr: str) -> str:
        """Bitcast expression to target type."""
        ...

    # --- Extended methods (non-abstract, with defaults) ---

    def infinity_expr(self, negative: bool = False) -> str:
        """Expression for float infinity constant."""
        return "(-HUGE_VALF)" if negative else "HUGE_VALF"

    def supports_ptr_cast(self) -> bool:
        """Whether target supports C-style pointer casting for vec4 load/store.

        True for MSL (device/threadgroup pointer casts), False for HLSL.
        When False, vec4 optimizations that rely on pointer casts are disabled
        and scalar fallback paths are used instead.
        """
        return False

    # --- Vec4 operations (shared/device memory) ---

    def vec4_load_shared(self, dtype: str, arr: str, idx: str) -> str:
        """Expression that loads 4 consecutive elements from shared memory as vec4.

        dtype: element type name ('half' or 'float')
        arr: shared memory array name
        idx: base index expression
        """
        return f"*(({self._shared_qualifier()} {dtype}4*)&{arr}[{idx}])"

    def vec4_store_shared(self, dtype: str, arr: str, idx: str, val: str) -> str:
        """Statement that stores vec4 to 4 consecutive shared memory elements."""
        return f"*(({self._shared_qualifier()} {dtype}4*)&{arr}[{idx}]) = {val};"

    def vec4_load_device(self, dtype: str, buf: str, idx: str) -> str:
        """Expression that loads 4 consecutive elements from device buffer as vec4."""
        return f"*((device const {dtype}4*)&{buf}[{idx}])"

    def vec4_store_device(self, dtype: str, buf: str, idx: str, components: list) -> str:
        """Statement that stores 4 values to consecutive device buffer elements."""
        return f"*((device {dtype}4*)&{buf}[{idx}]) = {dtype}4({', '.join(components)});"

    def vec4_copy_device_to_shared(self, dtype: str, s_arr: str, s_idx: str,
                                   d_buf: str, d_idx: str) -> str:
        """Statement that copies 4 elements from device buffer to shared memory."""
        return (f"*(({self._shared_qualifier()} {dtype}4*)&{s_arr}[{s_idx}]) = "
                f"*((device const {dtype}4*)&{d_buf}[{d_idx}]);")

    def _shared_qualifier(self) -> str:
        """Address space qualifier for shared memory ('threadgroup' or 'groupshared')."""
        return "threadgroup"

    def requires_global_shared_memory(self) -> bool:
        """Whether shared memory declarations must be at global scope.

        True for HLSL (groupshared must be outside functions).
        False for MSL (threadgroup can be inside kernel functions).
        """
        return False

    # --- Wave / subgroup intrinsics ---

    def supports_wave_reduce(self) -> bool:
        """Whether this target supports wave/subgroup-level reductions.

        When True, reductions use two-level wave reduce (2 barriers)
        instead of tree-based shared memory reduction (log2(N) barriers).
        """
        return False

    def wave_lane_count_expr(self) -> str:
        """Expression for the number of lanes in a wave/subgroup."""
        return "32u"

    def wave_lane_index_expr(self) -> str:
        """Expression for this thread's lane index within its wave/subgroup."""
        return f"((uint){self.thread_id_expr()} % 32u)"

    def wave_id_expr(self, thread_id: str) -> str:
        """Expression for the wave containing ``thread_id``."""
        return f"((uint){thread_id} / {self.wave_lane_count_expr()})"

    def wave_count_expr(self, thread_count: str) -> str:
        """Expression for the number of waves covering ``thread_count`` threads."""
        lanes = self.wave_lane_count_expr()
        return f"(({thread_count} + {lanes} - 1u) / {lanes})"
