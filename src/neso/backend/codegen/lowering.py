"""
TTIR lowering engine: translates parsed TTIR ops into target code via a CodeEmitter.

This module contains the backend-agnostic lowering logic that decomposes
block-level tensor operations into per-thread work (1D kernels) or shared-memory
tile operations (2D kernels). Target-specific syntax (MSL, HLSL, etc.) is
delegated to the CodeEmitter interface.

Design principles:
  - 1D tensors and scalars use per-thread register expressions (one value per thread).
  - 2D tensors are stored in threadgroup shared memory ("tiles"). Operations on
    tiles use cooperative multi-pass loops so any tile size is supported regardless
    of thread count.
  - tt.dot reads inputs from tiles, computes via simdgroup MMA (or scalar fallback),
    and writes output to a tile.
  - tt.reduce reads from a tile and produces a smaller tile (or per-thread value).
"""
from __future__ import annotations

import re
from typing import ClassVar

from .analysis import build_op_map, compute_liveness, flatten_ops
from .emitter_base import CodeEmitter
from .ir import FuncArg, Op, SSAValue, TType
from .model import RegAccInfo, TileInfo, UnsupportedOperationError


class TritonLowering:
    """Lowers parsed TTIR operations to target code via a CodeEmitter."""

    MAX_THREADS = 1024  # Metal hardware limit

    def __init__(self, emitter: CodeEmitter, block_size: int = 256,
                 force_acc_fp16: bool = False, max_threads: int = 0):
        self.emitter = emitter
        self.block_size = block_size
        self.force_acc_fp16 = force_acc_fp16
        self._max_threads_override = max_threads  # 0 = auto
        self._half4_bufs: set[str] = set()  # buffer names using RWStructuredBuffer<half4>
        self.var_counter = 0
        self.ssa_map: dict[str, SSAValue] = {}
        self.lines: list[str] = []
        self.indent = 1
        # Block shape for 2D kernels (from tt.dot result shape)
        self.block_shape: list[int] | None = None
        # Threadgroup memory declarations (legacy, used by matmul-optimized path)
        self.tg_memory_decls: list[str] = []
        # Dual expression tracking for 2D indexing (legacy, 1D-kernel compat)
        self._dual_exprs: dict[str, tuple[str, str]] = {}
        # Track program_id dimensions used
        self.pid_dims_used: set[str] = set()
        # Program ID expressions by axis
        self._pid_exprs: dict[str, str] = {}
        # Function arguments (set during generate)
        self.func_args: list[FuncArg] = []
        # Flag: dot-optimized path already emitted the C store
        self._dot_store_emitted = False
        # SSA values consumed by the dot-optimized store (truncf, convert_layout
        # chain between loop result and tt.store). Ops producing only these values
        # are skipped to avoid wasting threadgroup memory on unused tiles.
        self._dot_consumed_ssa: set = set()
        # Matmul C store params for post-ops efficient store (avoids index tile)
        self._matmul_store_params: dict | None = None
        # --- Tile infrastructure ---
        self._tiles: dict[str, TileInfo] = {}  # SSA name -> tile info
        self._op_map: dict[str, Op] = {}       # SSA result name -> producing Op
        # --- Tile memory reuse ---
        self._tile_decls: list[str] = []        # Hoisted threadgroup declarations
        self._tg_bytes_allocated: int = 0       # Total threadgroup memory allocated (bytes)
        self._tile_pool: dict[tuple[int, str], list[str]] = {}  # (total, dtype) -> free shared_names
        self._backing_refs: dict[str, set[str]] = {}  # shared_name -> set of live SSA names
        self._pinned_backing: set[str] = set()  # Backing stores that must never be freed
        self._const_fill: dict[str, str] = {}  # SSA name -> scalar fill value for dense constants
        self._deferred_tiles: dict[str, tuple] = {}  # SSA name -> (shape, dtype, fill_val, target_type)
        self._last_use: dict[str, int] = {}     # SSA name -> global op index of last use
        self._current_op_idx: int = 0           # Current op index during generation
        # --- Register accumulator tiles ---
        self._reg_tiles: dict[str, RegAccInfo] = {}  # SSA name -> register accumulator info
        # --- Lazy barrier optimization ---
        self._barrier_pending: bool = False
        self._barrier_loop_size: int = 0  # tile loop total of last pending write
        # --- Tile loop fusion ---
        self._fused_loop_bodies: list[str] = []  # pending loop bodies to fuse
        self._fused_loop_total: int = 0           # total elements for pending fused loop
        self._fused_loop_barrier: bool = False    # whether fused loop needs barrier
        self._fused_loop_dirty: set[str] = set()  # tiles dirtied in current fused batch
        # --- Dirty tile tracking (for smart barrier elision) ---
        self._dirty_tiles: set[str] = set()       # shared_names modified since last barrier
        # --- Thread count optimization ---
        self._max_coop_load_size: int = 0          # Largest cooperative load (elements)
        # --- Diagonal scratch buffer (reused across _emit_reg_diag_op calls) ---
        self._diag_scratch: str | None = None   # shared_name for 64-element diag scratch
        # --- Register accumulator tracking ---
        self._reg_acc_bps: int = 0                  # Max blocks_per_sg across reg accumulators
        self._reg_acc_min_sgs: int = 0              # Min SIMD groups needed for reg acc coverage
        # --- Fused scale skip set ---
        self._fused_scale_skip: set[str] = set()    # SSA result names to skip (fused into matmul store)
        # --- Register accumulator cast store ---
        self._reg_acc_cast_dtype: dict[str, str] = {}  # SSA name -> target dtype for cast store
        self._cast_scratch_name: str | None = None   # shared_name for per-SG cast scratch buffer
        # Cast writes that a later cooperative load can synchronize.
        self._deferred_cast_barriers: set[str] = set()
        # --- Index expression tracking (for broadcast materialization) ---
        # Maps SSA name -> (base_expr, start) such that value at position i = base_expr + (i + start)
        self._index_exprs: dict[str, tuple[str, int]] = {}
        # Maps SSA name -> (varying_axis, n_cols) for expand_dims results
        # varying_axis=0 means varies by column, =1 means varies by row
        self._expand_axis: dict[str, tuple[int, int]] = {}
        # Virtual tile expressions: SSA name -> MSL expression template with "_fi" placeholder
        # These represent broadcast values computable per-element without shared memory
        self._virtual_tiles: dict[str, tuple[str, list[int], str]] = {}  # name -> (expr, shape, dtype)
        # Buffer base tracking for non-pointer-cast backends (HLSL).
        # Maps SSA name → original buffer argument expression.
        # For HLSL, ptr arithmetic is decomposed: expr = offset, _buf_base = buffer name.
        self._buf_base: dict[str, str] = {}  # SSA name -> base buffer expr
        # --- Register tile optimization ---
        self._register_tile_names: set[str] = set()  # shared_names that are per-thread registers
        self._register_decls: list[str] = []  # local variable declarations for register tiles
        # --- Wave reduction batching ---
        # Deferred wave reductions: each entry stores the WaveActiveSum result and
        # shared memory buffer, but the shared write + barrier + cross-wave sum are
        # deferred until the result is actually consumed.  Consecutive tt.reduce ops
        # accumulate here and flush with a single barrier.
        self._pending_wave_reductions: list[dict] = []
        # --- Auto-unroll for wave reduction batching ---
        # When processing an unrolled loop copy, this is set to the copy index
        # (0..N-1).  _gen_tt_reduce uses it to create per-copy result SSA names
        # so that batched reductions don't overwrite each other.
        self._unroll_copy_idx: int | None = None
        self._in_unrolled_loop: bool = False  # True during entire auto-unrolled loop
        # --- rsqrt peephole: track scalar constant values ---
        self._scalar_const_value: dict[str, float] = {}  # SSA name -> numeric value
        # --- Integer div/rem pairing ---
        # Scoped per lexical op list so cached locals never escape their block.
        self._int_divrem_pairs: set[tuple[bool, str, str, str]] = set()
        self._int_divrem_cache: dict[
            tuple[bool, str, str, str], tuple[str, str | None]
        ] = {}

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _emit(self, line: str):
        self.lines.append("    " * self.indent + line)

    def _emit_barrier_direct(self):
        """Emit a threadgroup barrier and clear all pending/dirty state."""
        self._emit(self.emitter.barrier())
        self._barrier_pending = False
        self._barrier_loop_size = 0
        self._dirty_tiles.clear()

    def _fresh_var(self, prefix: str = "v") -> str:
        self.var_counter += 1
        return f"_{prefix}{self.var_counter}"

    def _get_val(self, ssa_name: str) -> SSAValue:
        if self._pending_wave_reductions and any(
                p['result_ssa'] == ssa_name for p in self._pending_wave_reductions):
            self._flush_wave_reductions()
        return self.ssa_map[ssa_name]

    def _get_expr(self, ssa_name: str) -> str:
        if self._pending_wave_reductions and any(
                p['result_ssa'] == ssa_name for p in self._pending_wave_reductions):
            self._flush_wave_reductions()
        return self.ssa_map[ssa_name].expr

    def _set_val(self, ssa_name: str, ttype: TType, expr: str):
        self.ssa_map[ssa_name] = SSAValue(name=ssa_name, ttype=ttype, expr=expr)

    def _is_dual(self, ssa_name: str) -> bool:
        return ssa_name in self._dual_exprs

    def _get_dual(self, ssa_name: str) -> tuple[str, str]:
        return self._dual_exprs[ssa_name]

    def _set_dual(self, ssa_name: str, row_expr: str, col_expr: str):
        self._dual_exprs[ssa_name] = (row_expr, col_expr)

    def _arg_name(self, arg: FuncArg) -> str:
        return f"arg{arg.index}"

    # --- Tile helpers ---

    def _is_tile(self, ssa_name: str) -> bool:
        return ssa_name in self._tiles or ssa_name in self._deferred_tiles

    def _materialize_deferred_tile(self, ssa_name: str):
        """Materialize a deferred dense constant tile into shared memory."""
        if ssa_name not in self._deferred_tiles:
            return
        shape, dtype, fill_val, target_type = self._deferred_tiles.pop(ssa_name)
        tile = self._alloc_tile(shape, dtype)
        self._emit_tile_loop(tile.total,
            f"{tile.shared_name}[_fi] = ({target_type}){fill_val};")
        self._register_tile(ssa_name, tile)

    def _get_tile(self, ssa_name: str) -> TileInfo:
        if ssa_name in self._deferred_tiles:
            self._materialize_deferred_tile(ssa_name)
        return self._tiles[ssa_name]

    def _needs_cooperative_loads(self) -> bool:
        """True when threads < block_size, so 1D loads need grid-stride shared memory."""
        if self._max_threads_override > 0:
            return self._max_threads_override < self.block_size
        return False  # Default: threads = min(block_size, MAX_THREADS) >= block_size

    def _alloc_tile(self, shape: list[int], dtype: str,
                    force_shared: bool = False) -> TileInfo:
        """Allocate shared memory tile, reusing from pool if possible.

        For 1D kernels where total == block_size (no 2D block shape), tiles are
        allocated as per-thread registers instead of shared memory.  Each thread
        owns exactly one element, eliminating barriers and shared memory overhead.

        Set force_shared=True for tiles that must be shared (e.g., cooperative
        loads that need inter-thread visibility via shared memory).
        """
        total = 1
        for d in shape:
            total *= d

        # Register tile: 1D tile with total == block_size (1:1 thread mapping).
        # Only for pure 1D kernels (no 2D dot), and total > 1 to avoid
        # converting scalar tiles whose broadcast semantics differ.
        # Disabled when num_threads < block_size because grid-stride loops
        # need shared memory for inter-thread data.
        if (not force_shared and not self._needs_cooperative_loads()
                and len(shape) == 1 and total > 1
                and total == self.block_size and self.block_shape is None):
            sn = self._fresh_var("reg")
            target_type = self.emitter.map_dtype(dtype)
            self._register_decls.append(f"{target_type} {sn};")
            self._register_tile_names.add(sn)
            return TileInfo(shared_name=sn, shape=list(shape), dtype=dtype,
                            is_register=True)

        key = (total, dtype)
        pool = self._tile_pool.get(key)
        if pool:
            sn = pool.pop()
        else:
            sn = self._fresh_var("tile")
            self._tile_decls.append(self.emitter.shared_memory_decl(sn, dtype, total))
            elem_size = 2 if dtype in ('f16', 'bf16', 'i16') else (1 if dtype == 'i8' else 4)
            self._tg_bytes_allocated += total * elem_size
        return TileInfo(shared_name=sn, shape=list(shape), dtype=dtype)

    def _register_tile(self, ssa_name: str, tile: TileInfo):
        """Register an SSA value as a shared memory tile (or per-thread register)."""
        self._tiles[ssa_name] = tile
        ttype = TType(dtype=tile.dtype, shape=tile.shape)
        if tile.is_register:
            # Register tile: per-thread variable, no array indexing
            self._set_val(ssa_name, ttype, tile.shared_name)
        else:
            tid = self.emitter.thread_id_expr()
            # Per-thread read expression (for element tid, single-pass only)
            self._set_val(ssa_name, ttype, f"{tile.shared_name}[(uint){tid}]")
        # Track which SSA names reference this backing store
        backing = self._real_backing(tile)
        if backing not in self._backing_refs:
            self._backing_refs[backing] = set()
        self._backing_refs[backing].add(ssa_name)

    def _real_backing(self, tile: TileInfo) -> str:
        """Get the actual backing shared_name (resolve views)."""
        if tile.transposed_from is not None:
            return tile.transposed_from.shared_name
        if tile.broadcast_src is not None:
            return tile.broadcast_src.shared_name
        return tile.shared_name

    def _release_tile(self, ssa_name: str):
        """Release an SSA name's tile back to the pool if no other refs remain."""
        # Drop deferred tiles without materializing
        self._deferred_tiles.pop(ssa_name, None)
        tile = self._tiles.get(ssa_name)
        if tile is None:
            return
        # Register tiles: just clean up refs, don't pool
        if tile.is_register:
            backing = self._real_backing(tile)
            refs = self._backing_refs.get(backing)
            if refs:
                refs.discard(ssa_name)
            return
        backing = self._real_backing(tile)
        if backing in self._pinned_backing:
            return  # Never free pinned tiles (iter_args, etc.)
        refs = self._backing_refs.get(backing)
        if refs:
            refs.discard(ssa_name)
            if not refs:
                # No more live references — return backing to pool
                orig = tile.transposed_from or tile.broadcast_src or tile
                key = (orig.total, orig.dtype)
                self._tile_pool.setdefault(key, []).append(backing)

    def _release_dead_operands(self, op: Op):
        """After processing op, release tiles for operands at their last use."""
        for operand in op.operands:
            if (operand in self._last_use and self._last_use[operand] <= self._current_op_idx
                    and (operand in self._tiles or operand in self._deferred_tiles)):
                self._release_tile(operand)

    def _try_reuse_in_place(self, operands: list[str], out_shape: list[int],
                             out_dtype: str) -> TileInfo | None:
        """Try to reuse an input tile's memory for the output (in-place op).

        For element-wise ops, if an input tile:
          - has the same total elements as the output
          - is at its last use (won't be read again)
          - owns its memory (not a transposed/broadcast view)
        then we can write the output directly to that tile's backing.
        """
        # Disable in-place reuse during auto-unrolled processing — the
        # _last_use map is stale (computed for the original body, not
        # the unrolled copies) so liveness checks would be wrong.
        if self._in_unrolled_loop:
            return None
        out_total = 1
        for d in out_shape:
            out_total *= d
        for operand in operands:
            if not self._is_tile(operand):
                continue
            if operand not in self._last_use:
                continue
            if self._last_use[operand] > self._current_op_idx:
                continue  # Still alive after this op
            tile = self._get_tile(operand)
            if tile.transposed_from is not None or tile.broadcast_src is not None:
                continue  # View — don't overwrite source
            if tile.total != out_total:
                continue  # Size mismatch
            if tile.dtype != out_dtype:
                continue  # Dtype mismatch (bool vs float etc.)
            # Check if other SSA values alias the same backing and are still alive.
            # This catches cases where tt.addptr aliases a pointer tile to this
            # tile's register — mutating the register would corrupt pointer loads.
            backing = self._real_backing(tile)
            refs = self._backing_refs.get(backing, set())
            alias_alive = False
            for ref in refs:
                if ref != operand and ref in self._last_use and self._last_use[ref] > self._current_op_idx:
                    alias_alive = True
                    break
            if alias_alive:
                continue  # Alias of this backing is still live
            # Can reuse! Create new TileInfo with same backing but output shape
            return TileInfo(shared_name=tile.shared_name, shape=list(out_shape),
                            dtype=out_dtype, is_register=tile.is_register)
        return None

    def _tile_read(self, tile: TileInfo, flat_idx: str) -> str:
        """Expression to read one element from a tile at flat index."""
        # Register tile: per-thread variable, no array indexing
        if tile.is_register and tile.broadcast_src is None and tile.transposed_from is None and tile.binop_view is None:
            expr = tile.shared_name
            if tile.pending_scale is not None:
                expr = f"({expr} * {tile.pending_scale})"
            return expr
        if tile.binop_view is not None:
            lhs, rhs, binop = tile.binop_view
            lhs_read = self._tile_read(lhs, flat_idx)
            rhs_read = self._tile_read(rhs, flat_idx)
            expr = f"({lhs_read} {binop} {rhs_read})"
            if tile.pending_scale is not None:
                expr = f"({expr} * {tile.pending_scale})"
            return expr
        if tile.transposed_from is not None:
            src = tile.transposed_from
            expr = f"{src.shared_name}[({flat_idx} % {tile.cols}u) * {src.cols}u + ({flat_idx} / {tile.cols}u)]"
        elif tile.broadcast_src is not None:
            src = tile.broadcast_src
            # Register source: just use the variable name (no array indexing)
            if src.is_register:
                expr = src.shared_name
            elif src.rank == 1 and tile.rank == 2:
                src_size = src.shape[0] if src.shape else 1
                if src_size == tile.cols or tile.rows == 1:
                    # src[N] broadcast to [M, N] (or reshape [N] → [1, N])
                    expr = f"{src.shared_name}[{flat_idx} % {tile.cols}u]"
                else:
                    # src[M] broadcast to [M, N]
                    expr = f"{src.shared_name}[{flat_idx} / {tile.cols}u]"
            elif src.rank == 2 and tile.rank == 2:
                if src.cols == 1:
                    expr = f"{src.shared_name}[{flat_idx} / {tile.cols}u]"
                elif src.rows == 1:
                    expr = f"{src.shared_name}[{flat_idx} % {tile.cols}u]"
                else:
                    expr = f"{src.shared_name}[{flat_idx}]"
            elif src.rank == 1 and tile.rank == 1:
                # 1D→1D broadcast: e.g., [1] → [N] reads element 0 for all
                if src.total == 1:
                    expr = f"{src.shared_name}[0]"
                else:
                    expr = f"{src.shared_name}[{flat_idx}]"
            else:
                expr = f"{src.shared_name}[{flat_idx}]"
        else:
            expr = f"{tile.shared_name}[{flat_idx}]"
        # Apply deferred scalar multiplication if present
        if tile.pending_scale is not None:
            expr = f"({expr} * {tile.pending_scale})"
        return expr

    def _tile_write(self, tile: TileInfo, flat_idx: str) -> str:
        """Expression to write one element to a tile at flat index."""
        if tile.is_register:
            return tile.shared_name
        return f"{tile.shared_name}[{flat_idx}]"

    def _tile_or_index_read(self, ssa: str, tile, flat_idx: str = "_fi") -> str:
        """Read expression for a tile operand, index expression, or scalar.

        Handles the case where an operand is not materialized as a tile but
        IS an index expression (from tl.arange / tt.make_range) that must
        vary with the loop variable.  Without this, grid-stride store loops
        (num_warps*32 < BLOCK_SIZE) write the same scalar value to every
        position instead of recomputing per element.
        """
        if tile:
            return self._tile_read(tile, flat_idx)
        if ssa in self._index_exprs:
            base, start = self._index_exprs[ssa]
            idx = f"(int){flat_idx}"
            if start != 0:
                idx = f"((int){flat_idx} + {start})"
            if base == "0":
                return idx
            return f"({base} + {idx})"
        return self._get_expr(ssa)

    def _tile_passes(self, tile: TileInfo) -> int:
        """Number of multi-pass iterations needed for a tile."""
        threads = min(self.block_size, self.MAX_THREADS)
        return max(1, (tile.total + threads - 1) // threads)

    def _flush_wave_reductions(self):
        """Flush all pending wave reductions with a single barrier.

        Multiple consecutive tt.reduce ops accumulate their WaveActiveSum
        results in _pending_wave_reductions.  When any result is consumed,
        this method emits all wave-leader writes, ONE barrier, and all
        cross-wave partial sums — batching N reductions into 1 barrier
        instead of N barriers.
        """
        if not self._pending_wave_reductions:
            return
        tid = self.emitter.thread_id_expr()
        lane_idx = self.emitter.wave_lane_index_expr()
        wid = self._fresh_var("wid")
        nw = self._fresh_var("nw")
        self._emit(f"uint {wid} = {self.emitter.wave_id_expr(tid)};")
        self._emit(f"uint {nw} = {self.emitter.wave_count_expr('_tg_size.x')};")

        # Emit all wave leader writes
        for p in self._pending_wave_reductions:
            self._emit(f"if ({lane_idx} == 0u) {{ {p['svar']}[{wid}] = {p['sg_r']}; }}")

        # ONE barrier for all pending reductions
        self._emit(self.emitter.barrier())

        # All threads compute cross-wave sums for each pending reduction.
        # Hoist the result variable to function scope so it's visible to
        # stores outside the loop where the reduction was computed.
        for p in self._pending_wave_reductions:
            merge = {'max': 'max', 'min': 'min'}.get(p['reduce_op'], None)
            var = self._fresh_var("red")
            rs = self._fresh_var("rs")
            tt = p['target_type']
            svar = p['svar']
            self._tile_decls.append(f"{tt} {var};")
            if merge:
                self._emit(f"{var} = {svar}[0];")
                self._emit(f"for (uint {rs} = 1u; {rs} < {nw}; {rs}++) {var} = {merge}({var}, {svar}[{rs}]);")
            else:
                self._emit(f"{var} = {svar}[0];")
                self._emit(f"for (uint {rs} = 1u; {rs} < {nw}; {rs}++) {var} += {svar}[{rs}];")
            self._set_val(p['result_ssa'], TType(dtype=p['dtype']), var)

        self._pending_wave_reductions.clear()

    # --- Auto-unroll helpers ---

    def _try_eval_int_expr(self, ssa_name: str) -> int | None:
        """Try to evaluate an SSA value as a compile-time integer constant."""
        # Check if the producing op is arith.constant
        op = self._op_map.get(ssa_name)
        if op and op.opname == 'arith.constant':
            val = op.attrs.get('value')
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            # Fallback: try parsing from SSA name pattern %cN_iM
            m = re.match(r'%c(-?\d+)', ssa_name)
            if m:
                return int(m.group(1))
        # Try parsing the expression directly
        expr = self._get_expr(ssa_name)
        m = re.match(r'\(int\)(-?\d+)', expr)
        if m:
            return int(m.group(1))
        try:
            return int(expr)
        except (ValueError, TypeError):
            return None

    def _detect_wave_reduce_unroll(self, op) -> int:
        """Detect if a for-loop benefits from auto-unrolling for wave reduction batching.

        Returns unroll factor (1 = no unroll, 4 = unroll 4x).
        Targets loops with exactly 1 scalar wave reduction per iteration.
        """
        if not self.emitter.supports_wave_reduce():
            return 1
        if not op.body_ops:
            return 1

        # Count scalar (wave-reducible) tt.reduce ops
        n_scalar_reduces = 0
        for bop in op.body_ops:
            if bop.opname != 'tt.reduce':
                continue
            # Scalar result = wave-reducible.  Tile result = axis reduce.
            rt = bop.result_types[0] if bop.result_types else None
            if rt and not rt.shape:
                n_scalar_reduces += 1

        if n_scalar_reduces != 1:
            return 1  # Only unroll when exactly 1 reduction per iteration

        # Don't unroll if the reduce's input depends on a modified iter_arg.
        # Find modified iter_args (yield value differs from arg name)
        modified_iter_args = set()
        if hasattr(op, 'yield_operands') and op.yield_operands:
            for arg_name, yield_name in zip(op.iter_arg_names, op.yield_operands):
                if yield_name != arg_name:
                    modified_iter_args.add(arg_name)
        if modified_iter_args:
            # Compute transitive deps from modified iter_args
            dep_ssas = set(modified_iter_args)
            changed = True
            while changed:
                changed = False
                for bop in op.body_ops:
                    if not bop.results:
                        continue
                    if any(o in dep_ssas for o in bop.operands):
                        for r in bop.results:
                            if r not in dep_ssas:
                                dep_ssas.add(r)
                                changed = True
            # If any reduce operand is in dep_ssas, don't unroll
            for bop in op.body_ops:
                if bop.opname == 'tt.reduce' and any(o in dep_ssas for o in bop.operands):
                    return 1

        # Step must be a compile-time constant
        step_val = self._try_eval_int_expr(op.loop_step)
        if step_val is None or step_val <= 0:
            return 1

        return 4

    def _detect_simple_loop_unroll(self, op) -> int:
        """Detect if a simple accumulator loop benefits from unrolling.

        Returns unroll factor (1 = no unroll, 4 = unroll 4x).
        Targets loops with loads and scalar accumulation, no wave reductions
        or tt.dot (matmul has its own optimization path).
        """
        if not op.body_ops:
            return 1

        # Step must be compile-time constant
        step_val = self._try_eval_int_expr(op.loop_step)
        if step_val is None or step_val <= 0:
            return 1

        has_load = False
        for bop in op.body_ops:
            if bop.opname == 'tt.load':
                has_load = True
            if bop.opname == 'tt.dot':
                return 1  # Matmul uses its own optimization
            if bop.opname == 'tt.reduce':
                # Any reduce (scalar or tile) — skip simple unroll
                return 1

        if not has_load:
            return 1

        return 4

    def _classify_body_for_unroll(self, op):
        """Classify loop body ops into Phase 1+2 (pre-flush) and Phase 3 (post-flush).

        Phase 1: ops independent of reduce results and modified iter_args
        Phase 2: tt.reduce ops (wave reduction)
        Phase 3: ops that transitively depend on reduce results or modified iter_args

        Returns: (phase12_ops, phase3_ops, reduce_result_ssas, cross_phase_ssas)
        """
        body_ops = op.body_ops or []

        # Find reduce result SSAs
        reduce_ssas = set()
        for bop in body_ops:
            if bop.opname == 'tt.reduce':
                reduce_ssas.update(bop.results or [])

        # Find modified iter_args (yield value differs from arg name)
        modified_iter_args = set()
        if hasattr(op, 'yield_operands') and op.yield_operands:
            for arg_name, yield_name in zip(op.iter_arg_names, op.yield_operands):
                if yield_name != arg_name:
                    modified_iter_args.add(arg_name)

        # Transitive closure: Phase 3 includes anything that depends on
        # reduce results or modified iter_args
        phase3_ssas = reduce_ssas | modified_iter_args
        changed = True
        while changed:
            changed = False
            for bop in body_ops:
                if not bop.results:
                    continue
                if all(r in phase3_ssas for r in bop.results):
                    continue
                if any(o in phase3_ssas for o in bop.operands):
                    for r in bop.results:
                        if r not in phase3_ssas:
                            phase3_ssas.add(r)
                            changed = True

        # Partition ops
        phase12_ops = []
        phase3_ops = []
        for bop in body_ops:
            if bop.opname == 'scf.yield':
                continue  # Yield handled separately
            if bop.opname == 'tt.reduce':
                # Only put reduce in phase12 if none of its operands are in phase3
                if any(o in phase3_ssas for o in bop.operands):
                    phase3_ops.append(bop)
                else:
                    phase12_ops.append(bop)  # Phase 2
            elif any(r in phase3_ssas for r in (bop.results or [])):
                phase3_ops.append(bop)
            elif not bop.results and any(o in phase3_ssas for o in bop.operands):
                phase3_ops.append(bop)  # Side-effecting op depending on Phase 3
            else:
                phase12_ops.append(bop)  # Phase 1

        # Cross-phase SSAs: produced by Phase 1/2, consumed by Phase 3
        # (reduce results handled separately via per-copy SSA naming)
        phase12_results = set()
        for bop in phase12_ops:
            phase12_results.update(bop.results or [])

        phase3_operands = set()
        for bop in phase3_ops:
            phase3_operands.update(bop.operands or [])

        cross_phase_ssas = (phase12_results & phase3_operands) - reduce_ssas

        return phase12_ops, phase3_ops, reduce_ssas, cross_phase_ssas

    @staticmethod
    def _int_divrem_key(op: Op, unsigned: bool) -> tuple[bool, str, str, str]:
        dtype = op.result_types[0].dtype if op.result_types else "i32"
        return unsigned, op.operands[0], op.operands[1], dtype

    def _enter_int_divrem_scope(self, ops: list[Op]):
        """Find matching integer div/rem operations in one lexical block."""
        previous = self._int_divrem_pairs, self._int_divrem_cache
        seen: dict[tuple[bool, str, str, str], int] = {}
        kinds = {
            'arith.divsi': (False, 1),
            'arith.remsi': (False, 2),
            'arith.divui': (True, 1),
            'arith.remui': (True, 2),
        }
        for op in ops:
            kind = kinds.get(op.opname)
            if kind is None or len(op.operands) < 2:
                continue
            unsigned, flag = kind
            key = self._int_divrem_key(op, unsigned)
            seen[key] = seen.get(key, 0) | flag
        self._int_divrem_pairs = {key for key, flags in seen.items() if flags == 3}
        self._int_divrem_cache = {}
        return previous

    def _leave_int_divrem_scope(self, previous):
        self._int_divrem_pairs, self._int_divrem_cache = previous

    def _gen_ops_no_liveness(self, ops):
        """Process a list of ops without liveness tracking (for unrolled copies)."""
        previous = self._enter_int_divrem_scope(ops)
        try:
            for bop in ops:
                self._current_op_idx = self._gen_counter[0]
                self._gen_counter[0] += 1
                self._gen_op(bop)
        finally:
            self._leave_int_divrem_scope(previous)

    def _process_loop_yields(self, op, iter_var_names, iter_tiles):
        """Process scf.for yield operands — update iter vars/tiles for next iteration."""
        # Check if any yield copies need shared memory writes
        needs_yield_copy = False
        for i, yield_name in enumerate(op.yield_operands):
            if i < len(iter_var_names):
                var, _ttype = iter_var_names[i]
                if var is not None or yield_name in self._reg_tiles:
                    pass
                else:
                    if self._is_tile(yield_name) and i < len(iter_tiles):
                        yield_tile = self._get_tile(yield_name)
                        orig_tile = iter_tiles[i]
                        if (yield_tile.shared_name != orig_tile.shared_name
                                and not (yield_tile.is_register and orig_tile.is_register)):
                            needs_yield_copy = True
                            break

        if needs_yield_copy:
            self._flush_barrier()

        # Update iter vars from yield
        for i, yield_name in enumerate(op.yield_operands):
            if i < len(iter_var_names):
                var, _ttype = iter_var_names[i]
                if var is not None:
                    yield_val = self._get_val(yield_name)
                    self._emit(f"{var} = {yield_val.expr};")
                elif yield_name in self._reg_tiles:
                    arg_name = op.iter_arg_names[i]
                    self._reg_tiles[arg_name] = self._reg_tiles[yield_name]
                else:
                    if self._is_tile(yield_name) and i < len(iter_tiles):
                        yield_tile = self._get_tile(yield_name)
                        orig_tile = iter_tiles[i]
                        if yield_tile.shared_name != orig_tile.shared_name:
                            self._emit_tile_copy(orig_tile, yield_tile)
                        arg_name = op.iter_arg_names[i]
                        self._tiles[arg_name] = orig_tile
                    elif i < len(iter_tiles) and iter_tiles[i] and iter_tiles[i].is_register:
                        # Scalar yield (e.g. from scf.if) updating a register tile
                        yield_val = self._get_val(yield_name)
                        self._emit(f"{iter_tiles[i].shared_name} = {yield_val.expr};")

    def _flush_barrier(self):
        """Emit a pending barrier if one exists."""
        # During auto-unroll Phase 1+2, don't flush wave reductions — they're
        # being accumulated across copies for batched flushing.
        if self._unroll_copy_idx is None:
            self._flush_wave_reductions()
        self._flush_fused_loops()
        if self._barrier_pending:
            self._emit(self.emitter.barrier())
            self._barrier_pending = False
            self._barrier_loop_size = 0
            self._dirty_tiles.clear()

    def _flush_barrier_if_dirty(self, *shared_names: str):
        """Emit barrier only if any of the given tiles were modified since last barrier.

        This allows skipping unnecessary barriers when the operation only reads
        tiles that haven't been modified recently.
        """
        self._flush_fused_loops()
        if self._barrier_pending and any(sn in self._dirty_tiles for sn in shared_names):
            self._emit(self.emitter.barrier())
            self._barrier_pending = False
            self._barrier_loop_size = 0
            self._dirty_tiles.clear()

    def _flush_fused_loops(self):
        """Emit any pending fused tile loop bodies as a single loop."""
        if not self._fused_loop_bodies:
            return
        tid = self.emitter.thread_id_expr()
        total = self._fused_loop_total
        body = " ".join(self._fused_loop_bodies)
        # Replace register tile array accesses with direct register access.
        # Body strings are constructed with tile.shared_name[_fi] patterns;
        # for register tiles, we strip the array index since they're scalars.
        for reg_name in self._register_tile_names:
            body = body.replace(f"{reg_name}[_fi]", reg_name)
            body = body.replace(f"{reg_name}[_flat]", reg_name)
        # If the body no longer references _fi (all tiles are register tiles),
        # emit directly without a loop wrapper.  The for-loop acts as an
        # instruction scheduling barrier that prevents the Metal shader
        # compiler from pipelining memory loads across accumulate operations.
        if '_fi' not in body and '_flat' not in body:
            self._emit(f"{{ {body} }}")
        else:
            self._emit(f"for (uint _fi = (uint){tid}; _fi < {total}u; _fi += _tg_size.x) {{ {body} }}")
        if self._fused_loop_barrier:
            self._barrier_pending = True
            self._barrier_loop_size = total
            # Track which tiles were dirtied by this fused loop
            self._dirty_tiles.update(self._fused_loop_dirty)
        self._fused_loop_bodies = []
        self._fused_loop_total = 0
        self._fused_loop_barrier = False
        self._fused_loop_dirty = set()

    def _emit_tile_loop(self, total: int, body: str, needs_barrier: bool = True):
        """Emit a grid-stride per-element loop.

        Inside the body, `_fi` is the flat element index.
        Uses _tg_size.x (runtime thread count) for portability across GPUs.

        Consecutive tile loops of the same total size are fused into a single
        loop to reduce loop overhead and shared memory round-trips.
        """
        # Flush pending fused loops if the new total differs
        if self._fused_loop_bodies and self._fused_loop_total != total:
            self._flush_fused_loops()

        # After flushing, check if a barrier is needed before this tile loop
        # (cross-thread reads via broadcast indexing require sync)
        if self._barrier_pending and total != self._barrier_loop_size:
            self._emit(self.emitter.barrier())
            self._barrier_pending = False
            self._barrier_loop_size = 0
            self._dirty_tiles.clear()

        self._fused_loop_bodies.append(body)
        self._fused_loop_total = total
        # Extract written tile name from body (pattern: "tilename[_fi] = ...")
        # Only set barrier flag and dirty tracking for shared memory writes.
        # Register tiles (no [_fi] pattern) don't need barriers.
        import re
        m = re.match(r'(\w+)\[', body)
        if m:
            tile_name = m.group(1)
            # Only track dirty/barrier for shared memory tiles, not register tiles
            if tile_name not in self._register_tile_names:
                if needs_barrier:
                    self._fused_loop_barrier = True
                if not hasattr(self, '_fused_loop_dirty'):
                    self._fused_loop_dirty = set()
                self._fused_loop_dirty.add(tile_name)

    # -----------------------------------------------------------------------
    # 2D load analysis: trace operand chain to extract cooperative load params
    # -----------------------------------------------------------------------

    def _analyze_2d_load(self, load_op: Op) -> dict | None:
        """Trace a 2D load's pointer chain to extract base_ptr, stride, offsets.

        Returns dict with: base_expr, stride_expr, rows, cols, row_offset, col_offset
        Or None if the pattern isn't recognized (falls back to per-thread load).

        Handles two patterns:
          Pattern 1: addptr(splat(base), addi(row*stride, col))
          Pattern 2: addptr(broadcast(addptr(splat(base), row*stride)), broadcast(col))
        """
        if not load_op.result_types:
            return None
        shape = load_op.result_types[0].shape
        if not shape or len(shape) != 2:
            return None
        rows, cols = shape

        ptr_ssa = load_op.operands[0]
        ptr_op = self._op_map.get(ptr_ssa)
        if not ptr_op or ptr_op.opname != 'tt.addptr':
            return None

        comp_a = ptr_op.operands[0]
        comp_b = ptr_op.operands[1]

        # Pattern 1: addptr(splat(base), addi(row*stride, col))
        result = self._try_flat_addptr_pattern(comp_a, comp_b, rows, cols)
        if result:
            return result

        # Pattern 2: addptr(broadcast(ptr_chain_with_stride), broadcast(col_offsets))
        # Common in expand_dims patterns (outer product, 2D indexing)
        for ptr_comp, idx_comp in [(comp_a, comp_b), (comp_b, comp_a)]:
            result = self._try_broadcast_addptr_pattern(ptr_comp, idx_comp, rows, cols)
            if result:
                return result

        return None

    def _try_flat_addptr_pattern(self, base_ssa: str, idx_ssa: str,
                                  rows: int, cols: int) -> dict | None:
        """Pattern 1: addptr(splat(base), addi(row*stride, col))."""
        base_op = self._op_map.get(base_ssa)
        if not base_op or base_op.opname != 'tt.splat':
            return None
        base_arg_ssa = base_op.operands[0]
        if base_arg_ssa not in self.ssa_map:
            return None
        base_expr = self._get_expr(base_arg_ssa)

        # For HLSL: if base_arg has a buffer name, use it and carry offset separately
        extra_offset = None
        if base_arg_ssa in self._buf_base:
            extra_offset = base_expr
            base_expr = self._buf_base[base_arg_ssa]

        idx_op = self._op_map.get(idx_ssa)
        if not idx_op or idx_op.opname != 'arith.addi':
            return None

        comp_a = idx_op.operands[0]
        comp_b = idx_op.operands[1]

        stride_a = self._find_stride_in_chain(comp_a)
        stride_b = self._find_stride_in_chain(comp_b)

        if stride_a and not stride_b:
            row_comp_ssa, stride_expr = stride_a
            col_comp_ssa = comp_b
        elif stride_b and not stride_a:
            row_comp_ssa, stride_expr = stride_b
            col_comp_ssa = comp_a
        else:
            return None

        row_offset = self._extract_tile_offset(row_comp_ssa)
        col_offset = self._extract_tile_offset(col_comp_ssa)

        result = {
            'base': base_expr, 'stride': stride_expr,
            'rows': rows, 'cols': cols,
            'row_offset': row_offset, 'col_offset': col_offset,
        }
        if extra_offset:
            result['extra_offset'] = extra_offset
        return result

    def _try_broadcast_addptr_pattern(self, ptr_comp: str, idx_comp: str,
                                       rows: int, cols: int) -> dict | None:
        """Pattern 2: addptr(broadcast(addptr(splat(base), row*stride)), broadcast(col)).

        Handles expand_dims+broadcast pointer patterns (outer product, 2D indexing).
        """
        # Trace through broadcast/expand_dims to find the inner addptr
        inner_ptr = self._trace_through_reshapes(ptr_comp)
        inner_op = self._op_map.get(inner_ptr) if inner_ptr else None
        if not inner_op or inner_op.opname != 'tt.addptr':
            return None

        # Inner addptr should be: addptr(splat(base_ptr), offset)
        splat_ssa = inner_op.operands[0]
        offset_ssa = inner_op.operands[1]

        splat_op = self._op_map.get(splat_ssa)
        if not splat_op or splat_op.opname != 'tt.splat':
            return None
        base_arg = splat_op.operands[0]
        if base_arg not in self.ssa_map:
            return None
        base_expr = self._get_expr(base_arg)

        # For HLSL: if base_arg has a buffer name (from _buf_base), use buffer name
        # as base and carry the scalar offset separately. This handles patterns like:
        #   addptr(Q_ptr, head_off) → splat → addptr(..., row*stride) → broadcast → addptr(..., col)
        # where base_arg is the scalar addptr result (integer offset), not the buffer.
        extra_offset = None
        if base_arg in self._buf_base:
            extra_offset = base_expr  # scalar offset (e.g., "_idx14")
            base_expr = self._buf_base[base_arg]  # buffer name (e.g., "arg0")

        # Find stride (muli with splatted func arg) in offset chain
        stride_info = self._find_stride_in_chain(offset_ssa)
        if not stride_info:
            return None
        row_ssa, stride_expr = stride_info

        row_offset = self._extract_tile_offset(row_ssa)

        # The column expression may include a stride multiply:
        #   broadcast(muli(offs_n, splat(stride_cn)))
        # Strip the stride multiply to get the raw offset, track stride separately.
        col_inner = self._trace_through_reshapes(idx_comp)
        col_stride_expr = None
        col_stride_info = self._find_stride_in_chain(col_inner) if col_inner else None
        if col_stride_info:
            col_ssa, col_stride_expr = col_stride_info
            col_offset = self._extract_tile_offset(col_ssa)
        else:
            col_offset = self._extract_tile_offset(idx_comp)

        result = {
            'base': base_expr, 'stride': stride_expr,
            'rows': rows, 'cols': cols,
            'row_offset': row_offset, 'col_offset': col_offset,
        }
        if col_stride_expr:
            result['col_stride'] = col_stride_expr
        if extra_offset:
            result['extra_offset'] = extra_offset
        return result

    def _trace_through_reshapes(self, ssa: str) -> str | None:
        """Trace through broadcast/expand_dims to find the inner SSA name."""
        op = self._op_map.get(ssa)
        while op and op.opname in ('tt.broadcast', 'tt.expand_dims'):
            ssa = op.operands[0]
            op = self._op_map.get(ssa)
        return ssa

    def _find_stride_in_chain(self, ssa_name: str) -> tuple[str, str] | None:
        """Look for a muli(_, stride_splat) in the operand chain.

        Returns (the SSA with the row indices before multiply, stride_expression)
        or None.
        """
        op = self._op_map.get(ssa_name)
        if not op:
            return None

        # Check broadcast -> look inside
        if op.opname in ('tt.broadcast', 'tt.expand_dims'):
            return self._find_stride_in_chain(op.operands[0])

        if op.opname == 'arith.muli':
            # One operand is the index, the other is the stride (via splat)
            for i in range(2):
                other = 1 - i
                stride_op = self._op_map.get(op.operands[i])
                if stride_op and stride_op.opname == 'tt.splat':
                    stride_arg = stride_op.operands[0]
                    if stride_arg in self.ssa_map:
                        return (op.operands[other], self._get_expr(stride_arg))
                # Also check if it's a direct scalar (not splatted) - for expand_dims chain
                if stride_op and stride_op.opname in ('tt.broadcast', 'tt.expand_dims'):
                    inner = self._find_stride_in_chain(op.operands[i])
                    if inner:
                        return inner
        return None

    def _extract_tile_offset(self, ssa_name: str) -> str:
        """Extract a compile-time offset expression for a tile dimension.

        Traces through broadcast/expand_dims to find the additive offset
        (e.g., pid_m * BM, or the loop induction variable).
        Returns a string expression.
        """
        op = self._op_map.get(ssa_name)
        if not op:
            # It's a function arg or loop variable - return its expression
            if ssa_name in self.ssa_map:
                return self._get_expr(ssa_name)
            return "0"

        if op.opname in ('tt.broadcast', 'tt.expand_dims'):
            return self._extract_tile_offset(op.operands[0])

        if op.opname == 'arith.addi':
            # addi(splat(pid_offset), make_range) → the offset is pid_offset
            # addi(addi(splat(A), range), splat(B)) → offset is A + B
            # First check if either operand is make_range (contributes 0)
            for i in range(2):
                inner_op = self._op_map.get(op.operands[i])
                if inner_op and inner_op.opname == 'tt.make_range':
                    # The other operand is the offset
                    other = op.operands[1 - i]
                    return self._extract_tile_offset(other)
            # Neither is make_range — collect all scalar offset terms
            terms = []
            for operand in op.operands:
                term = self._extract_tile_offset(operand)
                if term != "0":
                    terms.append(term)
            if not terms:
                return "0"
            if len(terms) == 1:
                return terms[0]
            return " + ".join(f"({t})" if '+' in t or '-' in t else t for t in terms)

        if op.opname == 'tt.splat':
            return self._get_expr(op.operands[0])

        if op.opname == 'tt.make_range':
            return "0"  # range starts at 0, offset is added elsewhere

        if op.opname == 'arith.muli' and ssa_name in self.ssa_map:
            # This shouldn't happen for the offset chain (stride multiply is separate)
            # But if we hit it, try to extract value
            return self._get_expr(ssa_name)

        if ssa_name in self.ssa_map:
            return self._get_expr(ssa_name)
        return "0"

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def generate(self, func_name: str, func_args: list[FuncArg], ops: list[Op]) -> str:
        """Generate complete target source from parsed TTIR."""
        self.func_args = func_args

        # Build op map for operand tracing + flat list (includes void ops like tt.store)
        self._op_map = build_op_map(ops)
        self._all_ops = flatten_ops(ops)
        self._deferred_cast_barriers = self._find_deferred_cast_barriers(ops)

        # Detect 2D block shape from tt.dot
        self.block_shape = self._detect_block_shape(ops)

        # Detect block size from make_range (1D kernels) or block shape (2D)
        if self.block_shape and len(self.block_shape) == 2:
            self.block_size = min(self.block_shape[0] * self.block_shape[1],
                                  self.MAX_THREADS)
        else:
            for op in ops:
                if op.opname == 'tt.make_range':
                    end_val = int(op.attrs.get('end', str(self.block_size)))
                    start_val = int(op.attrs.get('start', '0'))
                    self.block_size = end_val - start_val
                    break

        # Pre-scan for program_id dimensions
        for op in ops:
            if op.opname == 'tt.get_program_id':
                dim = 'x'
                m = re.match(r'\s*(x|y|z)\b', op.raw_text)
                if m:
                    dim = m.group(1)
                for attr_key in op.attrs:
                    if attr_key == 'axis':
                        dim_val = int(op.attrs[attr_key])
                        dim = ['x', 'y', 'z'][dim_val]
                self.pid_dims_used.add(dim)

        # Pre-compute tile liveness for memory reuse
        self._last_use = compute_liveness(ops)

        # Register function arguments
        for arg in func_args:
            self._set_val(arg.name, arg.ttype, self._arg_name(arg))

        # Generate kernel
        header = self.emitter.kernel_signature(func_name, func_args)

        # Emit legacy threadgroup memory declarations (matmul-optimized path)
        for decl in self.tg_memory_decls:
            self._emit(decl)

        # Generate body (tile declarations are collected in _tile_decls)
        self._gen_counter = [0]
        self._gen_ops_with_liveness(ops)
        self._flush_fused_loops()  # Flush any remaining fused loop bodies

        # Insert hoisted tile/register declarations at the top of the body
        all_decl_lines = []
        if self._tile_decls:
            all_decl_lines.extend(["    " + d for d in self._tile_decls])
        if self._register_decls:
            all_decl_lines.extend(["    " + d for d in self._register_decls])
        if all_decl_lines:
            self.lines = all_decl_lines + self.lines

        body = '\n'.join(self.lines)
        file_header = self.emitter.file_header()

        # Compute recommended dispatch thread count.
        if self._max_threads_override > 0:
            # Explicit override: use exactly this many threads (rounded to SIMD).
            self.recommended_threads = ((self._max_threads_override + 31) // 32) * 32
        elif self._reg_acc_min_sgs > 0:
            # Register accumulator needs enough SGs for full block coverage.
            # Use the larger of: min SGs needed, or block_size / 32 SGs.
            min_threads = self._reg_acc_min_sgs * 32
            # For high bps (≥3), cap total SGs to avoid register pressure cliff.
            if self._reg_acc_bps >= 3:
                max_threads = 704  # 22 SGs
            else:
                max_threads = 832
            self.recommended_threads = min(max_threads, max(min_threads, self.block_size))
        elif self._max_coop_load_size > self.block_size:
            # More threads than block_size speeds up cooperative loads.
            # Target ~5 elements per thread for the largest load, capped at 832.
            target = (self._max_coop_load_size + 4) // 5  # ~5 elements/thread
            target = ((target + 31) // 32) * 32  # round to SIMD group
            self.recommended_threads = min(max(target, self.block_size), 832)
        else:
            self.recommended_threads = min(self.block_size, self.MAX_THREADS)

        # For HLSL: groupshared must be at global scope (outside functions).
        # Extract all shared memory declarations from body and hoist them.
        global_decls = ""
        if self.emitter.requires_global_shared_memory():
            body_lines = body.split('\n')
            shared_lines = []
            remaining_lines = []
            for line in body_lines:
                stripped = line.strip()
                if stripped.startswith('groupshared '):
                    shared_lines.append(stripped)
                else:
                    remaining_lines.append(line)

            # Dead tile elimination: remove bool tiles that are written but
            # never read (common for mask tiles in cooperative load kernels).
            dead_tiles = self._find_dead_tiles(shared_lines, remaining_lines)
            if dead_tiles:
                shared_lines = [l for l in shared_lines
                                if not any(t in l for t in dead_tiles)]
                remaining_lines = self._remove_dead_tile_writes(
                    remaining_lines, dead_tiles)

            body = '\n'.join(remaining_lines)
            if shared_lines:
                global_decls = '\n'.join(shared_lines) + '\n\n'

        return f"{file_header}\n{global_decls}{header}\n{body}\n}}\n"

    @classmethod
    def _find_deferred_cast_barriers(cls, ops: list[Op]) -> set[str]:
        """Find cast writes synchronized by a cooperative load before use.

        For example, FA2 writes a converted P tile, loads V into a different
        tile, and then consumes both in a dot. The load's barrier makes both
        writes visible, so a separate barrier after the conversion is wasted.
        """
        deferred: set[str] = set()
        for index, op in enumerate(ops):
            if op.opname == 'arith.truncf' and len(op.results) == 1:
                result = op.results[0]
                saw_2d_load = False
                for later in ops[index + 1:]:
                    if later.opname == 'tt.load' and any(
                            result_type.rank == 2
                            for result_type in later.result_types):
                        saw_2d_load = True
                    if result in later.operands:
                        if saw_2d_load and later.opname == 'tt.dot':
                            deferred.add(result)
                        break
                    if later.body_ops or later.else_ops:
                        break
            if op.body_ops:
                deferred.update(cls._find_deferred_cast_barriers(op.body_ops))
            if op.else_ops:
                deferred.update(cls._find_deferred_cast_barriers(op.else_ops))
        return deferred

    @staticmethod
    def _find_dead_tiles(shared_lines: list[str], body_lines: list[str]) -> set[str]:
        """Find groupshared bool tiles that are written but never read."""
        import re
        # Collect all bool tile names from declarations
        bool_tiles = set()
        for line in shared_lines:
            m = re.match(r'groupshared\s+bool\s+(_tile\d+)\[', line)
            if m:
                bool_tiles.add(m.group(1))

        if not bool_tiles:
            return set()

        # Check which tiles are read (appear as RHS expression, not just LHS assignment)
        dead = set(bool_tiles)
        for line in body_lines:
            stripped = line.strip()
            for tile in list(dead):
                # Check if tile appears on RHS: in any expression context except
                # pure assignment LHS like "_tileXX[idx] = expr;"
                pat = re.escape(tile) + r'\['
                matches = list(re.finditer(pat, stripped))
                if not matches:
                    continue
                # If the ONLY occurrence is at the start of an assignment, it's a write
                if (stripped.startswith((f'{tile}[', 'if ('))
                        and f'{tile}[' in stripped and '= ' in stripped):
                    # Could be "if (cond) { _tileXX[...] = ...; }" — check if it's
                    # a write-only pattern
                    # Check if tile also appears on the RHS of the assignment
                    assign_idx = stripped.index('= ')
                    rhs = stripped[assign_idx + 2:]
                    if tile in rhs:
                        dead.discard(tile)
                    continue
                # Tile appears in some other context — it's read
                dead.discard(tile)

        return dead

    @staticmethod
    def _remove_dead_tile_writes(lines: list[str], dead_tiles: set[str]) -> list[str]:
        """Remove writes to dead tiles and their following barriers."""
        result = []
        i = 0
        barrier_str = 'GroupMemoryBarrierWithGroupSync'
        while i < len(lines):
            stripped = lines[i].strip()
            # Check if this line writes to a dead tile
            is_dead_write = False
            for tile in dead_tiles:
                if tile + '[' in stripped:
                    is_dead_write = True
                    break

            if is_dead_write:
                # Skip this write line
                i += 1
                # If the next line is a barrier, check if it can be removed.
                # It's safe to remove if the previous live line is also a barrier
                # (meaning no other writes happened between barriers).
                if i < len(lines) and barrier_str in lines[i].strip():
                    # Check if previous live line is also a barrier
                    prev_is_barrier = False
                    for j in range(len(result) - 1, -1, -1):
                        pstripped = result[j].strip()
                        if pstripped:
                            prev_is_barrier = barrier_str in pstripped
                            break
                    if prev_is_barrier:
                        i += 1  # Skip redundant barrier
                continue

            result.append(lines[i])
            i += 1

        return result

    def _gen_ops_with_liveness(self, ops: list[Op]):
        """Generate ops while tracking liveness for tile reuse."""
        previous = self._enter_int_divrem_scope(ops)
        try:
            i = 0
            while i < len(ops):
                self._current_op_idx = self._gen_counter[0]
                self._gen_counter[0] += 1
                # Try to fuse online softmax pattern (saves ~4 barriers)
                consumed = self._try_fuse_online_softmax(ops, i)
                if consumed > 0:
                    for j in range(consumed):
                        self._release_dead_operands(ops[i + j])
                        if j > 0:
                            self._gen_counter[0] += 1
                    i += consumed
                    continue
                self._gen_op(ops[i])
                self._release_dead_operands(ops[i])
                i += 1
        finally:
            self._leave_int_divrem_scope(previous)

    def _try_fuse_online_softmax(self, ops: list[Op], idx: int) -> int:
        """Try to fuse the online softmax pattern into a single SIMD pass.

        Pattern (11 ops):
          0: tt.reduce {max, axis=1} QK_scaled → row_max [BM]
          1: arith.maximumf (m_i, row_max) → m_new [BM]
          2: arith.subf (m_i, m_new) → diff_m [BM]
          3: math.exp (diff_m) → alpha [BM]
          4: tt.expand_dims m_new → m_new_exp [BM,1]
          5: tt.broadcast m_new_exp → m_new_bc [BM,BN]
          6: arith.subf (QK_scaled, m_new_bc) → qk_shifted [BM,BN]
          7: math.exp (qk_shifted) → P [BM,BN]
          8: tt.reduce {add, axis=1} P → row_sum [BM]
          9: arith.mulf (l_i, alpha) → l_scaled [BM]
          10: arith.addf (l_scaled, row_sum) → l_new [BM]

        When BN ≤ 32 (SIMD width), fuses all into a single per-row pass
        that eliminates ~4 barriers.  Returns number of ops consumed (11 or 0).
        """
        if idx + 10 >= len(ops) or not self.emitter.supports_simd_matrix():
            return 0

        # --- Match pattern ---
        op0 = ops[idx]
        if op0.opname != 'tt.reduce':
            return 0
        if op0.attrs.get('reduce_op', '') != 'max' or int(op0.attrs.get('axis', -1)) != 1:
            return 0
        qk_name = op0.operands[0]
        if not self._is_tile(qk_name):
            return 0
        qk_tile = self._get_tile(qk_name)
        if qk_tile.rank != 2 or qk_tile.shape[1] > 32:
            return 0
        BM, BN = qk_tile.shape
        row_max_name = op0.results[0]

        # Op 1: arith.maximumf (m_i, row_max) → m_new
        op1 = ops[idx + 1]
        if op1.opname != 'arith.maximumf':
            return 0
        # Commutative: either order
        if row_max_name in op1.operands:
            m_i_name = next(x for x in op1.operands if x != row_max_name)
        else:
            return 0
        m_new_name = op1.results[0]

        # Op 2: arith.subf (m_i, m_new)
        op2 = ops[idx + 2]
        if op2.opname != 'arith.subf' or op2.operands[0] != m_i_name or op2.operands[1] != m_new_name:
            return 0
        diff_m_name = op2.results[0]

        # Op 3: math.exp (diff_m) → alpha
        op3 = ops[idx + 3]
        if op3.opname != 'math.exp' or op3.operands[0] != diff_m_name:
            return 0
        alpha_name = op3.results[0]

        # Op 4: tt.expand_dims m_new
        op4 = ops[idx + 4]
        if op4.opname != 'tt.expand_dims' or op4.operands[0] != m_new_name:
            return 0
        m_exp_name = op4.results[0]

        # Op 5: tt.broadcast m_new_exp
        op5 = ops[idx + 5]
        if op5.opname != 'tt.broadcast' or op5.operands[0] != m_exp_name:
            return 0
        m_bc_name = op5.results[0]

        # Op 6: arith.subf (QK_scaled, m_new_bc)
        op6 = ops[idx + 6]
        if op6.opname != 'arith.subf' or op6.operands[0] != qk_name or op6.operands[1] != m_bc_name:
            return 0
        qk_shifted_name = op6.results[0]

        # Op 7: math.exp (qk_shifted) → P
        op7 = ops[idx + 7]
        if op7.opname != 'math.exp' or op7.operands[0] != qk_shifted_name:
            return 0
        P_name = op7.results[0]

        # Op 8: tt.reduce {add, axis=1} P → row_sum
        op8 = ops[idx + 8]
        if op8.opname != 'tt.reduce':
            return 0
        if op8.attrs.get('reduce_op', '') not in ('add', 'sum') or int(op8.attrs.get('axis', -1)) != 1:
            return 0
        if op8.operands[0] != P_name:
            return 0
        row_sum_name = op8.results[0]

        # Op 9: arith.mulf (l_i, alpha)
        op9 = ops[idx + 9]
        if op9.opname != 'arith.mulf':
            return 0
        if alpha_name in op9.operands:
            l_i_name = next(x for x in op9.operands if x != alpha_name)
        else:
            return 0
        l_scaled_name = op9.results[0]

        # Op 10: arith.addf (l_scaled, row_sum)
        op10 = ops[idx + 10]
        if op10.opname != 'arith.addf':
            return 0
        if not (l_scaled_name in op10.operands and row_sum_name in op10.operands):
            return 0
        l_new_name = op10.results[0]

        # --- Pattern matched! Emit fused code. ---
        self._flush_barrier()  # Ensure QK data is visible

        dtype = qk_tile.dtype
        metal_type = self.emitter.map_dtype(dtype)
        tid = self.emitter.thread_id_expr()

        # Get the scale expression from pending_scale on the QK tile
        scale_expr = qk_tile.pending_scale

        # Get input tiles: m_i and l_i
        m_i_tile = self._get_tile(m_i_name) if self._is_tile(m_i_name) else None
        l_i_tile = self._get_tile(l_i_name) if self._is_tile(l_i_name) else None
        if m_i_tile is None or l_i_tile is None:
            return 0  # Can't fuse without 1D tiles

        # Allocate output tiles
        # P: reuse QK tile's backing (overwrite in place)
        p_tile = TileInfo(shared_name=qk_tile.shared_name, shape=[BM, BN], dtype=dtype)
        # alpha: allocate [BM]
        alpha_tile = self._alloc_tile([BM], dtype)
        # m_new: write directly to m_i tile (avoids yield copy)
        m_new_tile = TileInfo(shared_name=m_i_tile.shared_name, shape=[BM], dtype=dtype)
        # l_new: write directly to l_i tile (avoids yield copy)
        l_new_tile = TileInfo(shared_name=l_i_tile.shared_name, shape=[BM], dtype=dtype)

        # Emit the fused SIMD softmax pass
        sg_var = self._fresh_var("fsg")
        lane_var = self._fresh_var("flane")
        row_var = self._fresh_var("frow")
        self._emit(f"uint {sg_var} = (uint){tid} / 32u;")
        self._emit(f"uint {lane_var} = (uint){tid} % 32u;")
        self._emit(f"for (uint {row_var} = {sg_var}; {row_var} < {BM}u; {row_var} += _tg_size.x / 32u) {{")

        # Step 1: Read QK[row, lane], apply scale, compute row max
        qk_read = f"{qk_tile.shared_name}[{row_var} * {BN}u + {lane_var}]"
        if scale_expr:
            qk_read = f"({qk_read} * {scale_expr})"
        if BN == 32:
            self._emit(f"    {metal_type} _qk = {qk_read};")
        else:
            self._emit(f"    {metal_type} _qk = ({lane_var} < {BN}u) ? {qk_read} : (-({metal_type})HUGE_VALF);")
        self._emit(f"    {metal_type} _rmax = simd_max(_qk);")

        # Step 2: m_new = max(m_old, row_max), alpha = exp(m_old - m_new)
        self._emit(f"    {metal_type} _m_old = {m_i_tile.shared_name}[{row_var}];")
        self._emit(f"    {metal_type} _m_new = max(_m_old, _rmax);")
        self._emit(f"    {metal_type} _alpha = exp(_m_old - _m_new);")

        # Step 3: P = exp(qk - m_new), write to shared
        if BN == 32:
            self._emit(f"    {metal_type} _p = exp(_qk - _m_new);")
            self._emit(f"    {qk_tile.shared_name}[{row_var} * {BN}u + {lane_var}] = _p;")
            self._emit(f"    {metal_type} _rsum = simd_sum(_p);")
        else:
            self._emit(f"    {metal_type} _p = ({lane_var} < {BN}u) ? exp(_qk - _m_new) : ({metal_type})0;")
            self._emit(f"    if ({lane_var} < {BN}u) {qk_tile.shared_name}[{row_var} * {BN}u + {lane_var}] = _p;")
            self._emit(f"    {metal_type} _rsum = simd_sum(_p);")

        # Step 4: l_new = l_old * alpha + row_sum; write all outputs
        self._emit(f"    if ({lane_var} == 0u) {{")
        self._emit(f"        {metal_type} _l_old = {l_i_tile.shared_name}[{row_var}];")
        self._emit(f"        {l_new_tile.shared_name}[{row_var}] = _l_old * _alpha + _rsum;")
        self._emit(f"        {m_new_tile.shared_name}[{row_var}] = _m_new;")
        self._emit(f"        {alpha_tile.shared_name}[{row_var}] = _alpha;")
        self._emit("    }")
        self._emit("}")
        self._emit_barrier_direct()

        # Register all output SSA values
        self._register_tile(P_name, p_tile)
        self._register_tile(alpha_name, alpha_tile)
        self._register_tile(m_new_name, m_new_tile)
        self._register_tile(l_new_name, l_new_tile)

        # Register intermediate SSA values that won't be used downstream
        # but need entries so _release_dead_operands doesn't error.
        # row_max: dummy 1D value (computed inline, not stored separately)
        self._set_val(row_max_name, TType(dtype=dtype, shape=[BM]),
                      f"{m_new_tile.shared_name}[(uint){self.emitter.thread_id_expr()}]")
        self._tiles[row_max_name] = m_new_tile
        self._backing_refs.setdefault(m_new_tile.shared_name, set()).add(row_max_name)

        # diff_m, qk_shifted, l_scaled: register as dummy vals
        for name in [diff_m_name, qk_shifted_name, l_scaled_name]:
            self._set_val(name, TType(dtype=dtype), "0 /*fused*/")

        # m_new_exp and m_new_bc: register as view tiles (so expand_dims/broadcast
        # operand release works)
        m_exp_tile = TileInfo(shared_name=m_new_tile.shared_name,
                              shape=[BM, 1], dtype=dtype,
                              broadcast_src=m_new_tile)
        self._tiles[m_exp_name] = m_exp_tile
        self._set_val(m_exp_name, TType(dtype=dtype, shape=[BM, 1]),
                      f"{m_new_tile.shared_name}[0]")
        self._backing_refs.setdefault(m_new_tile.shared_name, set()).add(m_exp_name)

        m_bc_tile = TileInfo(shared_name=m_new_tile.shared_name,
                             shape=[BM, BN], dtype=dtype,
                             broadcast_src=m_exp_tile)
        self._tiles[m_bc_name] = m_bc_tile
        self._set_val(m_bc_name, TType(dtype=dtype, shape=[BM, BN]),
                      f"{m_new_tile.shared_name}[0]")
        self._backing_refs.setdefault(m_new_tile.shared_name, set()).add(m_bc_name)

        # row_sum: computed inline in fused pass, register as dummy
        self._set_val(row_sum_name, TType(dtype=dtype, shape=[BM]),
                      f"{l_new_tile.shared_name}[(uint){self.emitter.thread_id_expr()}]")
        self._tiles[row_sum_name] = l_new_tile
        self._backing_refs.setdefault(l_new_tile.shared_name, set()).add(row_sum_name)

        return 11

    def _detect_block_shape(self, ops: list[Op]) -> list[int] | None:
        # Prefer tt.dot result shape (matmul kernels)
        for op in ops:
            if op.opname == 'tt.dot' and op.result_types:
                shape = op.result_types[0].shape
                if shape and len(shape) == 2:
                    return shape
            if op.body_ops:
                result = self._detect_block_shape(op.body_ops)
                if result:
                    return result
        # Fallback: detect from 2D tt.load (tile kernels without tt.dot)
        best = None
        best_total = 0
        for op in ops:
            if op.opname == 'tt.load' and op.result_types:
                shape = op.result_types[0].shape
                if shape and len(shape) == 2:
                    total = shape[0] * shape[1]
                    if total > best_total:
                        best = shape
                        best_total = total
            if op.body_ops:
                for inner_op in op.body_ops:
                    if inner_op.opname == 'tt.load' and inner_op.result_types:
                        shape = inner_op.result_types[0].shape
                        if shape and len(shape) == 2:
                            total = shape[0] * shape[1]
                            if total > best_total:
                                best = shape
                                best_total = total
        return best

    def _gen_op(self, op: Op):
        # Skip ops whose results were fused into the matmul store
        if self._fused_scale_skip and any(r in self._fused_scale_skip for r in op.results):
            return
        # Skip ops that only feed into the already-emitted dot store
        # (truncf, convert_layout between loop result and tt.store)
        if (self._dot_consumed_ssa and op.operands
                and all(o in self._dot_consumed_ssa for o in op.operands)
                and op.opname in ('arith.truncf', 'arith.extf', 'ttg.convert_layout')):
            for r in op.results:
                self._dot_consumed_ssa.add(r)
                self._set_val(r, op.result_types[0] if op.result_types else TType(dtype='i32'), "0 /*dot_consumed*/")
            return
        handler = getattr(self, f'_gen_{op.opname.replace(".", "_")}', None)
        if handler:
            handler(op)
        else:
            raise UnsupportedOperationError(op)

    # -----------------------------------------------------------------------
    # TTGIR layout ops
    # -----------------------------------------------------------------------

    def _propagate_alias(self, src: str, dst: str, op: Op):
        """Propagate all value/tile/type/reg_acc info from src SSA to dst SSA.

        Used by TTGIR identity ops (convert_layout, local_alloc, local_load)
        that don't change data, only metadata.
        """
        if src in self._tiles:
            self._tiles[dst] = self._tiles[src]
        if src in self._deferred_tiles:
            self._deferred_tiles[dst] = self._deferred_tiles[src]
        if src in self._reg_tiles:
            self._reg_tiles[dst] = self._reg_tiles[src]
            if src in self._reg_acc_cast_dtype:
                self._reg_acc_cast_dtype[dst] = self._reg_acc_cast_dtype[src]
        if src in self.ssa_map:
            self._set_val(dst, self.ssa_map[src].ttype, self.ssa_map[src].expr)
        elif op.result_types:
            self._set_val(dst, op.result_types[0], f"/* alias {src} */")
        if src in self._dual_exprs:
            self._dual_exprs[dst] = self._dual_exprs[src]
        if src in self._op_map:
            self._op_map[dst] = self._op_map[src]
        if src in self._tiles:
            backing = self._real_backing(self._tiles[src])
            self._backing_refs.setdefault(backing, set()).add(dst)

    def _gen_ttg_convert_layout(self, op: Op):
        """Handle ttg.convert_layout: layout re-encoding (identity in our lowering).

        In TTGIR, this converts a tensor between different thread-to-element
        mappings (e.g. blocked -> dot_op). Since we handle thread mapping
        ourselves, this is just an alias — propagate all value/tile/type info.
        """
        if op.operands and op.results:
            src = op.operands[0]
            dst = op.results[0]
            self._propagate_alias(src, dst, op)

    def _gen_ttg_local_alloc(self, op: Op):
        """Handle ttg.local_alloc: tensor → shared memory descriptor (identity alias).

        In TTGIR, this explicitly allocates a tensor in shared memory.
        In our lowering, 2D loads already produce shared memory tiles,
        so this is just an alias propagation.
        """
        if op.operands and op.results:
            self._propagate_alias(op.operands[0], op.results[0], op)

    def _gen_ttg_memdesc_trans(self, op: Op):
        """Handle ttg.memdesc_trans: transpose a shared memory descriptor.

        Same semantics as tt.trans — creates a transposed view of the tile.
        """
        src = op.operands[0]
        result = op.results[0]

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            if src_tile.rank == 2:
                new_shape = [src_tile.shape[1], src_tile.shape[0]]
                trans_tile = TileInfo(
                    shared_name=src_tile.shared_name,
                    shape=new_shape,
                    dtype=src_tile.dtype,
                    transposed_from=src_tile,
                )
                self._tiles[result] = trans_tile
                src_val = self._get_val(src)
                self._set_val(result, TType(dtype=src_tile.dtype, shape=new_shape),
                              src_val.expr)
                backing = self._real_backing(trans_tile)
                self._backing_refs.setdefault(backing, set()).add(result)
                return

        # Scalar/non-tile fallback
        if src in self.ssa_map:
            result_type = op.result_types[0] if op.result_types else self.ssa_map[src].ttype
            self._set_val(result, result_type, self.ssa_map[src].expr)

    def _gen_ttg_local_load(self, op: Op):
        """Handle ttg.local_load: shared memory descriptor → register tensor (identity alias).

        In TTGIR, this loads data from shared memory into registers with a
        specific layout encoding. In our lowering, we keep data in shared
        memory tiles, so this is just an alias propagation.
        """
        if op.operands and op.results:
            self._propagate_alias(op.operands[0], op.results[0], op)

    # -----------------------------------------------------------------------
    # Control flow ops
    # -----------------------------------------------------------------------

    def _gen_cf_cond_br(self, op: Op):
        """Handle cf.cond_br: early-return guard pattern.

        In TTIR, this appears as:
            cf.cond_br %cond, ^bb_return, ^bb_continue
        where ^bb_return contains just tt.return. We emit: if (cond) return;
        """
        if op.operands:
            cond = self._get_expr(op.operands[0])
            self._emit(f"if ({cond}) {{ return; }}")

    # -----------------------------------------------------------------------
    # Triton ops
    # -----------------------------------------------------------------------

    def _gen_tt_get_program_id(self, op: Op):
        dim = 'x'
        m = re.match(r'\s*(x|y|z)\b', op.raw_text)
        if m:
            dim = m.group(1)
        for k, v in op.attrs.items():
            if k == 'axis':
                dim = ['x', 'y', 'z'][int(v)]
        result = op.results[0]
        var = self._fresh_var("pid")
        self._emit(f"int {var} = (int){self.emitter.threadgroup_id_expr(dim)};")
        self._set_val(result, TType(dtype='i32'), var)
        self._pid_exprs[dim] = var

    def _gen_tt_get_num_programs(self, op: Op):
        """Handle tt.get_num_programs: returns the grid size along an axis."""
        dim = 'x'
        m = re.match(r'\s*(x|y|z)\b', op.raw_text)
        if m:
            dim = m.group(1)
        for k, v in op.attrs.items():
            if k == 'axis':
                dim = ['x', 'y', 'z'][int(v)]
        result = op.results[0]
        var = self._fresh_var("nprg")
        # Metal doesn't expose grid dimensions directly in the kernel.
        # We pass them as metadata or compute from threadgroup position.
        # For now, use a special expression that the emitter provides.
        self._emit(f"int {var} = (int){self.emitter.grid_dim_expr(dim)};")
        self._set_val(result, TType(dtype='i32'), var)

    def _gen_tt_make_range(self, op: Op):
        start = int(op.attrs.get('start', '0'))
        end = int(op.attrs.get('end', str(self.block_size)))
        size = end - start
        result = op.results[0]
        ttype = TType(dtype='i32', shape=[size])
        tid = self.emitter.thread_id_expr()

        if self.block_shape and len(self.block_shape) == 2:
            _BM, BN = self.block_shape
            row_var = self._fresh_var("rng_row")
            col_var = self._fresh_var("rng_col")
            if start == 0:
                self._emit(f"int {row_var} = (int){tid} / {BN};")
                self._emit(f"int {col_var} = (int){tid} % {BN};")
            else:
                self._emit(f"int {row_var} = (int){tid} / {BN} + {start};")
                self._emit(f"int {col_var} = (int){tid} % {BN} + {start};")
            self._set_dual(result, row_var, col_var)
            # Use the correct dimension as the primary expression:
            # - If range size matches BN (column dim), use col_var
            # - If range size matches BM (row dim), use row_var
            primary = col_var if size == BN else row_var
            self._set_val(result, ttype, primary)
            # Track as index expression: value at position i = 0 + (i + start)
            self._index_exprs[result] = ("0", start)
        else:
            var = self._fresh_var("rng")
            if start == 0:
                self._emit(f"int {var} = (int){tid};")
            else:
                self._emit(f"int {var} = (int){tid} + {start};")
            self._set_val(result, ttype, var)

    def _gen_tt_splat(self, op: Op):
        src = op.operands[0]
        result = op.results[0]
        src_val = self._get_val(src)
        result_type = op.result_types[0] if op.result_types else TType(
            dtype=src_val.ttype.dtype, shape=[self.block_size])

        # If source is a tile and result is higher-rank (broadcasting via splat)
        if self._is_tile(src) and result_type.shape and len(result_type.shape) > 0:
            src_tile = self._get_tile(src)
            # Create a broadcast view
            out_tile = TileInfo(
                shared_name=src_tile.shared_name,
                shape=list(result_type.shape),
                dtype=src_tile.dtype,
                broadcast_src=src_tile,
            )
            self._tiles[result] = out_tile
            self._set_val(result, result_type, src_val.expr)
            # Track backing ref so source tile isn't freed while this view lives
            backing = self._real_backing(out_tile)
            self._backing_refs.setdefault(backing, set()).add(result)
            return

        self._set_val(result, result_type, src_val.expr)
        # Propagate buffer base tracking for HLSL pointer types
        if src in self._buf_base:
            self._buf_base[result] = self._buf_base[src]

    def _gen_tt_broadcast(self, op: Op):
        src = op.operands[0]
        result = op.results[0]
        src_val = self._get_val(src)
        result_type = op.result_types[0] if op.result_types else src_val.ttype

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            # Create broadcast tile view
            out_shape = list(result_type.shape) if result_type.shape else src_tile.shape
            out_tile = TileInfo(
                shared_name=src_tile.shared_name,
                shape=out_shape,
                dtype=src_tile.dtype,
                broadcast_src=src_tile if out_shape != src_tile.shape else None,
            )
            self._tiles[result] = out_tile
            self._set_val(result, result_type, src_val.expr)
            # Propagate buffer base tracking for HLSL
            if src in self._buf_base:
                self._buf_base[result] = self._buf_base[src]
            # Track backing ref so source tile isn't freed while this view lives
            backing = self._real_backing(out_tile)
            self._backing_refs.setdefault(backing, set()).add(result)
            return

        # Create virtual tile expression when source has tracked index expression + expand_axis
        # This handles: make_range → addi(scalar) → expand_dims → broadcast
        # Virtual tiles compute per-element values on-the-fly without shared memory
        result_shape = result_type.shape if result_type else None
        if (src in self._expand_axis and src in self._index_exprs
                and result_shape and len(result_shape) == 2):
            varying_axis, _ = self._expand_axis[src]
            base_expr, start = self._index_exprs[src]
            rows, cols = result_shape
            dtype = src_val.ttype.dtype or 'i32'
            # Build per-element expression using _fi
            if varying_axis == 0:  # varies by row
                idx_expr = f"(int)(_fi / {cols}u)"
            else:  # varies by column
                idx_expr = f"(int)(_fi % {cols}u)"
            if start != 0:
                idx_expr = f"({idx_expr} + {start})"
            if base_expr == "0":
                val_expr = idx_expr
            else:
                val_expr = f"({base_expr} + {idx_expr})"
            self._virtual_tiles[result] = (val_expr, [rows, cols], dtype)
            self._set_val(result, result_type, src_val.expr)
            return

        self._set_val(result, result_type, src_val.expr)

    def _gen_tt_expand_dims(self, op: Op):
        src = op.operands[0]
        result = op.results[0]
        src_val = self._get_val(src)
        result_type = op.result_types[0] if op.result_types else src_val.ttype

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            out_shape = list(result_type.shape) if result_type.shape else src_tile.shape
            out_tile = TileInfo(
                shared_name=src_tile.shared_name,
                shape=out_shape,
                dtype=src_tile.dtype,
                broadcast_src=src_tile,
            )
            self._tiles[result] = out_tile
            self._set_val(result, result_type, src_val.expr)
            # Track backing ref so source tile isn't freed while this view lives
            backing = self._real_backing(out_tile)
            self._backing_refs.setdefault(backing, set()).add(result)
            return

        if self._is_dual(src):
            axis = int(op.attrs.get('axis', '0'))
            row_expr, col_expr = self._get_dual(src)
            resolved = row_expr if axis == 1 else col_expr
            self._set_val(result, result_type, resolved)
            # Track expand axis + propagate index expr for broadcast materialization
            if src in self._index_exprs:
                # axis=1 means [N] → [N,1]: values vary by row (axis 0 of 2D)
                # axis=0 means [N] → [1,N]: values vary by column (axis 1 of 2D)
                varying_axis = 0 if axis == 1 else 1  # which 2D axis the values vary along
                result_shape = result_type.shape if result_type else None
                n_cols = result_shape[1] if result_shape and len(result_shape) == 2 else 32
                self._expand_axis[result] = (varying_axis, n_cols)
                self._index_exprs[result] = self._index_exprs[src]
            return

        # Check if this promotes a 1D per-thread value to 2D — materialize as tile
        result_shape = result_type.shape if result_type else None
        if result_shape and len(result_shape) == 2:
            src_shape = src_val.ttype.shape
            src_size = src_shape[0] if src_shape else self.block_size
            dtype = src_val.ttype.dtype
            target_type = self.emitter.map_dtype(dtype)
            tid = self.emitter.thread_id_expr()

            # Allocate 1D shared memory and write per-thread values.
            # Must be shared: all threads need to read all values for broadcast.
            src_tile_1d = self._alloc_tile([src_size], dtype, force_shared=True)
            self._emit(f"if ((uint){tid} < {src_size}u) {{ {src_tile_1d.shared_name}[(uint){tid}] = ({target_type})({src_val.expr}); }}")
            self._emit(self.emitter.barrier())

            # Create 2D broadcast view of the 1D tile
            out_tile = TileInfo(
                shared_name=src_tile_1d.shared_name,
                shape=list(result_shape),
                dtype=dtype,
                broadcast_src=src_tile_1d,
            )
            self._tiles[result] = out_tile
            self._set_val(result, result_type, src_val.expr)
            backing = self._real_backing(out_tile)
            self._backing_refs.setdefault(backing, set()).add(result)
            return

        self._set_val(result, result_type, src_val.expr)

    def _gen_tt_addptr(self, op: Op):
        ptr = op.operands[0]
        offset = op.operands[1]
        result = op.results[0]

        # If offset is a tile, the result is a multi-element pointer tensor.
        # Create an offset tile so scatter stores can use base + tile[tid].
        if self._is_tile(offset) or self._is_tile(ptr):
            ptr_val = self._get_val(ptr)
            result_shape = op.result_types[0].shape if op.result_types else ptr_val.ttype.shape
            result_type = TType(dtype=ptr_val.ttype.dtype, shape=result_shape, is_ptr=True)

            # Determine the base pointer expr (trace through chained addptrs)
            base_expr = ptr_val.expr

            if self._is_tile(offset) and self._is_tile(ptr):
                # Both are tiles — add them element-wise into a new offset tile
                ptr_tile = self._get_tile(ptr)
                off_tile = self._get_tile(offset)
                out_shape = list(off_tile.shape)
                out_tile = self._alloc_tile(out_shape, 'i32')
                ptr_read = self._tile_read(ptr_tile, "_fi")
                off_read = self._tile_read(off_tile, "_fi")
                self._emit_tile_loop(out_tile.total,
                    f"{out_tile.shared_name}[_fi] = (int)({ptr_read} + {off_read});")
                self._register_tile(result, out_tile)
            elif self._is_tile(offset):
                off_tile = self._get_tile(offset)
                # If ptr itself came from an addptr with a tile (chained addptrs),
                # just alias the offset tile and let the store trace the base
                self._tiles[result] = off_tile
                # Track backing ref so the offset tile isn't freed while result is alive
                backing = self._real_backing(off_tile)
                self._backing_refs.setdefault(backing, set()).add(result)
            elif self._is_tile(ptr):
                # ptr is a tile of offsets, offset is scalar — add scalar to each
                ptr_tile = self._get_tile(ptr)
                out_shape = list(ptr_tile.shape)
                out_tile = self._try_reuse_in_place([ptr], out_shape, 'i32')
                if out_tile is None:
                    out_tile = self._alloc_tile(out_shape, 'i32')
                off_expr = self._get_expr(offset)
                ptr_read = self._tile_read(ptr_tile, "_fi")
                self._emit_tile_loop(out_tile.total,
                    f"{out_tile.shared_name}[_fi] = (int)({ptr_read} + {off_expr});")
                self._register_tile(result, out_tile)
                base_expr = ptr_val.expr

            self._set_val(result, result_type, base_expr)
            # Propagate buffer base tracking through tile addptr for HLSL
            if ptr in self._buf_base:
                self._buf_base[result] = self._buf_base[ptr]
            return

        ptr_expr = self._get_expr(ptr)
        off_expr = self._get_expr(offset)
        ptr_type = self._get_val(ptr).ttype
        result_type = TType(dtype=ptr_type.dtype, shape=ptr_type.shape, is_ptr=True)

        if not self.emitter.supports_ptr_cast():
            # HLSL: decompose into base buffer + offset (no pointer arithmetic)
            base = self._buf_base.get(ptr, ptr_expr)
            if ptr in self._buf_base:
                # Chained addptr: accumulate offsets
                prev_off = self._get_expr(ptr)
                var = self._fresh_var("idx")
                self._emit(f"int {var} = (int)({prev_off}) + (int)({off_expr});")
            else:
                # First addptr from buffer arg
                var = self._fresh_var("idx")
                self._emit(f"int {var} = (int)({off_expr});")
            self._set_val(result, result_type, var)
            self._buf_base[result] = base
            return

        var = self._fresh_var("ptr")
        self._emit(f"auto {var} = {ptr_expr} + {off_expr};")
        self._set_val(result, result_type, var)

    def _gen_tt_load(self, op: Op):
        self._flush_barrier()  # Cooperative loads write shared memory
        ptr = op.operands[0]
        result = op.results[0]
        ptr_val = self._get_val(ptr)
        result_dtype = ptr_val.ttype.dtype
        result_shape = op.result_types[0].shape if op.result_types else ptr_val.ttype.shape
        result_type = TType(dtype=result_dtype, shape=result_shape)
        metal_type = self.emitter.map_dtype(result_dtype)

        has_mask = len(op.operands) >= 2
        has_other = len(op.operands) >= 3

        # Check if this is a 2D load that should produce a tile
        is_2d = result_shape and len(result_shape) == 2

        if is_2d:
            load_info = self._analyze_2d_load(op)
            if load_info:
                # Pass mask tile if available
                mask_tile = None
                other_val = None
                if has_mask:
                    mask_ssa = op.operands[1]
                    if self._is_tile(mask_ssa):
                        mask_tile = self._get_tile(mask_ssa)
                    if has_other:
                        other_ssa = op.operands[2]
                        other_val = self._const_fill.get(other_ssa, self._get_expr(other_ssa))
                self._gen_tt_load_tile(op, load_info, result_dtype,
                                       mask_tile=mask_tile, other_val=other_val)
                return

        # 1D cooperative tile load, needed in two cases:
        # 1. 1D-in-2D kernel: dual expressions give wrong addresses
        # 2. num_threads < block_size: grid-stride loops need shared memory
        is_1d_in_2d = (result_shape and len(result_shape) == 1
                       and self.block_shape and len(self.block_shape) == 2)
        if is_1d_in_2d or (result_shape and len(result_shape) == 1
                           and self._needs_cooperative_loads()):
            loaded = self._gen_tt_load_1d_tile(op, result_shape, result_dtype)
            if loaded:
                return

        # Fallback: per-thread scalar load (original path)
        # When the pointer has a tile offset (from addptr with tile operand),
        # use base + tile[tid] for the address.
        if self._is_tile(ptr):
            off_tile = self._get_tile(ptr)
            tid = self.emitter.thread_id_expr()
            off_read = self._tile_read(off_tile, f"(uint){tid}")
            if not self.emitter.supports_ptr_cast():
                if ptr in self._buf_base:
                    buf_name = self._buf_base[ptr]
                    load_expr = f"{buf_name}[(int){ptr_val.expr} + (int){off_read}]"
                else:
                    load_expr = f"{ptr_val.expr}[{off_read}]"
            else:
                load_expr = f"*({ptr_val.expr} + {off_read})"
        elif not self.emitter.supports_ptr_cast() and ptr in self._buf_base:
            base = self._buf_base[ptr]
            off = ptr_val.expr
            load_expr = f"{base}[{off}]"
        else:
            load_expr = f"*({ptr_val.expr})"

        var = self._fresh_var("ld")
        if has_mask:
            mask_ssa = op.operands[1]
            if self._is_tile(mask_ssa):
                tid = self.emitter.thread_id_expr()
                mask_expr = self._tile_read(self._get_tile(mask_ssa), f"(uint){tid}")
            else:
                mask_expr = self._get_expr(mask_ssa)
            if has_other:
                other_ssa = op.operands[2]
                # Use scalar fill value for constant dense tiles (avoids OOB tile reads)
                other_expr = self._const_fill.get(other_ssa, self._get_expr(other_ssa))
                self._emit(f"{metal_type} {var} = {mask_expr} ? {load_expr} : ({metal_type}){other_expr};")
            else:
                self._emit(f"{metal_type} {var} = {mask_expr} ? {load_expr} : ({metal_type})0;")
        else:
            self._emit(f"{metal_type} {var} = {load_expr};")
        self._set_val(result, result_type, var)

    def _gen_tt_load_tile(self, op: Op, load_info: dict, dtype: str,
                          mask_tile=None, other_val=None):
        """Generate a cooperative 2D load into a shared memory tile."""
        result = op.results[0]
        rows = load_info['rows']
        cols = load_info['cols']
        total = rows * cols
        base = load_info['base']
        stride = load_info['stride']
        row_off = load_info['row_offset']
        col_off = load_info['col_offset']
        extra_off = load_info.get('extra_offset')  # HLSL: scalar base offset

        tile = self._alloc_tile([rows, cols], dtype)
        in_type = self.emitter.map_dtype(dtype)
        tid = self.emitter.thread_id_expr()
        self._max_coop_load_size = max(self._max_coop_load_size, total)

        # Build the element address expression
        col_stride = load_info.get('col_stride')
        col_expr = f"({col_off} + (int)_c)" if not col_stride else f"({col_off} + (int)_c) * {col_stride}"
        elem_addr = f"({row_off} + (int)_r) * {stride} + {col_expr}"
        if extra_off:
            elem_addr = f"(int){extra_off} + {elem_addr}"

        # Build mask read expression if mask tile provided
        mask_read = None
        if mask_tile is not None:
            mask_read = self._tile_read(mask_tile, "_flat")
            if other_val is None:
                other_val = "0"

        # Detect 1D per-row mask (all columns in a row share the same mask value)
        # Broadcast source may be [BM] (rank 1) or [BM, 1] (expand_dims, rank 2).
        is_row_mask = False
        if mask_tile is not None:
            bsrc = mask_tile.broadcast_src
            if bsrc is not None:
                if bsrc.rank == 1 and bsrc.shape[0] == rows or (bsrc.rank == 2 and bsrc.shape[0] == rows and
                      bsrc.shape[1] == 1):
                    is_row_mask = True
            elif mask_tile.rank == 1 and mask_tile.shape[0] == rows:
                is_row_mask = True

        # Vectorized loads when cols is divisible by 4 (float4 for f32, half4 for f16)
        # Requires pointer-cast support (MSL) for *(threadgroup vec4*)& syntax
        # With a 1D row mask, vec4 is safe (all 4 elements share the same mask)
        use_vec4 = (cols % 4 == 0 and dtype in ('f32', 'f16') and
                    self.emitter.supports_simd_matrix() and cols >= 4 and
                    self.emitter.supports_ptr_cast() and
                    (mask_read is None or is_row_mask))
        if use_vec4:
            vec_type = 'float4' if dtype == 'f32' else 'half4'
            total4 = total // 4
            self._emit(f"for (uint _fi4 = (uint){tid}; _fi4 < {total4}u; _fi4 += _tg_size.x) {{")
            self._emit("    uint _fb = _fi4 * 4u;")
            self._emit(f"    uint _r = _fb / {cols}u, _c = _fb % {cols}u;")
            if is_row_mask:
                mask_src = mask_tile.broadcast_src if mask_tile.broadcast_src else mask_tile
                mask_check = f"{mask_src.shared_name}[_r]"
                zero_vec = f"{vec_type}(0)" if other_val == "0" else f"{vec_type}(({in_type}){other_val})"
                self._emit(f"    *((threadgroup {vec_type}*)&{tile.shared_name}[_fb]) = "
                           f"{mask_check} ? *((device {vec_type}*)&{base}[{elem_addr}]) : {zero_vec};")
            else:
                self._emit(f"    *((threadgroup {vec_type}*)&{tile.shared_name}[_fb]) = "
                           f"*((device {vec_type}*)&{base}[{elem_addr}]);")
            self._emit("}")
        else:
            addr = f"{base}[{elem_addr}]"
            # Grid-stride loop: works for any thread count (critical for Intel GPUs
            # where pipeline maxTotalThreadsPerThreadgroup may be < block_size)
            self._emit(f"for (uint _flat = (uint){tid}; _flat < {total}u; _flat += _tg_size.x) {{")
            self._emit(f"    uint _r = _flat / {cols}u, _c = _flat % {cols}u;")
            if mask_read:
                self._emit(f"    {tile.shared_name}[_flat] = {mask_read} ? ({in_type}){addr} : ({in_type}){other_val};")
            else:
                self._emit(f"    {tile.shared_name}[_flat] = ({in_type}){addr};")
            self._emit("}")
        self._emit_barrier_direct()

        self._register_tile(result, tile)

    def _collect_1d_scalar_offset(self, ssa: str) -> str:
        """Recursively collect all scalar additive terms from a 1D tile offset.

        Traces through arith.addi chains, collecting tt.splat(scalar) terms.
        The tt.make_range part maps to the cooperative loop variable (_li)
        and contributes "0" to the scalar offset.

        Example: addi(splat(A), addi(splat(B), make_range)) → "(A) + (B)"
        """
        op = self._op_map.get(ssa)
        if not op:
            if ssa in self.ssa_map:
                return self._get_expr(ssa)
            return "0"

        if op.opname == 'tt.splat':
            return self._get_expr(op.operands[0])

        if op.opname == 'tt.make_range':
            return "0"

        if op.opname == 'arith.addi':
            terms = []
            for operand in op.operands:
                term = self._collect_1d_scalar_offset(operand)
                if term != "0":
                    terms.append(term)
            if not terms:
                return "0"
            if len(terms) == 1:
                return terms[0]
            return " + ".join(f"({t})" for t in terms)

        # Unknown op — use compiled expression if available
        if ssa in self.ssa_map:
            return self._get_expr(ssa)
        return "0"

    def _gen_tt_load_1d_tile(self, op: Op, shape: list, dtype: str) -> bool:
        """Load 1D data cooperatively into a shared memory tile in a 2D kernel.

        In 2D kernels, dual expressions give wrong per-thread addresses for 1D loads.
        This traces the pointer chain to extract base+offset and loads cooperatively.
        Returns True if handled, False to fall back to scalar load.
        """
        result = op.results[0]
        size = shape[0]
        metal_type = self.emitter.map_dtype(dtype)

        # Trace pointer: addptr(splat(base), addi(splat(start), make_range))
        # Also handles nested: addptr(addptr(splat(base), range), splat(scalar))
        ptr_ssa = op.operands[0]
        ptr_op = self._op_map.get(ptr_ssa)
        if not ptr_op or ptr_op.opname != 'tt.addptr':
            return False

        # Unwrap nested addptr chains to find base splat and collect offsets.
        # Pattern: addptr(addptr(...(splat(base), range_or_addi)...), splat(scalar))
        # At each level, one operand should lead to the splat+range, the other is a scalar offset.
        extra_scalar_offsets = []
        cur_op = ptr_op

        while True:
            base_ssa = cur_op.operands[0]
            base_op = self._op_map.get(base_ssa)
            if base_op and base_op.opname == 'tt.splat':
                # Found the base splat — this is the standard pattern
                splat_src = base_op.operands[0]
                if not self.emitter.supports_ptr_cast() and splat_src in self._buf_base:
                    base_expr = self._buf_base[splat_src]
                    base_scalar_off = self._get_expr(splat_src)
                else:
                    base_expr = self._get_expr(splat_src)
                    base_scalar_off = None
                # The offset operand contains the range + any addi scalar terms
                off_ssa = cur_op.operands[1]
                break
            elif base_op and base_op.opname == 'tt.addptr':
                # Nested addptr: operand[1] should be a splat(scalar) offset
                # and operand[0] is the inner addptr to continue unwinding
                off_op = self._op_map.get(cur_op.operands[1])
                if off_op and off_op.opname == 'tt.splat':
                    extra_scalar_offsets.append(self._get_expr(off_op.operands[0]))
                else:
                    # operand[1] has the range, operand[0] has the scalar
                    # Try swapping: maybe base has the scalar and off has the inner addptr
                    off_op2 = self._op_map.get(cur_op.operands[1])
                    if off_op2 and off_op2.opname == 'tt.addptr':
                        # operand[0] is splat(scalar), operand[1] is inner addptr
                        base_splat = self._op_map.get(cur_op.operands[0])
                        if base_splat and base_splat.opname == 'tt.splat':
                            extra_scalar_offsets.append(self._get_expr(base_splat.operands[0]))
                            cur_op = off_op2
                            continue
                    return False
                cur_op = base_op
                continue
            else:
                return False

        # Get starting offset from addi(splat(start), make_range) or just make_range
        # Must recursively collect ALL scalar terms from nested addi chains:
        #   addi(splat(A), addi(splat(B), make_range)) → start = A + B
        base_start_expr = self._collect_1d_scalar_offset(off_ssa)

        # Fold in scalar offset from base pointer (tt.addptr(buf, scalar) → splat)
        if base_scalar_off is not None:
            if base_start_expr == "0":
                base_start_expr = base_scalar_off
            else:
                base_start_expr = f"({base_scalar_off}) + ({base_start_expr})"

        # Full start_expr includes extra scalar offsets from nested addptr.
        # These are NOT part of the original mask comparison, so we track them separately.
        start_expr = base_start_expr
        for extra in extra_scalar_offsets:
            if start_expr == "0":
                start_expr = extra
            else:
                start_expr = f"({start_expr}) + ({extra})"

        tile = self._alloc_tile(shape, dtype, force_shared=True)
        tid = self.emitter.thread_id_expr()
        tg_size = "_tg_size.x"  # Runtime thread count for grid-stride

        has_mask = len(op.operands) >= 2
        has_other = len(op.operands) >= 3
        if has_other:
            other_ssa = op.operands[2]
            other_expr = self._const_fill.get(other_ssa, self._get_expr(other_ssa))
        else:
            other_expr = "0"

        if has_mask:
            # Extract comparison bound from mask: typically offs < N
            mask_ssa = op.operands[1]
            mask_op = self._op_map.get(mask_ssa)
            if mask_op and mask_op.opname == 'arith.cmpi':
                # Compare operand is typically a splat of N
                bound_ssa = mask_op.operands[1]
                bound_op = self._op_map.get(bound_ssa)
                if bound_op and bound_op.opname == 'tt.splat':
                    bound_expr = self._get_expr(bound_op.operands[0])
                else:
                    bound_expr = self._get_expr(bound_ssa)
                # When nested addptr adds extra offsets (e.g., bias_ptr + offs_n + N),
                # the mask was computed on the base range (offs_n < N), not the shifted range.
                # Use base_start_expr for the mask check, full start_expr for the array index.
                if extra_scalar_offsets:
                    self._emit(f"for (uint _li = (uint){tid}; _li < {size}u; _li += {tg_size}) {{")
                    self._emit(f"    int _loff = ({start_expr}) + (int)_li;")
                    self._emit(f"    int _moff = ({base_start_expr}) + (int)_li;")
                    self._emit(f"    {tile.shared_name}[_li] = (_moff < {bound_expr}) ? ({metal_type}){base_expr}[_loff] : ({metal_type}){other_expr};")
                    self._emit("}")
                else:
                    self._emit(f"for (uint _li = (uint){tid}; _li < {size}u; _li += {tg_size}) {{")
                    self._emit(f"    int _loff = ({start_expr}) + (int)_li;")
                    self._emit(f"    {tile.shared_name}[_li] = (_loff < {bound_expr}) ? ({metal_type}){base_expr}[_loff] : ({metal_type}){other_expr};")
                    self._emit("}")
            else:
                # Can't reconstruct mask — load unconditionally
                self._emit(f"for (uint _li = (uint){tid}; _li < {size}u; _li += {tg_size}) {{")
                self._emit(f"    {tile.shared_name}[_li] = ({metal_type}){base_expr}[({start_expr}) + (int)_li];")
                self._emit("}")
        else:
            self._emit(f"for (uint _li = (uint){tid}; _li < {size}u; _li += {tg_size}) {{")
            self._emit(f"    {tile.shared_name}[_li] = ({metal_type}){base_expr}[({start_expr}) + (int)_li];")
            self._emit("}")

        self._emit(self.emitter.barrier())
        self._register_tile(result, tile)
        return True

    def _gen_tt_store(self, op: Op):
        self._flush_barrier()  # Store reads from shared memory
        # Skip if the matmul optimized path already stored directly to C
        if self._dot_store_emitted:
            self._dot_store_emitted = False
            return

        ptr = op.operands[0]
        val = op.operands[1]
        has_mask = len(op.operands) >= 3

        # Check if value is a register accumulator (simdgroup registers)
        if val in self._reg_tiles:
            self._gen_tt_store_reg(op)
            return

        # Check if value is a tile
        if self._is_tile(val):
            self._gen_tt_store_tile(op)
            return

        # Check if the pointer is a tile of offsets (scatter store pattern).
        # The pointer SSA itself may have a tile registered from tt.addptr.
        if self._is_tile(ptr):
            tid = self.emitter.thread_id_expr()
            off_tile = self._get_tile(ptr)
            base_expr = self._get_expr(ptr)  # the base pointer (set by addptr)
            val_expr = self._get_expr(val)
            off_read = self._tile_read(off_tile, f'(uint){tid}')
            guard = f"(uint){tid} < {off_tile.total}u"
            if has_mask:
                mask_ssa = op.operands[2]
                if self._is_tile(mask_ssa):
                    mask_expr = self._tile_read(self._get_tile(mask_ssa), f"(uint){tid}")
                else:
                    mask_expr = self._get_expr(mask_ssa)
                guard = f"{guard} && {mask_expr}"
            if not self.emitter.supports_ptr_cast():
                if ptr in self._buf_base:
                    buf_name = self._buf_base[ptr]
                    self._emit(f"if ({guard}) {{ {buf_name}[(int){base_expr} + (int){off_read}] = {val_expr}; }}")
                else:
                    self._emit(f"if ({guard}) {{ {base_expr}[(int){off_read}] = {val_expr}; }}")
            else:
                self._emit(f"if ({guard}) {{ *({base_expr} + (int){off_read}) = {val_expr}; }}")
            return

        ptr_expr = self._get_expr(ptr)
        val_expr = self._get_expr(val)

        if not self.emitter.supports_ptr_cast() and ptr in self._buf_base:
            base = self._buf_base[ptr]
            off = ptr_expr  # the offset expression from addptr
            if has_mask:
                mask_ssa = op.operands[2]
                if self._is_tile(mask_ssa):
                    tid = self.emitter.thread_id_expr()
                    mask_expr = self._tile_read(self._get_tile(mask_ssa), f"(uint){tid}")
                else:
                    mask_expr = self._get_expr(mask_ssa)
                self._emit(f"if ({mask_expr}) {{ {base}[{off}] = {val_expr}; }}")
            else:
                self._emit(f"{base}[{off}] = {val_expr};")
        else:
            if has_mask:
                mask_ssa = op.operands[2]
                if self._is_tile(mask_ssa):
                    tid = self.emitter.thread_id_expr()
                    mask_expr = self._tile_read(self._get_tile(mask_ssa), f"(uint){tid}")
                else:
                    mask_expr = self._get_expr(mask_ssa)
                self._emit(f"if ({mask_expr}) {{ *({ptr_expr}) = {val_expr}; }}")
            else:
                self._emit(f"*({ptr_expr}) = {val_expr};")

    def _gen_tt_store_tile(self, op: Op):
        """Generate cooperative 2D store from a shared memory tile."""
        ptr_ssa = op.operands[0]
        val_ssa = op.operands[1]
        has_mask = len(op.operands) >= 3
        mask_tile = None
        if has_mask and self._is_tile(op.operands[2]):
            mask_tile = self._get_tile(op.operands[2])
        tile = self._get_tile(val_ssa)

        # Use matmul store params if available (avoids materializing BM×BN index tile)
        if self._matmul_store_params and tile.rank == 2:
            p = self._matmul_store_params
            self._matmul_store_params = None  # consume
            BM, BN = p['BM'], p['BN']
            if tile.shape == [BM, BN]:
                out_type = self.emitter.map_dtype(tile.dtype)
                tid = self.emitter.thread_id_expr()
                total = BM * BN
                row_off = f"(int){p['pid_m']} * {BM}"
                col_off = f"(int){p['pid_n']} * {BN}"
                addr = f"{p['c_base']}[({row_off} + (int)_r) * {p['stride_cm']} + ({col_off} + (int)_c)]"
                read_expr = self._tile_read(tile, '_flat')
                self._emit(f"for (uint _flat = (uint){tid}; _flat < {total}u; _flat += _tg_size.x) {{")
                self._emit(f"    uint _r = _flat / {BN}u, _c = _flat % {BN}u;")
                self._emit(f"    {addr} = ({out_type}){read_expr};")
                self._emit("}")
                return

        # Try to analyze the pointer chain to get base and stride
        ptr_op = self._op_map.get(ptr_ssa)

        # Full cooperative store analysis (for 2D tiles)
        store_info = None
        if ptr_op and ptr_op.opname == 'tt.addptr':
            store_info = self._analyze_2d_load(Op(
                results=[], opname='tt.load', operands=[ptr_ssa],
                attrs={}, type_str='',
                result_types=[TType(dtype=tile.dtype, shape=tile.shape)],
            ))

        if not store_info:
            # Fallback: cooperative 1D tile store
            # Trace the pointer chain to extract base_ptr + offset formula
            if self._gen_1d_tile_store(op, tile, has_mask, mask_tile):
                return
            # Last resort: per-thread store with bounds guard
            ptr_expr = self._get_expr(ptr_ssa)
            tid = self.emitter.thread_id_expr()
            total = tile.total
            read_expr = self._tile_read(tile, f'(uint){tid}')
            guard = f"(uint){tid} < {total}u"
            if has_mask and not mask_tile:
                mask_ssa = op.operands[2]
                if self._is_tile(mask_ssa):
                    mask_expr = self._tile_read(self._get_tile(mask_ssa), f"(uint){tid}")
                else:
                    mask_expr = self._get_expr(mask_ssa)
                guard = f"{guard} && {mask_expr}"
            if not self.emitter.supports_ptr_cast() and ptr_ssa in self._buf_base:
                base = self._buf_base[ptr_ssa]
                self._emit(f"if ({guard}) {{ {base}[(int){ptr_expr}] = {read_expr}; }}")
            else:
                self._emit(f"if ({guard}) {{ *({ptr_expr}) = {read_expr}; }}")
            return

        rows = store_info['rows']
        cols = store_info['cols']
        total = rows * cols
        base = store_info['base']
        stride = store_info['stride']
        row_off = store_info['row_offset']
        col_off = store_info['col_offset']

        extra_off = store_info.get('extra_offset')  # HLSL: scalar base offset
        out_type = self.emitter.map_dtype(tile.dtype)
        tid = self.emitter.thread_id_expr()

        # Build the store statement, optionally guarded by mask
        col_stride = store_info.get('col_stride')
        col_expr = f"({col_off} + (int)_c)" if not col_stride else f"({col_off} + (int)_c) * {col_stride}"
        elem_addr = f"({row_off} + (int)_r) * {stride} + {col_expr}"
        if extra_off:
            elem_addr = f"(int){extra_off} + {elem_addr}"
        addr = f"{base}[{elem_addr}]"
        store_stmt = f"{addr} = ({out_type}){self._tile_read(tile, '_flat')};"
        if mask_tile:
            mask_read = self._tile_read(mask_tile, "_flat")
            store_stmt = f"if ({mask_read}) {{ {store_stmt} }}"

        # Grid-stride loop: works for any thread count (critical for Intel GPUs
        # where pipeline maxTotalThreadsPerThreadgroup may be < block_size)
        self._emit(f"for (uint _flat = (uint){tid}; _flat < {total}u; _flat += _tg_size.x) {{")
        self._emit(f"    uint _r = _flat / {cols}u, _c = _flat % {cols}u;")
        self._emit(f"    {store_stmt}")
        self._emit("}")

    def _gen_tt_store_reg(self, op: Op):
        """Store register accumulator directly to device memory via simdgroup_store."""
        val_ssa = op.operands[1]
        ptr_ssa = op.operands[0]
        reg = self._reg_tiles[val_ssa]
        _BM_r, d_r = reg.shape

        # Analyze the pointer chain to get base pointer and stride
        ptr_op = self._op_map.get(ptr_ssa)
        store_info = None
        if ptr_op and ptr_op.opname == 'tt.addptr':
            store_info = self._analyze_2d_load(Op(
                results=[], opname='tt.load', operands=[ptr_ssa],
                attrs={}, type_str='',
                result_types=[TType(dtype=reg.dtype, shape=reg.shape)],
            ))

        if store_info:
            base = store_info['base']
            stride = store_info['stride']
            row_off = store_info['row_offset']
            col_off = store_info['col_offset']
        else:
            # Fallback: use func args for output pointer and stride
            base = self._get_expr(ptr_ssa)
            stride = f"{d_r}"
            row_off = "0"
            col_off = "0"

        # Check if a dtype cast is needed (e.g., f32 reg acc → f16 output)
        cast_dtype = self._reg_acc_cast_dtype.get(val_ssa)
        if cast_dtype and cast_dtype != reg.dtype:
            self._emit_reg_cast_store(reg, base, stride, row_off, col_off, cast_dtype)
            return

        bi_v = self._fresh_var("bi")
        blk_v = self._fresh_var("blk")
        self._emit(f"for (uint {bi_v} = 0; {bi_v} < {reg.blocks_per_sg}u; {bi_v}++) {{")
        self._emit(f"    uint {blk_v} = {reg.sg_var} * {reg.blocks_per_sg}u + {bi_v};")
        self._emit(f"    if ({blk_v} < {reg.num_blocks_total}u) {{")
        self._emit(f"        uint _br = {blk_v} / {reg.num_blocks_n}u, _bc = {blk_v} % {reg.num_blocks_n}u;")
        self._emit(f"        {self.emitter.simd_store(f'{reg.reg_name}[{bi_v}]', f'&{base}[({row_off} + (int)(_br * 8u)) * {stride} + ({col_off} + (int)(_bc * 8u))]', f'(ulong){stride}')}")
        self._emit("    }")
        self._emit("}")

    def _materialize_reg_acc(self, ssa_name: str) -> TileInfo:
        """Materialize a register accumulator to a shared memory tile.

        Stores simdgroup registers into a newly allocated shared tile so that
        normal tile operations (element-wise ops, stores) can access the data.
        Returns the allocated tile.
        """
        reg = self._reg_tiles[ssa_name]
        BM_r, d_r = reg.shape
        tile = self._alloc_tile([BM_r, d_r], reg.dtype)
        # Zero the tile first (SGs may not cover all elements)
        acc_type_str = self.emitter.map_dtype(reg.dtype)
        self._emit_tile_loop(tile.total,
            f"{tile.shared_name}[_fi] = ({acc_type_str})0;")
        self._flush_barrier()  # Must flush fused loops before simdgroup_store
        # Store each SG's blocks into the tile
        bi_v = self._fresh_var("bi")
        blk_v = self._fresh_var("blk")
        self._emit(f"for (uint {bi_v} = 0; {bi_v} < {reg.blocks_per_sg}u; {bi_v}++) {{")
        self._emit(f"    uint {blk_v} = {reg.sg_var} * {reg.blocks_per_sg}u + {bi_v};")
        self._emit(f"    if ({blk_v} < {reg.num_blocks_total}u) {{")
        self._emit(f"        uint _br = {blk_v} / {reg.num_blocks_n}u, _bc = {blk_v} % {reg.num_blocks_n}u;")
        self._emit(f"        {self.emitter.simd_store(f'{reg.reg_name}[{bi_v}]', f'&{tile.shared_name}[_br * {8 * d_r}u + _bc * 8u]', f'{d_r}ul')}")
        self._emit("    }")
        self._emit("}")
        self._flush_barrier()
        # Remove from reg_tiles, register as normal tile
        del self._reg_tiles[ssa_name]
        self._register_tile(ssa_name, tile)
        return tile

    def _emit_reg_cast_store(self, reg: RegAccInfo, base: str, stride: str,
                              row_off: str, col_off: str, cast_dtype: str):
        """Store register accumulator to device memory with dtype cast.

        Uses a per-SG scratch buffer: simdgroup_store 8x8 block to scratch,
        barrier, then per-lane read + cast + write to global memory.
        """
        # Allocate per-SG scratch buffer (64 floats each) if not yet allocated
        if self._cast_scratch_name is None:
            num_sgs = max(self._reg_acc_min_sgs, self.block_size // 32)
            scratch_total = num_sgs * 64
            self._cast_scratch_name = self._fresh_var("sCast")
            self._tile_decls.append(
                self.emitter.shared_memory_decl(self._cast_scratch_name, 'f32', scratch_total))
            self._tg_bytes_allocated += scratch_total * 4

        scratch = self._cast_scratch_name
        out_type = self.emitter.map_dtype(cast_dtype)
        tid = self.emitter.thread_id_expr()
        sg_var = reg.sg_var
        bi_v = self._fresh_var("bi")
        blk_v = self._fresh_var("blk")

        self._emit(f"for (uint {bi_v} = 0; {bi_v} < {reg.blocks_per_sg}u; {bi_v}++) {{")
        self._emit(f"    uint {blk_v} = {sg_var} * {reg.blocks_per_sg}u + {bi_v};")
        # simdgroup_store to per-SG scratch (stride=8 for 8x8 block)
        self._emit(f"    if ({blk_v} < {reg.num_blocks_total}u)")
        self._emit(f"        {self.emitter.simd_store(f'{reg.reg_name}[{bi_v}]', f'&{scratch}[{sg_var} * 64u]', '8ul')}")
        self._emit(self.emitter.barrier())
        # Per-lane read, cast, write to global memory
        self._emit(f"    {{ uint _lane = (uint){tid} % 32u;")
        self._emit(f"      if ({blk_v} < {reg.num_blocks_total}u) {{")
        self._emit(f"        uint _br = {blk_v} / {reg.num_blocks_n}u, _bc = {blk_v} % {reg.num_blocks_n}u;")
        self._emit("        for (uint _ei = 0; _ei < 2u; _ei++) {")
        self._emit("          uint _idx = _lane * 2u + _ei;")
        self._emit("          uint _r = _idx / 8u, _c = _idx % 8u;")
        self._emit(f"          {base}[({row_off} + (int)(_br * 8u + _r)) * {stride} + ({col_off} + (int)(_bc * 8u + _c))] = ({out_type})({scratch}[{sg_var} * 64u + _idx]);")
        self._emit("        }")
        self._emit("      }")
        self._emit("    }")
        self._emit("}")

    def _gen_1d_tile_store(self, op: Op, tile: TileInfo, has_mask: bool,
                            mask_tile: TileInfo | None) -> bool:
        """Generate cooperative store for a 1D tile. Returns True if handled."""
        ptr_ssa = op.operands[0]
        ptr_op = self._op_map.get(ptr_ssa)
        if not ptr_op or ptr_op.opname != 'tt.addptr':
            return False

        # Trace: addptr(splat(base_ptr), offset_with_make_range)
        # Also handles nested: addptr(addptr(splat(base), range), splat(scalar))
        extra_scalar_offsets = []
        cur_op = ptr_op

        while True:
            base_ssa = cur_op.operands[0]
            base_op = self._op_map.get(base_ssa)
            if base_op and base_op.opname == 'tt.splat':
                base_arg = base_op.operands[0]
                if base_arg not in self.ssa_map:
                    return False
                if not self.emitter.supports_ptr_cast() and base_arg in self._buf_base:
                    base_expr = self._buf_base[base_arg]
                    base_scalar_off = self._get_expr(base_arg)
                else:
                    base_expr = self._get_expr(base_arg)
                    base_scalar_off = None
                off_ssa_for_store = cur_op.operands[1]
                break
            elif base_op and base_op.opname == 'tt.addptr':
                off_op = self._op_map.get(cur_op.operands[1])
                if off_op and off_op.opname == 'tt.splat':
                    extra_scalar_offsets.append(self._get_expr(off_op.operands[0]))
                else:
                    return False
                cur_op = base_op
                continue
            else:
                return False

        # Extract the per-block offset (e.g., pid * BLOCK_M)
        off_expr = self._extract_tile_offset(off_ssa_for_store)

        # Fold in scalar offset from base pointer
        if base_scalar_off is not None:
            if off_expr == "0":
                off_expr = base_scalar_off
            else:
                off_expr = f"({base_scalar_off}) + ({off_expr})"

        # Fold in extra scalar offsets from nested addptr unwinding
        for extra in extra_scalar_offsets:
            if off_expr == "0":
                off_expr = extra
            else:
                off_expr = f"({off_expr}) + ({extra})"

        total = tile.total
        out_type = self.emitter.map_dtype(tile.dtype)
        tid = self.emitter.thread_id_expr()

        # Reconstruct mask from the tt.store's mask operand
        mask_guard = None
        scalar_mask = None
        if has_mask:
            mask_ssa = op.operands[2]
            mask_op = self._op_map.get(mask_ssa)
            if mask_op and mask_op.opname == 'arith.cmpi':
                # cmpi slt, %lhs, %splat(N) — bounds check mask
                bound_ssa = mask_op.operands[1]
                bound_op = self._op_map.get(bound_ssa)
                if bound_op and bound_op.opname == 'tt.splat':
                    bound_expr = self._get_expr(bound_op.operands[0])
                    mask_lhs_ssa = mask_op.operands[0]
                    mask_off = self._extract_tile_offset(mask_lhs_ssa)
                    if mask_off == "0":
                        mask_guard = f"(int)_fi < {bound_expr}"
                    else:
                        mask_guard = f"({mask_off} + (int)_fi) < {bound_expr}"
                elif not self._is_tile(mask_ssa):
                    # Scalar comparison (e.g., program_id == constant) — uniform mask
                    scalar_mask = self._get_expr(mask_ssa)
            elif not self._is_tile(mask_ssa) and mask_ssa in self.ssa_map:
                # Other scalar mask (e.g., from select, arith ops)
                scalar_mask = self._get_expr(mask_ssa)

        # Grid-stride loop for portability across different GPU thread limits
        if scalar_mask:
            self._emit(f"if ({scalar_mask}) {{")
        self._emit(f"for (uint _fi = (uint){tid}; _fi < {total}u; _fi += _tg_size.x) {{")
        if mask_guard:
            self._emit(f"    if ({mask_guard})")
            self._emit(f"        {base_expr}[{off_expr} + (int)_fi] = ({out_type}){self._tile_read(tile, '_fi')};")
        else:
            self._emit(f"    {base_expr}[{off_expr} + (int)_fi] = ({out_type}){self._tile_read(tile, '_fi')};")
        self._emit("}")
        if scalar_mask:
            self._emit("}")
        return True

    def _gen_tt_reduce(self, op: Op):
        """Generate reduction. Supports axis reduction on tiles using SIMD ops."""
        # Check for multi-result reduce (argmax/argmin)
        if len(op.results) >= 2 and len(op.operands) >= 2:
            self._gen_tt_reduce_argmax(op)
            return

        src = op.operands[0]
        result = op.results[0]
        src_val = self._get_val(src)

        # Determine reduce operation
        reduce_op = op.attrs.get('reduce_op', 'add')
        # Parse axis from the reduce op's raw_text or type signature
        axis = -1  # default: full reduction
        if 'axis' in op.attrs:
            axis = int(op.attrs['axis'])

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            self._gen_tt_reduce_tile(op, src_tile, reduce_op, axis)
            return

        # 1D reduction: each thread holds one element, reduce across all threads
        target_type = self.emitter.map_dtype(src_val.ttype.dtype)
        tid = self.emitter.thread_id_expr()
        threads = min(self.block_size, self.MAX_THREADS)

        identity = {'max': f'-({target_type})HUGE_VALF',
                     'min': f'({target_type})HUGE_VALF'}.get(reduce_op, f'({target_type})0')

        svar = self._fresh_var("red_s")
        var = self._fresh_var("red")

        if self.emitter.supports_wave_reduce():
            # Wave + tree reduction: wave-level reduce, then defer cross-wave sum
            # Level 1: reduce within each wave
            wave_reduce_expr = self.emitter.simd_reduce(reduce_op, src_val.expr)
            sg_r = self._fresh_var("wr")
            self._emit(f"{target_type} {sg_r} = {wave_reduce_expr};")
            # Shared buffer for per-wave partial results (max 128 waves for wave_size=8)
            max_waves = max((threads + 7) // 8, 32)
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar, src_val.ttype.dtype, max_waves))
            # Defer: accumulate wave result for batched barrier flush
            result_ssa = result
            if self._unroll_copy_idx is not None:
                result_ssa = f"{result}_u{self._unroll_copy_idx}"
            self._pending_wave_reductions.append({
                'result_ssa': result_ssa,
                'svar': svar,
                'sg_r': sg_r,
                'target_type': target_type,
                'reduce_op': reduce_op,
                'dtype': src_val.ttype.dtype,
            })
            return  # _set_val deferred to _flush_wave_reductions
        elif not self.emitter.supports_simd_matrix():
            # Shared memory tree reduction — correct for any SIMD width
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar, src_val.ttype.dtype, threads))
            self._emit(f"{svar}[(uint){tid}] = {src_val.expr};")
            self._emit(self.emitter.barrier())
            merge = {'max': 'max', 'min': 'min'}.get(reduce_op, None)
            rs = self._fresh_var("rs")
            if merge:
                self._emit(f"for (uint {rs} = _tg_size.x / 2u; {rs} > 0u; {rs} >>= 1u) {{")
                self._emit(f"    if ((uint){tid} < {rs}) {svar}[(uint){tid}] = {merge}({svar}[(uint){tid}], {svar}[(uint){tid} + {rs}]);")
            else:
                self._emit(f"for (uint {rs} = _tg_size.x / 2u; {rs} > 0u; {rs} >>= 1u) {{")
                self._emit(f"    if ((uint){tid} < {rs}) {svar}[(uint){tid}] += {svar}[(uint){tid} + {rs}];")
            self._emit(f"    {self.emitter.barrier()}")
            self._emit("}")
            self._emit(f"{target_type} {var} = {svar}[0];")
        else:
            # Legacy MSL simd path (simdgroup_matrix enabled)
            n_simd_groups = max(1, (threads + 31) // 32)
            simd_fn = {'max': 'simd_max', 'min': 'simd_min'}.get(reduce_op, 'simd_sum')
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar, src_val.ttype.dtype, max(n_simd_groups, 32)))
            if n_simd_groups == 1:
                reduce_expr = self.emitter.simd_reduce(reduce_op, src_val.expr)
                self._emit(f"{target_type} {var} = {reduce_expr};")
            else:
                sg_r = self._fresh_var("sgr")
                self._emit(f"{target_type} {sg_r} = {simd_fn}({src_val.expr});")
                self._emit(f"if ((uint){tid} % 32u == 0u) {{ {svar}[(uint){tid} / 32u] = {sg_r}; }}")
                self._emit(self.emitter.barrier())
                self._emit(f"{target_type} {var};")
                self._emit(f"if ((uint){tid} < 32u) {{")
                self._emit(f"    {target_type} _pv = ((uint){tid} < {n_simd_groups}u) ? {svar}[(uint){tid}] : {identity};")
                self._emit(f"    {var} = {simd_fn}(_pv);")
                self._emit(f"}} else {{ {var} = 0; }}")
                self._emit(f"if ((uint){tid} == 0u) {{ {svar}[0] = {var}; }}")
                self._emit(self.emitter.barrier())
                self._emit(f"{var} = {svar}[0];")

        result_type = TType(dtype=src_val.ttype.dtype)
        self._set_val(result, result_type, var)

    def _gen_tt_reduce_tile(self, op: Op, src_tile: TileInfo, reduce_op: str, axis: int):
        """Reduce a tile along an axis using SIMD operations."""
        self._flush_barrier()  # Reduction reads across threads
        result = op.results[0]
        dtype = src_tile.dtype
        metal_type = self.emitter.map_dtype(dtype)
        tid = self.emitter.thread_id_expr()

        if src_tile.rank == 2 and axis == 1:
            # Row-wise reduction: [M, N] -> [M]
            M, N = src_tile.shape
            out_tile = self._alloc_tile([M], dtype)
            threads = min(self.block_size, self.MAX_THREADS)

            if N <= 32 and self.emitter.supports_simd_matrix():
                # N fits in SIMD width — use simd_max/simd_sum
                # Each SIMD group (32 threads) handles one row
                simd_fn = {'max': 'simd_max', 'add': 'simd_sum',
                           'sum': 'simd_sum', 'min': 'simd_min'}.get(reduce_op, 'simd_sum')
                identity = {'max': f"-({metal_type})HUGE_VALF",
                            'min': f"({metal_type})HUGE_VALF"}.get(
                            reduce_op, f"({metal_type})0")
                # Map thread to (row, col) in [M, 32] virtual tile
                # For N < 32, lanes >= N get identity value
                sg_idx = self._fresh_var("sgr")
                lane_var = self._fresh_var("lane")
                self._emit(f"uint {sg_idx} = (uint){tid} / 32u;")
                self._emit(f"uint {lane_var} = (uint){tid} % 32u;")
                # Loop so each SG can handle multiple rows (fewer threads required)
                row_var = self._fresh_var("sgrow")
                self._emit(f"for (uint {row_var} = {sg_idx}; {row_var} < {M}u; {row_var} += _tg_size.x / 32u) {{")
                if N == 32:
                    self._emit(f"    {metal_type} _rv = {self._tile_read(src_tile, f'{row_var} * {N}u + {lane_var}')};")
                else:
                    self._emit(f"    {metal_type} _rv = ({lane_var} < {N}u) ? "
                               f"{self._tile_read(src_tile, f'{row_var} * {N}u + {lane_var}')} : {identity};")
                self._emit(f"    {metal_type} _reduced = {simd_fn}(_rv);")
                self._emit(f"    if ({lane_var} == 0u) {{")
                self._emit(f"        {out_tile.shared_name}[{row_var}] = _reduced;")
                self._emit("    }")
                self._emit("}")
                self._emit_barrier_direct()
            else:
                # General case: grid-stride reduction via shared memory
                # Each thread accumulates over its row's elements
                identity = {
                    'max': f"-({metal_type})HUGE_VALF",
                    'min': f"({metal_type})HUGE_VALF",
                }.get(reduce_op, f"({metal_type})0")
                merge = {
                    'max': 'max', 'min': 'min',
                    'add': '+', 'sum': '+',
                }.get(reduce_op, '+')

                self._emit(f"for (uint _row = (uint){tid}; _row < {M}u; _row += _tg_size.x) {{")

                self._emit(f"    {metal_type} _acc = {identity};")
                kk = self._fresh_var("rk")
                self._emit(f"    for (uint {kk} = 0; {kk} < {N}u; {kk}++) {{")
                elem = self._tile_read(src_tile, f"_row * {N}u + {kk}")
                if merge in ('+',):
                    self._emit(f"        _acc = _acc + {elem};")
                else:
                    self._emit(f"        _acc = {merge}(_acc, {elem});")
                self._emit("    }")
                self._emit(f"    {out_tile.shared_name}[_row] = _acc;")

                self._emit("}")
                self._emit(self.emitter.barrier())

            self._register_tile(result, out_tile)

        elif src_tile.rank == 2 and axis == 0:
            # Column-wise reduction: [M, N] -> [N]
            M, N = src_tile.shape
            out_tile = self._alloc_tile([N], dtype)
            identity = {'max': f"-({metal_type})HUGE_VALF",
                        'min': f"({metal_type})HUGE_VALF"}.get(reduce_op, f"({metal_type})0")
            merge = {'max': 'max', 'min': 'min'}.get(reduce_op, '+')

            self._emit(f"for (uint _col = (uint){tid}; _col < {N}u; _col += _tg_size.x) {{")

            self._emit(f"    {metal_type} _acc = {identity};")
            kk = self._fresh_var("rk")
            self._emit(f"    for (uint {kk} = 0; {kk} < {M}u; {kk}++) {{")
            elem = self._tile_read(src_tile, f"{kk} * {N}u + _col")
            if merge == '+':
                self._emit(f"        _acc = _acc + {elem};")
            else:
                self._emit(f"        _acc = {merge}(_acc, {elem});")
            self._emit("    }")
            self._emit(f"    {out_tile.shared_name}[_col] = _acc;")

            self._emit("}")
            self._emit(self.emitter.barrier())
            self._register_tile(result, out_tile)

        else:
            # Fallback: full reduction of 1D tile to scalar
            target_type = self.emitter.map_dtype(dtype)
            N = src_tile.total
            threads = min(self.block_size, self.MAX_THREADS)
            n_simd_groups = max(1, (threads + 31) // 32)

            identity = {'max': f'-({target_type})HUGE_VALF',
                         'min': f'({target_type})HUGE_VALF'}.get(reduce_op, f'({target_type})0')
            simd_fn = {'max': 'simd_max', 'min': 'simd_min'}.get(reduce_op, 'simd_sum')

            # Grid-stride accumulation: each thread accumulates its portion of the tile,
            # then wave/tree reduction combines partial sums across threads.
            # Always use grid-stride to be safe when tile size >= actual thread count.
            safe_val = self._fresh_var("rv")
            merge_op = {'max': 'max', 'min': 'min'}.get(reduce_op, '+')
            self._emit(f"{target_type} {safe_val} = {identity};")
            self._emit(f"for (uint _ri = (uint){tid}; _ri < {N}u; _ri += _tg_size.x) {{")
            elem = self._tile_read(src_tile, "_ri")
            if merge_op in ('+',):
                self._emit(f"    {safe_val} = {safe_val} + {elem};")
            else:
                self._emit(f"    {safe_val} = {merge_op}({safe_val}, {elem});")
            self._emit("}")

            if self.emitter.supports_wave_reduce():
                # Wave + tree reduction for tile 1D fallback — defer for batching
                svar = self._fresh_var("red_s")
                max_waves = max((threads + 7) // 8, 32)
                self._tile_decls.append(
                    self.emitter.shared_memory_decl(svar, dtype, max_waves))
                wave_reduce_expr = self.emitter.simd_reduce(reduce_op, safe_val)
                sg_r = self._fresh_var("wr")
                self._emit(f"{target_type} {sg_r} = {wave_reduce_expr};")
                # Defer: accumulate wave result for batched barrier flush
                result_ssa = result
                if self._unroll_copy_idx is not None:
                    result_ssa = f"{result}_u{self._unroll_copy_idx}"
                self._pending_wave_reductions.append({
                    'result_ssa': result_ssa,
                    'svar': svar,
                    'sg_r': sg_r,
                    'target_type': target_type,
                    'reduce_op': reduce_op,
                    'dtype': dtype,
                })
                return  # _set_val deferred to _flush_wave_reductions
            elif not self.emitter.supports_simd_matrix():
                # Shared memory tree reduction — correct for any SIMD width
                svar = self._fresh_var("red_s")
                self._tile_decls.append(
                    self.emitter.shared_memory_decl(svar, dtype, threads))
                self._emit(f"{svar}[(uint){tid}] = {safe_val};")
                self._emit(self.emitter.barrier())
                merge = {'max': 'max', 'min': 'min'}.get(reduce_op, None)
                rs = self._fresh_var("rs")
                if merge:
                    self._emit(f"for (uint {rs} = _tg_size.x / 2u; {rs} > 0u; {rs} >>= 1u) {{")
                    self._emit(f"    if ((uint){tid} < {rs}) {svar}[(uint){tid}] = {merge}({svar}[(uint){tid}], {svar}[(uint){tid} + {rs}]);")
                else:
                    self._emit(f"for (uint {rs} = _tg_size.x / 2u; {rs} > 0u; {rs} >>= 1u) {{")
                    self._emit(f"    if ((uint){tid} < {rs}) {svar}[(uint){tid}] += {svar}[(uint){tid} + {rs}];")
                self._emit(f"    {self.emitter.barrier()}")
                self._emit("}")
                var = self._fresh_var("red")
                self._emit(f"{target_type} {var} = {svar}[0];")
            else:
                # Legacy MSL simd path
                n_simd_groups = max(1, (threads + 31) // 32)
                simd_fn = {'max': 'simd_max', 'min': 'simd_min'}.get(reduce_op, 'simd_sum')
                svar = self._fresh_var("red_s")
                self._tile_decls.append(
                    self.emitter.shared_memory_decl(svar, dtype, max(n_simd_groups, 32)))
                sg_r = self._fresh_var("sgr")
                self._emit(f"{target_type} {sg_r} = {simd_fn}({safe_val});")
                self._emit(f"if ((uint){tid} % 32u == 0u) {{ {svar}[(uint){tid} / 32u] = {sg_r}; }}")
                self._emit(self.emitter.barrier())
                var = self._fresh_var("red")
                self._emit(f"{target_type} {var};")
                self._emit(f"if ((uint){tid} < 32u) {{")
                self._emit(f"    {target_type} _pv = ((uint){tid} < {n_simd_groups}u) ? {svar}[(uint){tid}] : {identity};")
                self._emit(f"    {var} = {simd_fn}(_pv);")
                self._emit(f"}} else {{ {var} = 0; }}")
                self._emit(f"if ((uint){tid} == 0u) {{ {svar}[0] = {var}; }}")
                self._emit(self.emitter.barrier())
                self._emit(f"{var} = {svar}[0];")

            self._set_val(result, TType(dtype=dtype), var)

    def _gen_tt_reduce_argmax(self, op: Op):
        """Generate argmax/argmin reduction (multi-result tt.reduce with value+index)."""
        val_src = op.operands[0]  # value tensor
        val_result = op.results[0]  # output value (scalar)
        idx_result = op.results[1]  # output index (scalar)

        # Detect argmin vs argmax from reduce_op attribute
        reduce_op = op.attrs.get('reduce_op', 'argmax')
        is_argmin = reduce_op == 'argmin'

        val_val = self._get_val(val_src)
        val_dtype = val_val.ttype.dtype
        val_type = self.emitter.map_dtype(val_dtype)
        tid = self.emitter.thread_id_expr()

        if self._is_tile(val_src):
            src_tile = self._get_tile(val_src)
            N = src_tile.total

            # Serial argmax/argmin over tile elements (thread 0 iterates)
            # Then broadcast result via shared memory
            svar_v = self._fresh_var("arg_sv")
            svar_i = self._fresh_var("arg_si")
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar_v, val_dtype, 1))
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar_i, 'i32', 1))

            cmp = '>' if not is_argmin else '<'
            init_val = f'-({val_type})HUGE_VALF' if not is_argmin else f'({val_type})HUGE_VALF'

            self._emit(f"if ((uint){tid} == 0u) {{")
            self._emit(f"    {val_type} _best_v = {init_val};")
            self._emit("    int _best_i = 0;")
            self._emit(f"    for (uint _ai = 0; _ai < {N}u; _ai++) {{")
            self._emit(f"        {val_type} _av = {self._tile_read(src_tile, '_ai')};")
            self._emit(f"        if (_av {cmp} _best_v) {{ _best_v = _av; _best_i = (int)_ai; }}")
            self._emit("    }")
            self._emit(f"    {svar_v}[0] = _best_v;")
            self._emit(f"    {svar_i}[0] = _best_i;")
            self._emit("}")
            self._emit(self.emitter.barrier())

            var_v = self._fresh_var("argv")
            var_i = self._fresh_var("argi")
            self._emit(f"{val_type} {var_v} = {svar_v}[0];")
            self._emit(f"int {var_i} = {svar_i}[0];")

            self._set_val(val_result, TType(dtype=val_dtype), var_v)
            self._set_val(idx_result, TType(dtype='i32'), var_i)
        else:
            # Per-thread values — use SIMD reduction for value, then find index
            # Simpler approach: shared memory serial scan
            threads = min(self.block_size, self.MAX_THREADS)
            svar_v = self._fresh_var("arg_sv")
            svar_i = self._fresh_var("arg_si")
            svar_vals = self._fresh_var("arg_vals")
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar_v, val_dtype, 1))
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar_i, 'i32', 1))
            self._tile_decls.append(
                self.emitter.shared_memory_decl(svar_vals, val_dtype, threads))

            cmp = '>' if not is_argmin else '<'
            init_val = f'-({val_type})HUGE_VALF' if not is_argmin else f'({val_type})HUGE_VALF'

            # Each thread writes its value to shared memory
            self._emit(f"{svar_vals}[(uint){tid}] = {val_val.expr};")
            self._emit(self.emitter.barrier())
            # Thread 0 scans all values
            self._emit(f"if ((uint){tid} == 0u) {{")
            self._emit(f"    {val_type} _best_v = {init_val};")
            self._emit("    int _best_i = 0;")
            self._emit("    for (uint _ai = 0; _ai < _tg_size.x; _ai++) {")
            self._emit(f"        {val_type} _av = {svar_vals}[_ai];")
            self._emit(f"        if (_av {cmp} _best_v) {{ _best_v = _av; _best_i = (int)_ai; }}")
            self._emit("    }")
            self._emit(f"    {svar_v}[0] = _best_v;")
            self._emit(f"    {svar_i}[0] = _best_i;")
            self._emit("}")
            self._emit(self.emitter.barrier())

            var_v = self._fresh_var("argv")
            var_i = self._fresh_var("argi")
            self._emit(f"{val_type} {var_v} = {svar_v}[0];")
            self._emit(f"int {var_i} = {svar_i}[0];")

            self._set_val(val_result, TType(dtype=val_dtype), var_v)
            self._set_val(idx_result, TType(dtype='i32'), var_i)

    def _gen_tt_dot(self, op: Op):
        """Generate tiled matrix multiply.

        If inputs are tiles (shared memory), uses them directly.
        Otherwise, materializes per-thread values into shared memory first.
        """
        self._flush_barrier()  # MMA reads from shared memory
        """
        """
        a_name = op.operands[0]
        b_name = op.operands[1]
        c_name = op.operands[2] if len(op.operands) > 2 else None
        result = op.results[0]

        a_val = self._get_val(a_name)
        b_val = self._get_val(b_name)

        a_shape = a_val.ttype.shape or [16, 16]
        b_shape = b_val.ttype.shape or [16, 16]
        BM = a_shape[0]
        BK = a_shape[1] if len(a_shape) > 1 else a_shape[0]
        BN = b_shape[1] if len(b_shape) > 1 else b_shape[0]

        in_dtype = a_val.ttype.dtype
        acc_dtype = op.result_types[0].dtype if op.result_types else in_dtype

        # Check if accumulator is a register accumulator
        if c_name and c_name in self._reg_tiles:
            reg = self._reg_tiles[c_name]
            a_tile = self._ensure_tile(a_name, [BM, BK], in_dtype)
            b_tile = self._ensure_tile(b_name, [BK, BN], in_dtype)
            self._gen_dot_into_reg_acc(a_tile, b_tile, reg, in_dtype, BM, BN, BK)
            self._reg_tiles[result] = reg
            result_type = op.result_types[0] if op.result_types else a_val.ttype
            self._set_val(result, result_type, "0 /*reg_acc*/")
            return

        in_type = self.emitter.map_dtype(in_dtype)
        acc_type = self.emitter.map_dtype(acc_dtype)

        # Ensure inputs are tiles
        a_tile = self._ensure_tile(a_name, [BM, BK], in_dtype)
        b_tile = self._ensure_tile(b_name, [BK, BN], in_dtype)

        # Check if accumulator is a zero constant (skip shared memory round-trip)
        zero_acc = False
        if c_name and c_name in self._deferred_tiles:
            _, _, fill_val, _ = self._deferred_tiles[c_name]
            if fill_val in ('0.000000e+00', '0', '0.0'):
                zero_acc = True
        elif c_name and c_name in self._const_fill:
            if self._const_fill[c_name] in ('0.000000e+00', '0', '0.0'):
                zero_acc = True

        # Ensure accumulator tile exists
        c_tile = None
        if c_name and not zero_acc and self._is_tile(c_name):
            c_tile = self._get_tile(c_name)

        use_hw = (self.emitter.supports_simd_matrix()
                  and BM % 8 == 0 and BN % 8 == 0 and BK % 8 == 0)

        # Allocate output tile (or reuse accumulator)
        if c_tile and c_tile.shape == [BM, BN]:
            out_tile = c_tile  # accumulate in-place
        elif zero_acc and use_hw:
            # For zero-init with HW MMA: allocate tile but skip filling —
            # simdgroup matrices will be initialized to zero directly
            out_tile = self._alloc_tile([BM, BN], acc_dtype)
        else:
            out_tile = self._alloc_tile([BM, BN], acc_dtype)
            # Initialize from accumulator
            if c_tile:
                self._emit_tile_copy(out_tile, c_tile)
            elif c_name:
                c_expr = self._get_expr(c_name)
                self._emit_tile_loop(out_tile.total,
                    f"{out_tile.shared_name}[_fi] = ({acc_type}){c_expr};")
            else:
                self._emit_tile_loop(out_tile.total,
                    f"{out_tile.shared_name}[_fi] = ({acc_type})0;")

        if use_hw:
            self._gen_dot_simdgroup_tile(a_tile, b_tile, out_tile,
                                          in_dtype, acc_dtype, BM, BN, BK,
                                          zero_acc=zero_acc)
        else:
            # Flush any pending tile loops (e.g. zero-init of accumulator)
            # before the scalar dot product, which uses direct _emit() calls.
            # Without this, a deferred zero-init can end up fused into a later
            # tile loop AFTER the accumulation, zeroing out the result.
            self._flush_fused_loops()
            self._gen_dot_scalar_tile(a_tile, b_tile, out_tile,
                                       in_type, acc_type, BM, BN, BK)

        self._register_tile(result, out_tile)

    def _ensure_tile(self, ssa_name: str, shape: list[int], dtype: str) -> TileInfo:
        """Ensure an SSA value is in a shared memory tile. Materialize if needed."""
        if self._is_tile(ssa_name):
            return self._get_tile(ssa_name)

        # Materialize per-thread value to shared memory
        val = self._get_val(ssa_name)
        tile = self._alloc_tile(shape, dtype)
        total = tile.total
        tid = self.emitter.thread_id_expr()
        metal_type = self.emitter.map_dtype(dtype)

        # Grid-stride fill for portability across GPU thread limits
        self._emit(f"for (uint _idx = (uint){tid}; _idx < {total}u; _idx += _tg_size.x)")
        self._emit(f"    {tile.shared_name}[_idx] = ({metal_type}){val.expr};")
        self._emit(self.emitter.barrier())
        self._tiles[ssa_name] = tile
        return tile

    def _emit_tile_copy(self, dst: TileInfo, src: TileInfo):
        """Copy one tile to another."""
        assert dst.total == src.total
        metal_type = self.emitter.map_dtype(dst.dtype)
        self._emit_tile_loop(dst.total,
            f"{dst.shared_name}[_fi] = ({metal_type}){self._tile_read(src, '_fi')};")

    def _gen_dot_simdgroup_tile(self, a_tile: TileInfo, b_tile: TileInfo,
                                  c_tile: TileInfo, in_dtype: str, acc_dtype: str,
                                  BM: int, BN: int, BK: int, zero_acc: bool = False):
        """SIMD group matrix multiply-accumulate on shared memory tiles."""
        tid = self.emitter.thread_id_expr()
        threads = min(self.block_size, self.MAX_THREADS)
        NUM_SG = max(1, threads // 32)
        num_blocks_n = BN // 8
        num_blocks_total = (BM // 8) * num_blocks_n
        blocks_per_sg = max(1, (num_blocks_total + NUM_SG - 1) // NUM_SG)

        # Check if B is a transposed view (for Q @ K^T pattern)
        b_transposed = b_tile.transposed_from is not None
        b_src = b_tile.transposed_from if b_transposed else b_tile
        # For transposed B: orig is [N_orig, K_orig], B^T is [K_orig, N_orig]
        # simdgroup_load with transpose flag reads from original layout
        b_stride = b_src.cols if b_transposed else BN

        sg_id = self._fresh_var("sg")
        self._emit(f"uint {sg_id} = (uint){tid} / 32u;")

        acc_mat_type = self.emitter.simd_matrix_type(acc_dtype, 8, 8)
        sg_a_type = self.emitter.simd_matrix_type(in_dtype, 8, 8)

        # Unique names for this dot instance
        sg_c = self._fresh_var("sgC")
        sg_a = self._fresh_var("sgA")
        sg_b = self._fresh_var("sgB")

        # Initialize accumulators
        self._emit(f"{acc_mat_type} {sg_c}[{blocks_per_sg}];")
        if zero_acc:
            # Zero-init directly — skip shared memory round-trip
            bi = self._fresh_var("bi")
            self._emit(f"for (uint {bi} = 0; {bi} < {blocks_per_sg}u; {bi}++)")
            self._emit(f"    {sg_c}[{bi}] = {acc_mat_type}(0.0f);")
        else:
            # Load accumulators from c_tile
            bi = self._fresh_var("bi")
            blk1 = self._fresh_var("blk")
            self._emit(f"for (uint {bi} = 0; {bi} < {blocks_per_sg}u; {bi}++) {{")
            self.indent += 1
            self._emit(f"uint {blk1} = {sg_id} * {blocks_per_sg}u + {bi};")
            self._emit(f"if ({blk1} < {num_blocks_total}u) {{")
            self._emit(f"    uint _br = {blk1} / {num_blocks_n}u, _bc = {blk1} % {num_blocks_n}u;")
            self._emit(f"    {self.emitter.simd_load(f'{sg_c}[{bi}]', f'&{c_tile.shared_name}[_br * {8 * BN}u + _bc * 8u]', f'{BN}ul')}")
            self._emit("}")
            self.indent -= 1
            self._emit("}")

        # MMA: iterate over K in steps of 8
        bi2 = self._fresh_var("bi")
        blk2 = self._fresh_var("blk")
        self._emit(f"for (uint {bi2} = 0; {bi2} < {blocks_per_sg}u; {bi2}++) {{")
        self.indent += 1
        self._emit(f"uint {blk2} = {sg_id} * {blocks_per_sg}u + {bi2};")
        self._emit(f"if ({blk2} < {num_blocks_total}u) {{")
        self.indent += 1
        self._emit(f"uint _br = {blk2} / {num_blocks_n}u, _bc = {blk2} % {num_blocks_n}u;")
        kk = self._fresh_var("kk")
        self._emit(f"for (uint {kk} = 0; {kk} < {BK}u; {kk} += 8u) {{")
        self.indent += 1
        self._emit(f"{sg_a_type} {sg_a}, {sg_b};")
        self._emit(self.emitter.simd_load(sg_a,
                   f"&{a_tile.shared_name}[_br * {8 * BK}u + {kk}]", f"{BK}ul"))
        if b_transposed:
            self._emit(self.emitter.simd_load(sg_b,
                       f"&{b_src.shared_name}[_bc * {8 * b_stride}u + {kk}]",
                       f"{b_stride}ul", transpose=True))
        else:
            self._emit(self.emitter.simd_load(sg_b,
                       f"&{b_tile.shared_name}[{kk} * {BN}u + _bc * 8u]", f"{BN}ul"))
        self._emit(self.emitter.simd_multiply_accumulate(
                   f"{sg_c}[{bi2}]", sg_a, sg_b, f"{sg_c}[{bi2}]"))
        self.indent -= 1
        self._emit("}")
        self.indent -= 1
        self._emit("}")
        self.indent -= 1
        self._emit("}")

        # Store accumulators back to c_tile
        bi3 = self._fresh_var("bi")
        blk3 = self._fresh_var("blk")
        self._emit(f"for (uint {bi3} = 0; {bi3} < {blocks_per_sg}u; {bi3}++) {{")
        self.indent += 1
        self._emit(f"uint {blk3} = {sg_id} * {blocks_per_sg}u + {bi3};")
        self._emit(f"if ({blk3} < {num_blocks_total}u) {{")
        self._emit(f"    uint _br = {blk3} / {num_blocks_n}u, _bc = {blk3} % {num_blocks_n}u;")
        self._emit(f"    {self.emitter.simd_store(f'{sg_c}[{bi3}]', f'&{c_tile.shared_name}[_br * {8 * BN}u + _bc * 8u]', f'{BN}ul')}")
        self._emit("}")
        self.indent -= 1
        self._emit("}")
        self._emit_barrier_direct()

    def _gen_dot_into_reg_acc(self, a_tile: TileInfo, b_tile: TileInfo,
                               reg: RegAccInfo, in_dtype: str,
                               BM: int, BN: int, BK: int):
        """MMA from shared memory tiles into persistent simdgroup register accumulator.

        Used for FA2's O += P @ V where O stays in registers across loop iterations.
        A=[BM,BK], B=[BK,BN], reg accumulator=[BM,BN].
        """
        b_transposed = b_tile.transposed_from is not None
        b_src = b_tile.transposed_from if b_transposed else b_tile
        b_stride = b_src.cols if b_transposed else BN

        sg_a_type = self.emitter.simd_matrix_type(in_dtype, 8, 8)
        sg_a = self._fresh_var("sgA")
        sg_b = self._fresh_var("sgB")

        bi = self._fresh_var("bi")
        blk = self._fresh_var("blk")
        self._emit(f"for (uint {bi} = 0; {bi} < {reg.blocks_per_sg}u; {bi}++) {{")
        self.indent += 1
        self._emit(f"uint {blk} = {reg.sg_var} * {reg.blocks_per_sg}u + {bi};")
        self._emit(f"if ({blk} < {reg.num_blocks_total}u) {{")
        self.indent += 1
        self._emit(f"uint _br = {blk} / {reg.num_blocks_n}u, _bc = {blk} % {reg.num_blocks_n}u;")
        kk = self._fresh_var("kk")
        self._emit(f"for (uint {kk} = 0; {kk} < {BK}u; {kk} += 8u) {{")
        self.indent += 1
        self._emit(f"{sg_a_type} {sg_a}, {sg_b};")
        self._emit(self.emitter.simd_load(sg_a,
                   f"&{a_tile.shared_name}[_br * {8 * BK}u + {kk}]", f"{BK}ul"))
        if b_transposed:
            self._emit(self.emitter.simd_load(sg_b,
                       f"&{b_src.shared_name}[_bc * {8 * b_stride}u + {kk}]",
                       f"{b_stride}ul", transpose=True))
        else:
            self._emit(self.emitter.simd_load(sg_b,
                       f"&{b_tile.shared_name}[{kk} * {BN}u + _bc * 8u]", f"{BN}ul"))
        self._emit(self.emitter.simd_multiply_accumulate(
                   f"{reg.reg_name}[{bi}]", sg_a, sg_b, f"{reg.reg_name}[{bi}]"))
        self.indent -= 1
        self._emit("}")
        self.indent -= 1
        self._emit("}")
        self.indent -= 1
        self._emit("}")
        # Use lazy barrier: reg-acc MMA only reads shared memory (A/B tiles),
        # never writes it.  The barrier can be deferred until the next shared
        # memory write (e.g. cooperative tile load in the next loop iteration).
        self._barrier_pending = True

    def _gen_dot_scalar_tile(self, a_tile: TileInfo, b_tile: TileInfo,
                               c_tile: TileInfo, in_type: str, acc_type: str,
                               BM: int, BN: int, BK: int):
        """Scalar matrix multiply-accumulate on shared memory tiles."""
        tid = self.emitter.thread_id_expr()
        C_ELEMS = BM * BN

        # Handle transposed B: B^T[k,n] = B_orig[n,k] = sB_orig[n * origCols + k]
        b_transposed = b_tile.transposed_from is not None
        b_src = b_tile.transposed_from if b_transposed else b_tile
        if b_transposed:
            b_elem = f"{b_src.shared_name}[_cc * {b_src.cols}u + {{kk}}]"
        else:
            b_elem = f"{b_tile.shared_name}[{{kk}} * {BN}u + _cc]"

        kk = self._fresh_var("dk")
        a_elem = f"{a_tile.shared_name}[_cr * {BK}u + {kk}]"
        b_elem_kk = b_elem.format(kk=kk)
        accum = f"{c_tile.shared_name}[_fi] += ({acc_type}){a_elem} * ({acc_type}){b_elem_kk};"

        # Grid-stride loop for portability across GPU thread limits
        self._emit(f"for (uint _fi = (uint){tid}; _fi < {C_ELEMS}u; _fi += _tg_size.x) {{")
        self._emit(f"    uint _cr = _fi / {BN}u, _cc = _fi % {BN}u;")
        self._emit(f"    for (uint {kk} = 0; {kk} < {BK}u; {kk}++) {{")
        self._emit(f"        {accum}")
        self._emit("    }")
        self._emit("}")
        self._emit(self.emitter.barrier())

    def _gen_tt_trans(self, op: Op):
        src = op.operands[0]
        result = op.results[0]
        src_val = self._get_val(src)
        result_type = op.result_types[0] if op.result_types else src_val.ttype

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            if src_tile.rank == 2:
                new_shape = [src_tile.shape[1], src_tile.shape[0]]
                trans_tile = TileInfo(
                    shared_name=src_tile.shared_name,
                    shape=new_shape,
                    dtype=src_tile.dtype,
                    transposed_from=src_tile,
                )
                self._tiles[result] = trans_tile
                self._set_val(result, TType(dtype=src_tile.dtype, shape=new_shape), src_val.expr)
                # Track backing ref so source tile isn't freed while this view lives
                backing = self._real_backing(trans_tile)
                self._backing_refs.setdefault(backing, set()).add(result)
                return

        self._set_val(result, result_type, src_val.expr)

    def _gen_tt_reshape(self, op: Op):
        """Handle tt.reshape: change tensor shape without moving data."""
        src = op.operands[0]
        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else self._get_val(src).ttype

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            new_shape = list(result_type.shape) if result_type.shape else [src_tile.total]
            reshaped = TileInfo(
                shared_name=src_tile.shared_name,
                shape=new_shape,
                dtype=src_tile.dtype,
            )
            self._tiles[result] = reshaped
            self._set_val(result, result_type, self._get_val(src).expr)
            backing = self._real_backing(reshaped)
            self._backing_refs.setdefault(backing, set()).add(result)
        else:
            self._set_val(result, result_type, self._get_expr(src))

    def _gen_tt_clampf(self, op: Op):
        """Handle tt.clampf: clamp float to [min, max]."""
        src = op.operands[0]
        lo = op.operands[1]
        hi = op.operands[2]
        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else self._get_val(src).ttype

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            lo_expr = self._tile_or_index_read(lo, self._get_tile(lo) if self._is_tile(lo) else None)
            hi_expr = self._tile_or_index_read(hi, self._get_tile(hi) if self._is_tile(hi) else None)
            target_type = self.emitter.map_dtype(result_type.dtype)
            out_tile = self._try_reuse_in_place(op.operands, list(src_tile.shape), result_type.dtype)
            if out_tile is None:
                out_tile = self._alloc_tile(list(src_tile.shape), result_type.dtype)
            src_read = self._tile_read(src_tile, "_fi")
            body = f"{out_tile.shared_name}[_fi] = ({target_type})min(max({src_read}, {lo_expr}), {hi_expr});"
            self._emit_tile_loop(out_tile.total, body)
            self._register_tile(result, out_tile)
        else:
            src_expr = self._get_expr(src)
            lo_expr = self._get_expr(lo)
            hi_expr = self._get_expr(hi)
            target_type = self.emitter.map_dtype(result_type.dtype)
            var = self._fresh_var("v")
            self._emit(f"{target_type} {var} = min(max({src_expr}, {lo_expr}), {hi_expr});")
            self._set_val(result, result_type, var)

    def _gen_tt_int_to_ptr(self, op: Op):
        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else TType(dtype='i64', is_ptr=True)
        self._set_val(result, result_type, self._get_expr(op.operands[0]))

    def _gen_tt_ptr_to_int(self, op: Op):
        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else TType(dtype='i64')
        src_expr = self._get_expr(op.operands[0])
        var = self._fresh_var("v")
        self._emit(f"long {var} = (long){src_expr};")
        self._set_val(result, result_type, var)

    def _gen_tt_mulhiui(self, op: Op):
        """Handle tt.mulhiui: upper 32 bits of unsigned integer multiply."""
        lhs = op.operands[0]
        rhs = op.operands[1]
        result = op.results[0]
        lhs_tile = self._is_tile(lhs)
        rhs_tile = self._is_tile(rhs)

        if lhs_tile or rhs_tile:
            lhs_t = self._get_tile(lhs) if lhs_tile else None
            rhs_t = self._get_tile(rhs) if rhs_tile else None
            ref_tile = lhs_t or rhs_t
            result_type = op.result_types[0] if op.result_types else self._get_val(lhs).ttype
            out_tile = self._alloc_tile(list(ref_tile.shape), result_type.dtype)
            lhs_r = self._tile_or_index_read(lhs, lhs_t)
            rhs_r = self._tile_or_index_read(rhs, rhs_t)
            body = f"{out_tile.shared_name}[_fi] = (int)mulhi((uint)({lhs_r}), (uint)({rhs_r}));"
            self._emit_tile_loop(out_tile.total, body)
            self._register_tile(result, out_tile)
            return

        lhs_val = self._get_val(lhs)
        rhs_val = self._get_val(rhs)
        result_type = op.result_types[0] if op.result_types else lhs_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("v")
        self._emit(f"{target_type} {var} = ({target_type})mulhi((uint)({lhs_val.expr}), (uint)({rhs_val.expr}));")
        self._set_val(result, result_type, var)

    def _gen_tt_fp_to_fp(self, op: Op):
        """Handle tt.fp_to_fp: float precision conversion (e.g., fp16 <-> fp32)."""
        self._gen_cast(op)

    def _gen_tt_bitcast(self, op: Op):
        """Handle tt.bitcast (Triton-level, delegates to arith.bitcast logic)."""
        self._gen_arith_bitcast(op)

    def _gen_tt_scan(self, op: Op):
        """Handle tt.scan (prefix sum/scan). Currently 1D only."""
        # tt.scan appears as a generic form: "tt.scan"(%input) <{axis=0, reverse=false}> ({ body })
        # For 1D tensors this is an inclusive prefix scan.
        # We implement it with a threadgroup shared memory parallel scan.
        src = op.operands[0] if op.operands else None
        result = op.results[0] if op.results else None
        axis = int(op.attrs.get('axis', '0'))

        # Determine scan op from body (attrs or raw_text)
        scan_op = 'add'
        raw = op.raw_text.lower() if op.raw_text else ''
        if 'maxnumf' in raw or 'maximumf' in raw:
            scan_op = 'max'
        elif 'minnumf' in raw or 'minimumf' in raw:
            scan_op = 'min'

        result_type = op.result_types[0] if op.result_types else self._get_val(src).ttype

        if self._is_tile(src):
            self._gen_tt_scan_tile(op, src, result, scan_op, axis, result_type)
            return

        # 1D per-thread scan: use threadgroup shared memory
        src_expr = self._get_expr(src)
        target_type = self.emitter.map_dtype(result_type.dtype)
        tid = self.emitter.thread_id_expr()
        n = result_type.shape[0] if result_type.shape else self.block_size

        svar = self._fresh_var("scan_s")
        self._emit(self.emitter.shared_memory_decl(svar, result_type.dtype, n))
        self._emit(f"{svar}[(uint){tid}] = {src_expr};")
        self._emit(self.emitter.barrier())

        # Hillis-Steele parallel inclusive scan
        step = self._fresh_var("step")
        self._emit(f"for (uint {step} = 1u; {step} < {n}u; {step} *= 2u) {{")
        self.indent += 1
        val = self._fresh_var("sv")
        if scan_op == 'add':
            self._emit(f"{target_type} {val} = ((uint){tid} >= {step}) ? {svar}[(uint){tid} - {step}] : ({target_type})0;")
        elif scan_op == 'max':
            self._emit(f"{target_type} {val} = ((uint){tid} >= {step}) ? {svar}[(uint){tid} - {step}] : {svar}[(uint){tid}];")
        else:
            self._emit(f"{target_type} {val} = ((uint){tid} >= {step}) ? {svar}[(uint){tid} - {step}] : ({target_type})0;")
        self._emit(self.emitter.barrier())
        if scan_op == 'add':
            self._emit(f"{svar}[(uint){tid}] += {val};")
        elif scan_op == 'max':
            self._emit(f"{svar}[(uint){tid}] = max({svar}[(uint){tid}], {val});")
        elif scan_op == 'min':
            self._emit(f"{svar}[(uint){tid}] = min({svar}[(uint){tid}], {val});")
        self._emit(self.emitter.barrier())
        self.indent -= 1
        self._emit("}")

        var = self._fresh_var("scan_r")
        self._emit(f"{target_type} {var} = {svar}[(uint){tid}];")
        self._set_val(result, result_type, var)

    def _gen_tt_scan_tile(self, op, src, result, scan_op, axis, result_type):
        """Scan for tile operands — copy to shared, run serial scan, copy back."""
        src_tile = self._get_tile(src)
        tid = self.emitter.thread_id_expr()

        out_tile = self._alloc_tile(list(src_tile.shape), result_type.dtype)
        # Copy source to output tile
        self._emit_tile_copy(out_tile, src_tile)
        self._emit(self.emitter.barrier())

        # Serial scan along axis (thread 0 does it, then broadcast)
        if len(src_tile.shape) == 1:
            inner = src_tile.shape[0]
            self._emit(f"if ((uint){tid} == 0u) {{")
            self.indent += 1
            i = self._fresh_var("si")
            if scan_op == 'add':
                self._emit(f"for (uint {i} = 1u; {i} < {inner}u; {i}++) {{ {out_tile.shared_name}[{i}] += {out_tile.shared_name}[{i} - 1u]; }}")
            elif scan_op == 'max':
                self._emit(f"for (uint {i} = 1u; {i} < {inner}u; {i}++) {{ {out_tile.shared_name}[{i}] = max({out_tile.shared_name}[{i}], {out_tile.shared_name}[{i} - 1u]); }}")
            elif scan_op == 'min':
                self._emit(f"for (uint {i} = 1u; {i} < {inner}u; {i}++) {{ {out_tile.shared_name}[{i}] = min({out_tile.shared_name}[{i}], {out_tile.shared_name}[{i} - 1u]); }}")
            self.indent -= 1
            self._emit("}")
        self._emit(self.emitter.barrier())
        self._register_tile(result, out_tile)

    def _gen_tt_debug_barrier(self, op: Op):
        self._emit(self.emitter.barrier())

    def _gen_tt_call(self, op: Op):
        raw = op.raw_text
        if 'zeros' in raw and op.result_types:
            result = op.results[0]
            result_type = op.result_types[0]
            t = self.emitter.map_dtype(result_type.dtype)
            self._set_val(result, result_type, f"({t})0")
        else:
            raise UnsupportedOperationError(op, "only the recognized zeros helper is supported")

    # --- CUDA libdevice -> Metal mapping for tt.extern_elementwise ---
    _LIBDEVICE_MAP: ClassVar[dict[str, str]] = {
        '__nv_erff': '_erf_approx',
        '__nv_erf': '_erf_approx',
        '__nv_erfcf': '_erfc_approx',
        '__nv_erfc': '_erfc_approx',
        '__nv_fabsf': 'abs',
        '__nv_sqrtf': 'sqrt',
        '__nv_rsqrtf': 'rsqrt',
        '__nv_expf': 'exp',
        '__nv_exp2f': 'exp2',
        '__nv_logf': 'log',
        '__nv_log2f': 'log2',
        '__nv_sinf': 'sin',
        '__nv_cosf': 'cos',
        '__nv_tanhf': 'tanh',
        '__nv_tanf': 'tan',
        '__nv_ceilf': 'ceil',
        '__nv_floorf': 'floor',
        '__nv_roundf': 'round',
        '__nv_fmaf': 'fma',
        '__nv_powf': 'pow',
        '__nv_copysignf': 'copysign',
        '__nv_fmodf': 'fmod',
        '__nv_fminf': 'fmin',
        '__nv_fmaxf': 'fmax',
    }

    def _gen_tt_extern_elementwise(self, op: Op):
        """Handle tt.extern_elementwise: map CUDA libdevice functions to Metal stdlib."""
        symbol = op.attrs.get('symbol', '')
        metal_fn = self._LIBDEVICE_MAP.get(symbol)

        if metal_fn is None:
            raise UnsupportedOperationError(op, f"unknown external symbol {symbol!r}")

        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else TType(dtype='f32')

        # Check if any operand is a tile
        has_tile = any(self._is_tile(o) for o in op.operands)

        if has_tile:
            self._gen_extern_elementwise_tile(op, metal_fn, result_type)
        else:
            operand_exprs = [self._get_expr(o) for o in op.operands]
            target_type = self.emitter.map_dtype(result_type.dtype)
            var = self._fresh_var("v")
            args = ', '.join(operand_exprs)
            self._emit(f"{target_type} {var} = {metal_fn}({args});")
            self._set_val(result, result_type, var)

    def _gen_extern_elementwise_tile(self, op: Op, metal_fn: str, result_type: TType):
        """Apply extern elementwise function to tile operands."""
        result = op.results[0]
        target_type = self.emitter.map_dtype(result_type.dtype)

        # Determine output shape from first tile operand
        out_shape = None
        for o in op.operands:
            if self._is_tile(o):
                t = self._get_tile(o)
                out_shape = list(t.shape)
                break
        if out_shape is None:
            out_shape = [result_type.shape[0]] if result_type.shape else [256]

        out_tile = self._try_reuse_in_place(op.operands, out_shape, result_type.dtype)
        if out_tile is None:
            out_tile = self._alloc_tile(out_shape, result_type.dtype)

        reads = []
        for o in op.operands:
            if self._is_tile(o):
                reads.append(self._tile_read(self._get_tile(o), "_fi"))
            else:
                reads.append(self._get_expr(o))
        args = ', '.join(reads)
        body = f"{out_tile.shared_name}[_fi] = ({target_type}){metal_fn}({args});"
        self._emit_tile_loop(out_tile.total, body)
        self._register_tile(result, out_tile)

    # --- Atomic operations ---

    def _gen_tt_atomic_rmw(self, op: Op):
        """Handle tt.atomic_rmw: atomic read-modify-write operations."""
        self._flush_fused_loops()
        if not self.emitter.supports_ptr_cast():
            raise UnsupportedOperationError(op, "atomics are not supported by this target")
        # Parse the atomic op kind from raw_text: "fadd, acq_rel, gpu, %ptr, %val, %mask : ..."
        raw = op.raw_text.strip()
        # Atomic op is the first word
        atomic_op_str = raw.split(',')[0].strip()
        ATOMIC_OPS = {
            'fadd': 'atomic_fetch_add_explicit',
            'add': 'atomic_fetch_add_explicit',
            'and': 'atomic_fetch_and_explicit',
            'or': 'atomic_fetch_or_explicit',
            'xor': 'atomic_fetch_xor_explicit',
            'max': 'atomic_fetch_max_explicit',
            'min': 'atomic_fetch_min_explicit',
            'umax': 'atomic_fetch_max_explicit',
            'umin': 'atomic_fetch_min_explicit',
            'xchg': 'atomic_exchange_explicit',
        }
        metal_fn = ATOMIC_OPS.get(atomic_op_str)
        if metal_fn is None:
            raise UnsupportedOperationError(op, f"unknown atomic operation {atomic_op_str!r}")

        # Operands: ptr_tensor, val_tensor, mask_tensor
        ptr_name = op.operands[0] if len(op.operands) > 0 else None
        val_name = op.operands[1] if len(op.operands) > 1 else None
        mask_name = op.operands[2] if len(op.operands) > 2 else None

        result_type = op.result_types[0] if op.result_types else TType(dtype='f32')
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("atom")
        tid = self.emitter.thread_id_expr()

        # Check for scatter pattern: ptr from tt.addptr with tile offset
        # (e.g., histogram: atomic_add to hist_ptr + bin_indices[tid])
        ptr_op = self._op_map.get(ptr_name)
        is_scatter = (ptr_op and ptr_op.opname == 'tt.addptr'
                      and self._is_tile(ptr_op.operands[1]))

        if is_scatter:
            base_expr = self._get_expr(ptr_op.operands[0])
            offset_tile = self._get_tile(ptr_op.operands[1])
            offset_read = self._tile_read(offset_tile, f'(uint){tid}')

            val_tile = self._get_tile(val_name) if self._is_tile(val_name) else None
            val_expr = self._tile_read(val_tile, f'(uint){tid}') if val_tile else self._get_expr(val_name)
            mask_expr = self._get_expr(mask_name) if mask_name else "true"

            self._emit(f"{target_type} {var} = ({target_type})0;")
            self._emit(f"if ({mask_expr}) {{")
            if atomic_op_str == 'fadd':
                self._emit(f"    volatile device atomic_float* _aptr = (volatile device atomic_float*)({base_expr} + {offset_read});")
                self._emit(f"    {var} = atomic_fetch_add_explicit(_aptr, {val_expr}, memory_order_relaxed);")
            elif atomic_op_str in ('add',):
                self._emit(f"    {var} = ({target_type}){metal_fn}((volatile device atomic_int*)({base_expr} + {offset_read}), ({target_type}){val_expr}, memory_order_relaxed);")
            else:
                self._emit(f"    {var} = ({target_type}){metal_fn}((volatile device atomic_uint*)({base_expr} + {offset_read}), as_type<uint>(({target_type}){val_expr}), memory_order_relaxed);")
            self._emit("}")
        else:
            ptr_expr = self._get_expr(ptr_name)
            val_expr = self._get_expr(val_name)
            mask_expr = self._get_expr(mask_name) if mask_name else "true"

            # Determine if this is a per-thread (tensor) or scalar atomic.
            # TTIR tensor atomic: result type is tensor<Nx...> → each thread operates on its own element
            # TTIR scalar atomic: result type is scalar → only thread 0 should execute
            result_type_is_tensor = (op.result_types and op.result_types[0].shape)
            if not result_type_is_tensor:
                # Also check raw text for tensor type signature
                raw = op.raw_text.strip() if hasattr(op, 'raw_text') and op.raw_text else ''
                result_type_is_tensor = 'tensor<' in raw.split('->')[-1] if '->' in raw else False

            if not result_type_is_tensor:
                # Scalar atomic (e.g. after tl.sum): only thread 0 should execute.
                # The value is the same across all threads, so having all threads
                # atomic-add would multiply the result by the number of threads.
                guard = f"(uint){tid} == 0u"
                if mask_expr != "true":
                    guard = f"{guard} && {mask_expr}"
            else:
                # Per-thread atomic (scatter pattern not detected by is_scatter
                # heuristic, but the tensor type tells us each thread has its own
                # ptr/val pair)
                guard = mask_expr

            if atomic_op_str == 'fadd':
                self._emit(f"{target_type} {var} = ({target_type})0;")
                self._emit(f"if ({guard}) {{")
                self._emit(f"    volatile device atomic_float* _aptr = (volatile device atomic_float*){ptr_expr};")
                self._emit(f"    {var} = atomic_fetch_add_explicit(_aptr, {val_expr}, memory_order_relaxed);")
                self._emit("}")
            else:
                self._emit(f"{target_type} {var} = ({target_type})0;")
                self._emit(f"if ({guard}) {{")
                self._emit(f"    {var} = {metal_fn}((volatile device atomic_uint*){ptr_expr}, as_type<uint>({val_expr}), memory_order_relaxed);")
                self._emit("}")

        self._set_val(op.results[0], result_type, var)

    # -----------------------------------------------------------------------
    # Arith ops — tile-aware
    # -----------------------------------------------------------------------

    def _gen_arith_constant(self, op: Op):
        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else TType(dtype='i32')
        val = None
        for k, v in op.attrs.items():
            if k == 'value':
                val = v
                break
        if val is None:
            raw = op.raw_text
            # Match dense<value> including negative infinity and scientific notation
            m = re.match(r'\s*dense<(0x[0-9A-Fa-f]+|-?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|inf))>', raw)
            if m:
                val = m.group(1)
                if val in ('0xFF800000', '0xFC00', '-inf'):
                    val = '-HUGE_VALF'
                elif val in ('0x7F800000', '0x7C00', 'inf'):
                    val = 'HUGE_VALF'
                elif val.startswith('0x'):
                    # Generic hex float constant — interpret based on dtype
                    hex_int = int(val, 16)
                    import struct as _struct
                    try:
                        if result_type and result_type.dtype == 'f16':
                            float_val = _struct.unpack('e', _struct.pack('H', hex_int))[0]
                        else:
                            float_val = _struct.unpack('f', _struct.pack('I', hex_int))[0]
                        import math as _math
                        if _math.isinf(float_val):
                            val = '-HUGE_VALF' if float_val < 0 else 'HUGE_VALF'
                        elif _math.isnan(float_val):
                            val = 'NAN'
                        else:
                            val = str(float_val)
                    except (ValueError, _struct.error):
                        val = '0'
            if val is None:
                # Match hex float constants (e.g., 0xFF800000 for -inf, 0x7F800000 for +inf)
                m = re.match(r'\s*(0x[0-9A-Fa-f]+)\s*:', raw)
                if m:
                    hex_val = m.group(1)
                    if hex_val == '0xFF800000':
                        val = '-HUGE_VALF'
                    elif hex_val == '0x7F800000':
                        val = 'HUGE_VALF'
                    elif hex_val == '0x7FC00000':
                        val = 'NAN'
                    else:
                        # Generic hex -> float via struct conversion at compile time
                        import struct
                        try:
                            float_val = struct.unpack('f', struct.pack('I', int(hex_val, 16)))[0]
                            val = str(float_val)
                        except (ValueError, struct.error):
                            val = hex_val
            if val is None:
                m = re.match(r'\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*:', raw)
                if m:
                    val = m.group(1)
            if val is None:
                if re.match(r'\s*true\b', raw):
                    val = 'true'
                elif re.match(r'\s*false\b', raw):
                    val = 'false'
            if val is None:
                m = re.match(r'%c(-?\d+)', result)
                if m:
                    val = m.group(1)
                else:
                    val = '0'

        result_type = op.result_types[0] if op.result_types else TType(dtype='i32')
        target_type = self.emitter.map_dtype(result_type.dtype)

        # Check if this is a dense tensor constant (1D or 2D)
        if result_type.is_tensor and result_type.shape and len(result_type.shape) >= 1:
            # Defer tile allocation — many dense constants (e.g., acc init, ptr advance)
            # are consumed by optimized matmul paths that generate their own code.
            # The tile is materialized on first _get_tile() or _is_tile() access.
            self._deferred_tiles[result] = (
                list(result_type.shape), result_type.dtype, val, target_type)
            # Remember the scalar fill value so loads can use it directly
            self._const_fill[result] = f"({target_type}){val}"
            # Set a placeholder SSA value so non-tile accesses work
            self._set_val(result, result_type, f"({target_type}){val}")
            return

        if result_type.is_tensor:
            self._set_val(result, result_type, f"({target_type}){val}")
        else:
            var = self._fresh_var("c")
            self._emit(f"{target_type} {var} = ({target_type}){val};")
            self._set_val(result, result_type, var)
            # Track scalar numeric value for peephole optimizations (e.g., rsqrt)
            if result_type.dtype in ('f16', 'f32', 'f64', 'bf16'):
                try:
                    self._scalar_const_value[result] = float(val)
                except (ValueError, TypeError):
                    pass

    def _gen_arith_index_cast(self, op: Op):
        self._gen_cast(op)

    def _gen_arith_index_castui(self, op: Op):
        self._gen_cast(op)

    def _gen_cast(self, op: Op):
        src = op.operands[0]
        result = op.results[0]
        src_val = self._get_val(src)
        result_type = op.result_types[0] if op.result_types else TType(dtype='i32')
        target_type = self.emitter.map_dtype(result_type.dtype)

        # Handle register accumulator: propagate reg_acc through the cast,
        # record the target dtype so _gen_tt_store_reg can emit a cast store.
        if src in self._reg_tiles:
            reg = self._reg_tiles[src]
            self._reg_tiles[result] = reg
            self._reg_acc_cast_dtype[result] = result_type.dtype
            self._set_val(result, result_type, "0 /*reg_acc*/")
            return

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            out_tile = self._alloc_tile(src_tile.shape, result_type.dtype)
            self._emit_tile_loop(out_tile.total,
                f"{out_tile.shared_name}[_fi] = ({target_type}){self._tile_read(src_tile, '_fi')};",
                needs_barrier=result not in self._deferred_cast_barriers)
            self._register_tile(result, out_tile)
            return

        if self._is_dual(src):
            row_expr, col_expr = self._get_dual(src)
            row_var = self._fresh_var("cast_row")
            col_var = self._fresh_var("cast_col")
            self._emit(f"{target_type} {row_var} = {self.emitter.cast_expr(target_type, row_expr)};")
            self._emit(f"{target_type} {col_var} = {self.emitter.cast_expr(target_type, col_expr)};")
            self._set_dual(result, row_var, col_var)
            self._set_val(result, result_type, row_var)
        else:
            var = self._fresh_var("cast")
            self._emit(f"{target_type} {var} = {self.emitter.cast_expr(target_type, src_val.expr)};")
            self._set_val(result, result_type, var)

    def _gen_binop(self, op: Op, msl_op: str):
        lhs = op.operands[0]
        rhs = op.operands[1]
        result = op.results[0]
        lhs_tile = self._is_tile(lhs)
        rhs_tile = self._is_tile(rhs)
        lhs_reg = lhs in self._reg_tiles
        rhs_reg = rhs in self._reg_tiles

        if lhs_tile or rhs_tile or lhs_reg or rhs_reg:
            self._gen_binop_tile(op, msl_op)
            return

        lhs_val = self._get_val(lhs)
        rhs_val = self._get_val(rhs)
        result_type = op.result_types[0] if op.result_types else lhs_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        lhs_dual = self._is_dual(lhs)
        rhs_dual = self._is_dual(rhs)

        if lhs_dual or rhs_dual:
            lhs_row, lhs_col = self._get_dual(lhs) if lhs_dual else (lhs_val.expr, lhs_val.expr)
            rhs_row, rhs_col = self._get_dual(rhs) if rhs_dual else (rhs_val.expr, rhs_val.expr)
            row_var = self._fresh_var("v_row")
            col_var = self._fresh_var("v_col")
            self._emit(f"{target_type} {row_var} = {lhs_row} {msl_op} {rhs_row};")
            self._emit(f"{target_type} {col_var} = {lhs_col} {msl_op} {rhs_col};")
            self._set_dual(result, row_var, col_var)
            self._set_val(result, result_type, row_var)
            # Propagate index expression through add/sub with scalar
            if msl_op in ('+', '-'):
                idx_op = None
                scalar_op = None
                if lhs in self._index_exprs and lhs_dual:
                    idx_op, scalar_op = lhs, rhs
                elif rhs in self._index_exprs and rhs_dual:
                    idx_op, scalar_op = rhs, lhs
                if idx_op is not None and scalar_op is not None:
                    base, start = self._index_exprs[idx_op]
                    s_expr = self._get_expr(scalar_op)
                    if base == "0":
                        new_base = s_expr if msl_op == '+' else f"(-({s_expr}))"
                    else:
                        new_base = f"({base} {msl_op} {s_expr})" if idx_op == lhs else f"({s_expr} {msl_op} {base})"
                    self._index_exprs[result] = (new_base, start)
        else:
            var = self._fresh_var("v")
            self._emit(f"{target_type} {var} = {lhs_val.expr} {msl_op} {rhs_val.expr};")
            self._set_val(result, result_type, var)
            # Propagate index expression through scalar add/sub
            if msl_op in ('+', '-'):
                idx_op = scalar_op = None
                if lhs in self._index_exprs and rhs not in self._index_exprs:
                    idx_op, scalar_op = lhs, rhs
                elif rhs in self._index_exprs and lhs not in self._index_exprs:
                    idx_op, scalar_op = rhs, lhs
                if idx_op is not None and scalar_op is not None:
                    base, start = self._index_exprs[idx_op]
                    s_expr = self._get_expr(scalar_op)
                    if base == "0":
                        new_base = s_expr if msl_op == '+' else f"(-({s_expr}))"
                    else:
                        new_base = f"({base} {msl_op} {s_expr})" if idx_op == lhs else f"({s_expr} {msl_op} {base})"
                    self._index_exprs[result] = (new_base, start)
                    if idx_op in self._expand_axis:
                        self._expand_axis[result] = self._expand_axis[idx_op]

    def _gen_binop_tile(self, op: Op, msl_op: str):
        """Binary op where at least one operand is a tile."""
        lhs = op.operands[0]
        rhs = op.operands[1]
        result = op.results[0]

        # Check for register accumulator operand (mulf/divf for O rescaling)
        reg_operand = None
        other_operand = None
        if lhs in self._reg_tiles:
            reg_operand, other_operand = lhs, rhs
        elif rhs in self._reg_tiles:
            reg_operand, other_operand = rhs, lhs

        if reg_operand is not None:
            if msl_op in ('*', '/'):
                reg = self._reg_tiles[reg_operand]
                # The other operand should be a broadcast from a 1D [BM] tile
                # (e.g., alpha_bc or l_bc). Find the source 1D tile.
                src_1d = self._find_broadcast_source_1d(other_operand)
                if src_1d is not None:
                    self._emit_reg_diag_op(reg, src_1d, msl_op)
                    self._reg_tiles[result] = reg
                    # Set dummy val for downstream SSA tracking
                    result_type = op.result_types[0] if op.result_types else self._get_val(lhs).ttype
                    self._set_val(result, result_type, "0 /*reg_acc*/")
                    return
            # Any op on reg_acc not handled above: materialize to shared tile first
            self._materialize_reg_acc(reg_operand)
            # Fall through to normal tile binop path

        # Deferred scalar multiplication: mulf(tile, splatted_scalar)
        # Instead of emitting a tile loop, attach the scale factor to the tile
        # so it's applied at read time. Eliminates one tile loop + barrier.
        if msl_op == '*':
            tile_operand = scalar_operand = None
            if self._is_tile(lhs) and not self._is_tile(rhs):
                tile_operand, scalar_operand = lhs, rhs
            elif self._is_tile(rhs) and not self._is_tile(lhs):
                tile_operand, scalar_operand = rhs, lhs
            if tile_operand is not None and scalar_operand is not None:
                tile = self._get_tile(tile_operand)
                # Only defer if tile is a simple tile (not a view)
                if tile.broadcast_src is None and tile.transposed_from is None:
                    scalar_expr = self._get_expr(scalar_operand)
                    # Compose with existing pending_scale if any
                    if tile.pending_scale is not None:
                        new_scale = f"({tile.pending_scale} * {scalar_expr})"
                    else:
                        new_scale = scalar_expr
                    scaled_tile = TileInfo(
                        shared_name=tile.shared_name,
                        shape=list(tile.shape),
                        dtype=tile.dtype,
                        pending_scale=new_scale,
                        is_register=tile.is_register,
                    )
                    self._tiles[result] = scaled_tile
                    result_type = op.result_types[0] if op.result_types else self._get_val(lhs).ttype
                    self._set_val(result, result_type, f"{self._tile_read(scaled_tile, '(uint)' + self.emitter.thread_id_expr())}")
                    backing = self._real_backing(scaled_tile)
                    self._backing_refs.setdefault(backing, set()).add(result)
                    return

        # In-place broadcast multiply/divide: tile * broadcast(1D_vec) or tile / broadcast(1D_vec)
        # Applies the operation directly to the tile, avoiding a full 2D allocation.
        # Used for per-channel scale after matmul: C[m,n] *= scale[n]
        if msl_op in ('*', '/') and self._is_tile(lhs) and self._is_tile(rhs):
            tile_op, bc_op = None, None
            bc_ssa = None
            lhs_t = self._get_tile(lhs)
            rhs_t = self._get_tile(rhs)
            # One must be a regular tile, the other a broadcast from 1D
            if lhs_t.broadcast_src is None and lhs_t.binop_view is None:
                src_1d = self._find_broadcast_source_1d(rhs)
                if src_1d is not None:
                    tile_op, bc_op, bc_ssa = lhs_t, rhs_t, rhs
            if tile_op is None and rhs_t.broadcast_src is None and rhs_t.binop_view is None:
                src_1d = self._find_broadcast_source_1d(lhs)
                if src_1d is not None:
                    tile_op, bc_op, bc_ssa = rhs_t, lhs_t, lhs

            if tile_op is not None and tile_op.rank == 2:
                src_1d = self._find_broadcast_source_1d(bc_ssa)
                result_type = op.result_types[0] if op.result_types else self._get_val(lhs).ttype
                target_type = self.emitter.map_dtype(result_type.dtype)
                # Emit in-place loop: tile[flat] op= src_1d[col_or_row]
                total = tile_op.total
                cols = tile_op.cols
                # Determine broadcast dimension (row or col)
                bc_src = bc_op.broadcast_src
                if bc_src and bc_src.rank == 2 and bc_src.shape[0] == 1:
                    # broadcast from [1, N] → scale is per-column
                    scale_read = self._tile_read(src_1d, f"_fi % {cols}u")
                elif bc_src and bc_src.rank == 2 and bc_src.shape[1] == 1:
                    # broadcast from [M, 1] → scale is per-row
                    scale_read = self._tile_read(src_1d, f"_fi / {cols}u")
                else:
                    scale_read = self._tile_read(src_1d, f"_fi % {src_1d.total}u")
                self._emit_tile_loop(total,
                    f"{tile_op.shared_name}[_fi] = ({target_type})({tile_op.shared_name}[_fi] {msl_op} {scale_read});")
                # Result aliases the modified tile
                result_tile = TileInfo(
                    shared_name=tile_op.shared_name,
                    shape=list(tile_op.shape),
                    dtype=tile_op.dtype,
                )
                self._tiles[result] = result_tile
                self._set_val(result, result_type, "0 /*inplace_bc_op*/")
                backing = self._real_backing(result_tile)
                self._backing_refs.setdefault(backing, set()).add(result)
                return

        result_type = op.result_types[0] if op.result_types else self._get_val(lhs).ttype
        target_type = self.emitter.map_dtype(result_type.dtype)

        lhs_tile = self._get_tile(lhs) if self._is_tile(lhs) else None
        rhs_tile = self._get_tile(rhs) if self._is_tile(rhs) else None

        # Determine output shape (largest shape)
        if lhs_tile and rhs_tile:
            out_shape = lhs_tile.shape if lhs_tile.total >= rhs_tile.total else rhs_tile.shape
        elif lhs_tile:
            out_shape = lhs_tile.shape
        else:
            out_shape = rhs_tile.shape

        # Virtual tile optimization: when both operands are broadcast views
        # (or one is a broadcast and the other is a small tile), defer the
        # computation to read time. Avoids allocating a full BM×BN tile for
        # index computations that may only be used by tt.addptr → tt.store.
        if (lhs_tile and rhs_tile
                and (lhs_tile.broadcast_src is not None or lhs_tile.binop_view is not None)
                and (rhs_tile.broadcast_src is not None or rhs_tile.binop_view is not None)):
            virt_tile = TileInfo(
                shared_name="__virtual__",
                shape=list(out_shape),
                dtype=result_type.dtype,
                binop_view=(lhs_tile, rhs_tile, msl_op),
            )
            self._tiles[result] = virt_tile
            self._set_val(result, result_type, "0 /*virtual_tile*/")
            # Track backing refs for BOTH source tiles
            for src_tile in (lhs_tile, rhs_tile):
                backing = self._real_backing(src_tile)
                if backing != "__virtual__":
                    self._backing_refs.setdefault(backing, set()).add(result)
            return

        # Try in-place reuse before allocating
        out_tile = self._try_reuse_in_place(op.operands, list(out_shape), result_type.dtype)
        if out_tile is None:
            out_tile = self._alloc_tile(list(out_shape), result_type.dtype)

        # Build read expressions (handles index expressions from tl.arange)
        lhs_read = self._tile_or_index_read(lhs, lhs_tile)
        rhs_read = self._tile_or_index_read(rhs, rhs_tile)

        total = out_tile.total
        body = f"{out_tile.shared_name}[_fi] = ({target_type})({lhs_read} {msl_op} {rhs_read});"
        self._emit_tile_loop(total, body)
        self._register_tile(result, out_tile)

    def _find_broadcast_source_1d(self, ssa_name: str) -> TileInfo | None:
        """Trace a broadcast tile back to its 1D source tile.

        For register accumulator rescaling, the 'other' operand is typically
        broadcast(expand_dims(alpha_1d)) -> [BM, d].  Return the 1D tile.
        """
        if ssa_name in self._tiles:
            tile = self._tiles[ssa_name]
            if tile.broadcast_src is not None:
                src = tile.broadcast_src
                if src.rank == 1:
                    return src
                # expand_dims: [BM, 1] from [BM] — trace further
                if src.rank == 2 and (src.shape[0] == 1 or src.shape[1] == 1):
                    # The expand_dims source might be tracked via broadcast_src chain
                    if src.broadcast_src and src.broadcast_src.rank == 1:
                        return src.broadcast_src
                    # Or via _op_map tracing
                    for t in self._tiles.values():
                        if t.shared_name == src.shared_name and t.rank == 1:
                            return t
        return None

    def _emit_reg_diag_op(self, reg: RegAccInfo, src_1d: TileInfo, msl_op: str):
        """Emit diagonal matrix multiply/divide for register accumulator.

        O = diag(alpha) @ O  (for msl_op='*')
        O = diag(1/l) @ O    (for msl_op='/')

        Fills ALL diagonal matrices at once into scratch (num_diags × 64
        elements), single barrier, then multiplies all blocks without
        intermediate barriers.  For BM=32 (4 diags) this reduces from
        3 barriers to 1.
        """
        # Only barrier if the source 1D tile has been modified since last barrier
        self._flush_barrier_if_dirty(src_1d.shared_name)
        BM_r = reg.shape[0]
        num_diags = BM_r // 8
        acc_type = self.emitter.simd_matrix_type(reg.dtype, 8, 8)
        tid = self.emitter.thread_id_expr()
        elem_size = 2 if reg.dtype in ('f16', 'bf16') else 4

        # Allocate scratch for ALL diags at once: num_diags * 64 elements.
        need_elems = num_diags * 64
        if self._diag_scratch is None:
            self._diag_scratch = self._fresh_var("diag")
            self._tile_decls.append(
                self.emitter.shared_memory_decl(self._diag_scratch, reg.dtype, need_elems))
            self._tg_bytes_allocated += need_elems * elem_size
            self._diag_scratch_size = need_elems
        elif hasattr(self, '_diag_scratch_size') and self._diag_scratch_size < need_elems:
            old_decl = self.emitter.shared_memory_decl(
                self._diag_scratch, reg.dtype, self._diag_scratch_size)
            new_decl = self.emitter.shared_memory_decl(
                self._diag_scratch, reg.dtype, need_elems)
            for j, d in enumerate(self._tile_decls):
                if d == old_decl:
                    self._tile_decls[j] = new_decl
                    break
            self._tg_bytes_allocated += (need_elems - self._diag_scratch_size) * elem_size
            self._diag_scratch_size = need_elems
        diag_name = self._diag_scratch

        metal_type = self.emitter.map_dtype(reg.dtype)
        if msl_op == '*':
            val_fn = lambda brr: f"{src_1d.shared_name}[{brr} * 8u + _r]"
        else:  # '/'
            # Clamp denominator to avoid inf/NaN for fully-masked softmax rows (l=0)
            val_fn = lambda brr: f"1.0f / max({src_1d.shared_name}[{brr} * 8u + _r], 1e-6f)"

        bi_v = self._fresh_var("bi")
        blk_v = self._fresh_var("blk")
        tmp_v = self._fresh_var("sg_tmp")
        diag_v = self._fresh_var("sg_diag")

        fill_total = num_diags * 64
        # Fill ALL diagonal matrices cooperatively in one loop
        self._emit(f"for (uint _di = (uint){tid}; _di < {fill_total}u; _di += _tg_size.x) {{")
        if num_diags > 1:
            self._emit("    uint _slot = _di / 64u;")
            self._emit("    uint _local = _di % 64u;")
            self._emit("    uint _r = _local / 8u, _c = _local % 8u;")
            val_expr = val_fn("_slot")
            self._emit(f"    {diag_name}[_di] = (_r == _c) ? ({metal_type}){val_expr} : ({metal_type})0;")
        else:
            self._emit("    uint _r = _di / 8u, _c = _di % 8u;")
            val_expr = val_fn("0")
            self._emit(f"    {diag_name}[_di] = (_r == _c) ? ({metal_type}){val_expr} : ({metal_type})0;")
        self._emit("}")
        self._emit(f"{self.emitter.barrier()}")

        # Multiply all blocks — each block reads from its corresponding diag slot
        self._emit(f"for (uint {bi_v} = 0; {bi_v} < {reg.blocks_per_sg}u; {bi_v}++) {{")
        self._emit(f"    uint {blk_v} = {reg.sg_var} * {reg.blocks_per_sg}u + {bi_v};")
        self._emit(f"    if ({blk_v} < {reg.num_blocks_total}u) {{")
        self._emit(f"        uint _brr = {blk_v} / {reg.num_blocks_n}u;")
        self._emit(f"        {acc_type} {diag_v}, {tmp_v};")
        self._emit(f"        {self.emitter.simd_load(diag_v, f'&{diag_name}[_brr * 64u]', '8ul')}")
        self._emit(f"        {self.emitter.simd_multiply(tmp_v, diag_v, f'{reg.reg_name}[{bi_v}]')}")
        self._emit(f"        {reg.reg_name}[{bi_v}] = {tmp_v};")
        self._emit("    }")
        self._emit("}")
        self._barrier_pending = False
        self._barrier_loop_size = 0
        self._dirty_tiles.clear()

    def _gen_arith_addi(self, op): self._gen_binop(op, '+')
    def _gen_arith_addf(self, op): self._gen_binop(op, '+')
    def _gen_arith_subi(self, op): self._gen_binop(op, '-')
    def _gen_arith_subf(self, op): self._gen_binop(op, '-')
    def _gen_arith_muli(self, op): self._gen_binop(op, '*')
    def _gen_arith_mulf(self, op): self._gen_binop(op, '*')
    def _gen_arith_divsi(self, op): self._gen_int_divrem(op, unsigned=False, remainder=False)
    def _gen_arith_divf(self, op):
        # Peephole: 1.0 / sqrt(x) → rsqrt(x)
        lhs, rhs = op.operands[0], op.operands[1]
        if not (self._is_tile(lhs) or self._is_tile(rhs) or
                lhs in self._reg_tiles or rhs in self._reg_tiles):
            rhs_op = self._op_map.get(rhs)
            if (self._scalar_const_value.get(lhs) == 1.0 and
                    rhs_op is not None and rhs_op.opname == 'math.sqrt'):
                # Emit rsqrt(sqrt_input) instead of 1.0 / sqrt(sqrt_input)
                sqrt_input_val = self._get_val(rhs_op.operands[0])
                result_type = op.result_types[0] if op.result_types else self._get_val(lhs).ttype
                target_type = self.emitter.map_dtype(result_type.dtype)
                func = self.emitter.math_func('rsqrt')
                var = self._fresh_var("v")
                self._emit(f"{target_type} {var} = {func}({sqrt_input_val.expr});")
                self._set_val(op.results[0], result_type, var)
                return
        self._gen_binop(op, '/')
    def _gen_arith_remsi(self, op): self._gen_int_divrem(op, unsigned=False, remainder=True)
    def _gen_arith_andi(self, op): self._gen_binop(op, '&')
    def _gen_arith_ori(self, op): self._gen_binop(op, '|')
    def _gen_arith_xori(self, op): self._gen_binop(op, '^')
    def _gen_arith_shli(self, op): self._gen_binop(op, '<<')
    def _gen_arith_shrsi(self, op): self._gen_binop(op, '>>')

    def _gen_int_divrem(self, op: Op, *, unsigned: bool, remainder: bool):
        """Share one quotient between matching integer division and remainder."""
        lhs, rhs = op.operands[:2]
        key = self._int_divrem_key(op, unsigned)
        can_pair = (
            key in self._int_divrem_pairs
            and not (self._is_tile(lhs) or self._is_tile(rhs))
            and lhs not in self._reg_tiles
            and rhs not in self._reg_tiles
            and not (self._is_dual(lhs) or self._is_dual(rhs))
        )
        if not can_pair:
            if not unsigned:
                self._gen_binop(op, '%' if remainder else '/')
                return
            self._gen_unsigned_int_divrem(op, remainder=remainder)
            return

        lhs_val = self._get_val(op.operands[0])
        rhs_val = self._get_val(op.operands[1])
        result_type = op.result_types[0] if op.result_types else lhs_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        cached = self._int_divrem_cache.get(key)
        quotient = cached[0] if cached else None
        rem = cached[1] if cached else None

        if quotient is None:
            quotient = self._fresh_var("quot")
            if unsigned:
                unsigned_type = self.emitter.map_dtype_unsigned(result_type.dtype)
                expr = (f"({target_type})(({unsigned_type}){lhs_val.expr} / "
                        f"({unsigned_type}){rhs_val.expr})")
            else:
                expr = f"{lhs_val.expr} / {rhs_val.expr}"
            self._emit(f"{target_type} {quotient} = {expr};")

        if remainder and rem is None:
            rem = self._fresh_var("rem")
            if unsigned:
                unsigned_type = self.emitter.map_dtype_unsigned(result_type.dtype)
                expr = (f"({target_type})(({unsigned_type}){lhs_val.expr} - "
                        f"({unsigned_type}){quotient} * ({unsigned_type}){rhs_val.expr})")
            else:
                expr = f"{lhs_val.expr} - {quotient} * {rhs_val.expr}"
            self._emit(f"{target_type} {rem} = {expr};")

        self._int_divrem_cache[key] = quotient, rem
        self._set_val(op.results[0], result_type, rem if remainder else quotient)

    def _gen_unsigned_int_divrem(self, op: Op, *, remainder: bool):
        lhs_val = self._get_val(op.operands[0])
        rhs_val = self._get_val(op.operands[1])
        result_type = op.result_types[0] if op.result_types else lhs_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        unsigned_type = self.emitter.map_dtype_unsigned(result_type.dtype)
        var = self._fresh_var("v")
        operator = '%' if remainder else '/'
        self._emit(f"{target_type} {var} = ({target_type})(({unsigned_type}){lhs_val.expr} {operator} ({unsigned_type}){rhs_val.expr});")
        self._set_val(op.results[0], result_type, var)

    def _gen_arith_divui(self, op: Op):
        self._gen_int_divrem(op, unsigned=True, remainder=False)

    def _gen_arith_remui(self, op: Op):
        self._gen_int_divrem(op, unsigned=True, remainder=True)

    def _gen_arith_remf(self, op: Op):
        lhs_val = self._get_val(op.operands[0])
        rhs_val = self._get_val(op.operands[1])
        result_type = op.result_types[0] if op.result_types else lhs_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("v")
        self._emit(f"{target_type} {var} = fmod({lhs_val.expr}, {rhs_val.expr});")
        self._set_val(op.results[0], result_type, var)

    def _gen_arith_shrui(self, op: Op):
        lhs_val = self._get_val(op.operands[0])
        rhs_val = self._get_val(op.operands[1])
        result_type = op.result_types[0] if op.result_types else lhs_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        unsigned_type = self.emitter.map_dtype_unsigned(result_type.dtype)
        var = self._fresh_var("v")
        self._emit(f"{target_type} {var} = ({target_type})(({unsigned_type}){lhs_val.expr} >> {rhs_val.expr});")
        self._set_val(op.results[0], result_type, var)

    def _gen_arith_negf(self, op: Op):
        src = op.operands[0]
        if self._is_tile(src):
            src_tile = self._get_tile(src)
            result_type = op.result_types[0] if op.result_types else TType(dtype=src_tile.dtype, shape=src_tile.shape)
            target_type = self.emitter.map_dtype(result_type.dtype)
            out_tile = self._try_reuse_in_place(op.operands, list(src_tile.shape), result_type.dtype)
            if out_tile is None:
                out_tile = self._alloc_tile(src_tile.shape, result_type.dtype)
            body = f"{out_tile.shared_name}[_fi] = ({target_type})(-{self._tile_read(src_tile, '_fi')});"
            self._emit_tile_loop(out_tile.total, body)
            self._register_tile(op.results[0], out_tile)
            return
        src_val = self._get_val(src)
        result_type = op.result_types[0] if op.result_types else src_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("v")
        self._emit(f"{target_type} {var} = -{src_val.expr};")
        self._set_val(op.results[0], result_type, var)

    def _gen_arith_maximumf(self, op): self._gen_binary_func(op, 'max')
    def _gen_arith_minimumf(self, op): self._gen_binary_func(op, 'min')
    def _gen_arith_maxnumf(self, op): self._gen_binary_func(op, 'max')
    def _gen_arith_minnumf(self, op): self._gen_binary_func(op, 'min')
    def _gen_arith_maxsi(self, op): self._gen_binary_func(op, 'max')
    def _gen_arith_minsi(self, op): self._gen_binary_func(op, 'min')
    def _gen_arith_maxui(self, op): self._gen_binary_func(op, 'max')
    def _gen_arith_minui(self, op): self._gen_binary_func(op, 'min')

    def _gen_arith_ceildivsi(self, op: Op):
        lhs = self._get_val(op.operands[0])
        rhs = self._get_val(op.operands[1])
        result_type = op.result_types[0] if op.result_types else lhs.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("v")
        # ceil(a/b) = (a + b - 1) / b for positive a and b
        self._emit(f"{target_type} {var} = ({lhs.expr} + {rhs.expr} - 1) / {rhs.expr};")
        self._set_val(op.results[0], result_type, var)

    def _gen_arith_ceildivui(self, op: Op):
        lhs = self._get_val(op.operands[0])
        rhs = self._get_val(op.operands[1])
        result_type = op.result_types[0] if op.result_types else lhs.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        unsigned_type = self.emitter.map_dtype_unsigned(result_type.dtype)
        var = self._fresh_var("v")
        self._emit(f"{target_type} {var} = ({target_type})((({unsigned_type}){lhs.expr} + ({unsigned_type}){rhs.expr} - 1u) / ({unsigned_type}){rhs.expr});")
        self._set_val(op.results[0], result_type, var)

    def _gen_binary_func(self, op: Op, func_name: str):
        lhs = op.operands[0]
        rhs = op.operands[1]
        lhs_tile = self._is_tile(lhs)
        rhs_tile = self._is_tile(rhs)

        if lhs_tile or rhs_tile:
            self._gen_binary_func_tile(op, func_name)
            return

        lhs_val = self._get_val(lhs)
        rhs_val = self._get_val(rhs)
        result_type = op.result_types[0] if op.result_types else lhs_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("v")
        self._emit(f"{target_type} {var} = {func_name}({lhs_val.expr}, {rhs_val.expr});")
        self._set_val(op.results[0], result_type, var)

    def _gen_binary_func_tile(self, op: Op, func_name: str):
        """Binary function (max, min) where at least one operand is a tile."""
        lhs = op.operands[0]
        rhs = op.operands[1]
        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else self._get_val(lhs).ttype
        target_type = self.emitter.map_dtype(result_type.dtype)

        lhs_tile = self._get_tile(lhs) if self._is_tile(lhs) else None
        rhs_tile = self._get_tile(rhs) if self._is_tile(rhs) else None

        if lhs_tile and rhs_tile:
            out_shape = lhs_tile.shape if lhs_tile.total >= rhs_tile.total else rhs_tile.shape
        elif lhs_tile:
            out_shape = lhs_tile.shape
        else:
            out_shape = rhs_tile.shape

        out_tile = self._try_reuse_in_place(op.operands, list(out_shape), result_type.dtype)
        if out_tile is None:
            out_tile = self._alloc_tile(list(out_shape), result_type.dtype)
        lhs_read = self._tile_or_index_read(lhs, lhs_tile)
        rhs_read = self._tile_or_index_read(rhs, rhs_tile)

        body = f"{out_tile.shared_name}[_fi] = ({target_type}){func_name}({lhs_read}, {rhs_read});"
        self._emit_tile_loop(out_tile.total, body)
        self._register_tile(result, out_tile)

    # --- Comparison ops ---

    def _gen_arith_cmpi(self, op: Op):
        lhs = op.operands[0]
        rhs = op.operands[1]

        pred_map = {
            'eq': '==', 'ne': '!=',
            'slt': '<', 'sle': '<=', 'sgt': '>', 'sge': '>=',
            'ult': '<', 'ule': '<=', 'ugt': '>', 'uge': '>=',
        }
        pred = ''
        m = re.match(r'\s*(\w+)\s*,', op.raw_text)
        if m and m.group(1) in pred_map:
            pred = m.group(1)
        cmp_op = pred_map.get(pred, '==')

        lhs_tile = self._is_tile(lhs)
        rhs_tile = self._is_tile(rhs)
        lhs_virt = lhs in self._virtual_tiles
        rhs_virt = rhs in self._virtual_tiles

        if lhs_tile or rhs_tile or lhs_virt or rhs_virt:
            lhs_t = self._get_tile(lhs) if lhs_tile else None
            rhs_t = self._get_tile(rhs) if rhs_tile else None
            # Determine output shape from tiles or virtual tiles
            if lhs_t or rhs_t:
                out_shape = (lhs_t or rhs_t).shape
            elif lhs_virt:
                out_shape = self._virtual_tiles[lhs][1]
            else:
                out_shape = self._virtual_tiles[rhs][1]
            out_tile = self._alloc_tile(list(out_shape), 'i1')
            # Read expressions: prefer tile, then virtual tile, then index expr, then scalar
            if lhs_virt:
                lhs_r = self._virtual_tiles[lhs][0]
            else:
                lhs_r = self._tile_or_index_read(lhs, lhs_t)
            if rhs_virt:
                rhs_r = self._virtual_tiles[rhs][0]
            else:
                rhs_r = self._tile_or_index_read(rhs, rhs_t)
            body = f"{out_tile.shared_name}[_fi] = ({lhs_r} {cmp_op} {rhs_r}) ? 1 : 0;"
            self._emit_tile_loop(out_tile.total, body)
            self._register_tile(op.results[0], out_tile)
            return

        # Handle tensor comparisons where operands come from make_range/splat
        # (not stored as tiles but vary per element — need tile result for
        # correct grid-stride loops where thread processes multiple elements)
        lhs_idx = lhs in self._index_exprs
        rhs_idx = rhs in self._index_exprs
        if lhs_idx or rhs_idx:
            lhs_val = self._get_val(lhs)
            rhs_val = self._get_val(rhs)
            shape = lhs_val.ttype.shape or rhs_val.ttype.shape or [self.block_size]
            out_tile = self._alloc_tile(list(shape), 'i1')

            def _idx_expr_i(ssa):
                if ssa in self._index_exprs:
                    base, start = self._index_exprs[ssa]
                    idx = "(int)_fi"
                    if start != 0:
                        idx = f"((int)_fi + {start})"
                    if base == "0":
                        return idx
                    return f"({base} + {idx})"
                return self._get_expr(ssa)

            lhs_r = _idx_expr_i(lhs)
            rhs_r = _idx_expr_i(rhs)
            body = f"{out_tile.shared_name}[_fi] = ({lhs_r} {cmp_op} {rhs_r}) ? 1 : 0;"
            self._emit_tile_loop(out_tile.total, body)
            self._register_tile(op.results[0], out_tile)
            return

        lhs_val = self._get_val(lhs)
        rhs_val = self._get_val(rhs)
        result_type = TType(dtype='i1', shape=lhs_val.ttype.shape)
        var = self._fresh_var("cmp")
        self._emit(f"bool {var} = {lhs_val.expr} {cmp_op} {rhs_val.expr};")
        self._set_val(op.results[0], result_type, var)

    def _gen_arith_cmpf(self, op: Op):
        """Float comparison — handle ordered/unordered predicates."""
        lhs = op.operands[0]
        rhs = op.operands[1]

        pred_map = {
            'oeq': '==', 'ogt': '>', 'oge': '>=',
            'olt': '<', 'ole': '<=', 'one': '!=',
            'eq': '==', 'ne': '!=', 'gt': '>', 'ge': '>=', 'lt': '<', 'le': '<=',
            # Unordered variants — same ops (Metal NaN semantics match)
            'ueq': '==', 'ugt': '>', 'uge': '>=',
            'ult': '<', 'ule': '<=', 'une': '!=',
        }
        pred = ''
        m = re.match(r'\s*(\w+)\s*,', op.raw_text)
        if m and m.group(1) in pred_map:
            pred = m.group(1)
        cmp_op = pred_map.get(pred, '==')

        lhs_tile = self._is_tile(lhs)
        rhs_tile = self._is_tile(rhs)

        if lhs_tile or rhs_tile:
            lhs_t = self._get_tile(lhs) if lhs_tile else None
            rhs_t = self._get_tile(rhs) if rhs_tile else None
            out_shape = (lhs_t or rhs_t).shape
            out_tile = self._alloc_tile(list(out_shape), 'i1')
            lhs_r = self._tile_or_index_read(lhs, lhs_t)
            rhs_r = self._tile_or_index_read(rhs, rhs_t)
            body = f"{out_tile.shared_name}[_fi] = ({lhs_r} {cmp_op} {rhs_r}) ? 1 : 0;"
            self._emit_tile_loop(out_tile.total, body)
            self._register_tile(op.results[0], out_tile)
            return

        lhs_val = self._get_val(lhs)
        rhs_val = self._get_val(rhs)
        result_type = TType(dtype='i1', shape=lhs_val.ttype.shape)
        var = self._fresh_var("cmp")
        self._emit(f"bool {var} = {lhs_val.expr} {cmp_op} {rhs_val.expr};")
        self._set_val(op.results[0], result_type, var)

    # --- Cast/conversion ops ---

    def _gen_arith_extsi(self, op): self._gen_cast(op)
    def _gen_arith_extui(self, op): self._gen_cast(op)
    def _gen_arith_extf(self, op): self._gen_cast(op)
    def _gen_arith_trunci(self, op): self._gen_cast(op)
    def _gen_arith_truncf(self, op): self._gen_cast(op)
    def _gen_arith_sitofp(self, op): self._gen_cast(op)
    def _gen_arith_uitofp(self, op): self._gen_cast(op)
    def _gen_arith_fptosi(self, op): self._gen_cast(op)
    def _gen_arith_fptoui(self, op): self._gen_cast(op)

    def _gen_arith_bitcast(self, op: Op):
        src = op.operands[0]
        result = op.results[0]
        result_type = op.result_types[0] if op.result_types else TType(dtype='i32')
        target_type = self.emitter.map_dtype(result_type.dtype)

        if self._is_tile(src):
            src_tile = self._get_tile(src)
            src_dtype = src_tile.dtype
            dst_dtype = result_type.dtype
            if src_dtype == dst_dtype:
                # Same type bitcast — just alias the tile
                self._register_tile(result, src_tile)
            else:
                out_tile = self._alloc_tile(list(src_tile.shape), dst_dtype)
                src_read = self._tile_read(src_tile, "_fi")
                body = f"{out_tile.shared_name}[_fi] = {self.emitter.bitcast_expr(target_type, src_read)};"
                self._emit_tile_loop(out_tile.total, body)
                self._register_tile(result, out_tile)
            return

        src_val = self._get_val(src)
        var = self._fresh_var("bc")
        # Pointer bitcast: keep as pointer (TTIR does ptr->int for atomic patterns)
        if src_val.ttype.is_ptr:
            # For atomic max/min patterns: the TTIR bitcasts ptr to int for address.
            # In MSL, keep as a pointer and pass through — the atomic code will cast.
            self._set_val(result, TType(dtype=result_type.dtype, is_ptr=True), src_val.expr)
            return
        self._emit(f"{target_type} {var} = {self.emitter.bitcast_expr(target_type, src_val.expr)};")
        self._set_val(result, result_type, var)

    def _gen_arith_select(self, op: Op):
        cond = op.operands[0]
        true_op = op.operands[1]
        false_op = op.operands[2]

        # Constant-fill tiles can use scalar values directly (avoid materializing)
        true_const = self._const_fill.get(true_op)
        false_const = self._const_fill.get(false_op)

        # Treat constant-fill operands as scalars, not tiles
        cond_is_tile = self._is_tile(cond)
        true_is_tile = self._is_tile(true_op) and true_const is None
        false_is_tile = self._is_tile(false_op) and false_const is None

        if cond_is_tile or true_is_tile or false_is_tile:
            # Tile-aware select
            cond_t = self._get_tile(cond) if cond_is_tile else None
            true_t = self._get_tile(true_op) if true_is_tile else None
            false_t = self._get_tile(false_op) if false_is_tile else None
            ref_tile = cond_t or true_t or false_t
            result_type = self._get_val(true_op).ttype
            target_type = self.emitter.map_dtype(result_type.dtype)
            # Try in-place reuse (exclude constant-fill operands from candidates)
            reuse_candidates = [o for o in op.operands if o not in self._const_fill]
            out_tile = self._try_reuse_in_place(reuse_candidates, list(ref_tile.shape), result_type.dtype)
            if out_tile is None:
                out_tile = self._alloc_tile(ref_tile.shape, result_type.dtype)
            cond_r = self._tile_or_index_read(cond, cond_t)
            true_r = true_const or self._tile_or_index_read(true_op, true_t)
            false_r = false_const or self._tile_or_index_read(false_op, false_t)
            body = f"{out_tile.shared_name}[_fi] = ({target_type})({cond_r} ? {true_r} : {false_r});"
            self._emit_tile_loop(out_tile.total, body)
            self._register_tile(op.results[0], out_tile)
            return

        cond_expr = self._get_expr(cond)
        true_expr = self._get_expr(true_op)
        false_expr = self._get_expr(false_op)
        result_type = self._get_val(true_op).ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("sel")
        self._emit(f"{target_type} {var} = {cond_expr} ? {true_expr} : {false_expr};")
        self._set_val(op.results[0], result_type, var)

    # --- Math ops ---

    def _gen_unary_func(self, op: Op, func_name: str):
        src = op.operands[0]

        if self._is_tile(src):
            self._gen_unary_func_tile(op, func_name)
            return

        src_val = self._get_val(src)
        result_type = op.result_types[0] if op.result_types else src_val.ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        func = self.emitter.math_func(func_name)
        var = self._fresh_var("v")
        self._emit(f"{target_type} {var} = {func}({src_val.expr});")
        self._set_val(op.results[0], result_type, var)

    def _gen_unary_func_tile(self, op: Op, func_name: str):
        """Unary function on a tile."""
        src = op.operands[0]
        result = op.results[0]
        src_tile = self._get_tile(src)
        result_type = op.result_types[0] if op.result_types else TType(dtype=src_tile.dtype, shape=src_tile.shape)
        target_type = self.emitter.map_dtype(result_type.dtype)
        func = self.emitter.math_func(func_name)

        out_tile = self._try_reuse_in_place(op.operands, list(src_tile.shape), result_type.dtype)
        if out_tile is None:
            out_tile = self._alloc_tile(src_tile.shape, result_type.dtype)
        body = f"{out_tile.shared_name}[_fi] = ({target_type}){func}({self._tile_read(src_tile, '_fi')});"
        self._emit_tile_loop(out_tile.total, body)
        self._register_tile(result, out_tile)

    def _gen_math_exp(self, op): self._gen_unary_func(op, 'exp')
    def _gen_math_exp2(self, op): self._gen_unary_func(op, 'exp2')
    def _gen_math_log(self, op): self._gen_unary_func(op, 'log')
    def _gen_math_log2(self, op): self._gen_unary_func(op, 'log2')
    def _gen_math_sqrt(self, op): self._gen_unary_func(op, 'sqrt')
    def _gen_math_rsqrt(self, op): self._gen_unary_func(op, 'rsqrt')
    def _gen_math_absf(self, op): self._gen_unary_func(op, 'abs')
    def _gen_math_absi(self, op): self._gen_unary_func(op, 'abs')
    def _gen_math_sin(self, op): self._gen_unary_func(op, 'sin')
    def _gen_math_cos(self, op): self._gen_unary_func(op, 'cos')
    def _gen_math_tanh(self, op): self._gen_unary_func(op, 'tanh')
    def _gen_math_tan(self, op): self._gen_unary_func(op, 'tan')
    def _gen_math_asin(self, op): self._gen_unary_func(op, 'asin')
    def _gen_math_acos(self, op): self._gen_unary_func(op, 'acos')
    def _gen_math_atan(self, op): self._gen_unary_func(op, 'atan')
    def _gen_math_floor(self, op): self._gen_unary_func(op, 'floor')
    def _gen_math_ceil(self, op): self._gen_unary_func(op, 'ceil')
    def _gen_math_round(self, op): self._gen_unary_func(op, 'round')
    def _gen_math_trunc(self, op): self._gen_unary_func(op, 'trunc')
    def _gen_math_log10(self, op): self._gen_unary_func(op, 'log10')
    def _gen_math_erf(self, op): self._gen_unary_func(op, '_erf_approx')

    def _gen_math_atan2(self, op): self._gen_binary_func(op, 'atan2')
    def _gen_math_copysign(self, op): self._gen_binary_func(op, 'copysign')
    def _gen_math_powf(self, op): self._gen_binary_func(op, 'pow')

    def _gen_math_fma(self, op: Op):
        a = self._get_expr(op.operands[0])
        b = self._get_expr(op.operands[1])
        c = self._get_expr(op.operands[2])
        result_type = self._get_val(op.operands[0]).ttype
        target_type = self.emitter.map_dtype(result_type.dtype)
        var = self._fresh_var("v")
        fma_fn = self.emitter.math_func('fma')
        self._emit(f"{target_type} {var} = {fma_fn}({a}, {b}, {c});")
        self._set_val(op.results[0], result_type, var)

    # -----------------------------------------------------------------------
    # SCF (Structured Control Flow) ops
    # -----------------------------------------------------------------------

    def _extract_matmul_params(self, op: Op, exclude_ptr_indices: set | None = None) -> dict:
        """Extract matmul role->arg_name mapping from the scf.for op structure.

        Instead of assuming a fixed ordering of scalar arguments, this traces
        the TTIR SSA chains to identify which func_arg plays each role:
          k_param:   loop bound (K dimension)
          a_base, b_base, c_base: pointer bases
          stride_am: A matrix row stride
          stride_bk: B matrix row stride
          stride_cm: C matrix row stride

        exclude_ptr_indices: set of func_arg indices to exclude from C
          candidate search (e.g., scale pointer index from fused_scale).
        """
        exclude = exclude_ptr_indices or set()
        ptr_args = [a for a in self.func_args if a.ttype.is_ptr]
        a_base = self._arg_name(ptr_args[0])
        b_base = self._arg_name(ptr_args[1])
        # C is the first pointer arg that isn't A, B, or an excluded arg
        # (e.g., the scale pointer for fused Q8 matmul).
        a_idx, b_idx = ptr_args[0].index, ptr_args[1].index
        c_func_arg = None
        for p in ptr_args:
            if p.index not in (a_idx, b_idx) and p.index not in exclude:
                c_func_arg = p
                break
        if c_func_arg is None:
            c_func_arg = ptr_args[-1]  # final fallback
        c_base = self._arg_name(c_func_arg)

        # 1. K: the upper bound of the scf.for loop
        k_param = None
        if op.loop_end and op.loop_end in self.ssa_map:
            k_param = self._get_expr(op.loop_end)

        # 2-3. stride_am, stride_bk: trace pointer-type iter_arg init chains
        #   iter_arg_inits may include non-pointer entries (e.g., offs_k for K offset
        #   tracking). Skip those and extract strides from the first two pointer inits.
        ptr_strides = []
        for init_ssa in op.iter_arg_inits:
            s = self._find_stride_arg_in_ptr_init(init_ssa)
            if s is not None:
                ptr_strides.append(s)
            if len(ptr_strides) == 2:
                break
        stride_am = ptr_strides[0] if len(ptr_strides) > 0 else None
        stride_bk = ptr_strides[1] if len(ptr_strides) > 1 else None

        # 4. stride_cm: trace from the C pointer's addptr chain
        #    Find the addptr that uses c_ptr (via splat) as base and extract
        #    the row stride from its offset operand.
        found_strides = {stride_am, stride_bk}
        stride_cm = None
        c_ptr_name = c_func_arg.name
        for sop in self._op_map.values():
            if sop.opname == 'tt.addptr':
                for oi, operand in enumerate(sop.operands):
                    base_op = self._op_map.get(operand)
                    if base_op and base_op.opname == 'tt.splat' and base_op.operands[0] == c_ptr_name:
                        # Found addptr(splat(c_ptr), offset) — extract stride from offset
                        offset_ssa = sop.operands[1 - oi]
                        s = self._search_stride_in_addi_chain(offset_ssa)
                        if s and s not in found_strides and s != k_param:
                            stride_cm = s
                            break
            if stride_cm:
                break

        # Fallback to positional if tracing failed
        scalar_args = [a for a in self.func_args if not a.ttype.is_ptr]
        if k_param is None:
            k_param = self._arg_name(scalar_args[0])
        if stride_am is None:
            stride_am = self._arg_name(scalar_args[1] if len(scalar_args) > 1 else scalar_args[0])
        if stride_bk is None:
            stride_bk = self._arg_name(scalar_args[2] if len(scalar_args) > 2 else scalar_args[0])
        if stride_cm is None:
            stride_cm = self._arg_name(scalar_args[3] if len(scalar_args) > 3 else scalar_args[0])

        return {
            'a_base': a_base, 'b_base': b_base, 'c_base': c_base,
            'k_param': k_param,
            'stride_am': stride_am, 'stride_bk': stride_bk, 'stride_cm': stride_cm,
            'c_out_dtype': c_func_arg.ttype.dtype,
            'a_dev_dtype': ptr_args[0].ttype.dtype,
        }

    def _find_stride_arg_in_ptr_init(self, ptr_init_ssa: str) -> str | None:
        """Trace from a tensor<MxNx!tt.ptr> init SSA back to find the row stride arg."""
        addptr_op = self._op_map.get(ptr_init_ssa)
        if not addptr_op or addptr_op.opname != 'tt.addptr':
            return None
        # The offset (non-base) operand contains the stride info.
        # The chain may have nested addptr (base+row_offset, then +col_offset),
        # so recurse through all operands.
        for offset_ssa in [addptr_op.operands[0], addptr_op.operands[1]]:
            result = self._search_stride_in_addi_chain(offset_ssa)
            if result:
                return result
            # Recurse through nested addptr (e.g., broadcast(addptr(...)))
            inner_op = self._op_map.get(offset_ssa)
            if inner_op and inner_op.opname in ('tt.broadcast', 'tt.expand_dims'):
                result = self._find_stride_arg_in_ptr_init(inner_op.operands[0])
                if result:
                    return result
            elif inner_op and inner_op.opname == 'tt.addptr':
                result = self._find_stride_arg_in_ptr_init(offset_ssa)
                if result:
                    return result
        return None

    def _search_stride_in_addi_chain(self, ssa: str) -> str | None:
        """Search through addi/broadcast/muli chains for a splatted scalar func_arg stride."""
        op = self._op_map.get(ssa)
        if not op:
            return None
        if op.opname == 'arith.addi':
            for operand in op.operands:
                result = self._search_stride_in_addi_chain(operand)
                if result:
                    return result
            return None
        # Use existing _find_stride_in_chain for broadcast/muli/splat tracing
        stride_info = self._find_stride_in_chain(ssa)
        if stride_info:
            _, stride_expr = stride_info
            return stride_expr
        return None

    def _acc_has_post_ops(self, acc_result: str, alias_ops: set, cast_ops: set) -> bool:
        """Check if the accumulator result feeds into real post-ops.

        Follows through TTGIR alias ops (convert_layout, local_alloc, etc.)
        and cast ops (truncf, extf) to see if the chain ends at tt.store.
        Returns True only if there are genuine post-ops (bias add, gelu, etc.).
        """
        # BFS through the consumer chain
        frontier = {acc_result}
        visited = set()
        while frontier:
            ssa = frontier.pop()
            if ssa in visited:
                continue
            visited.add(ssa)
            for sop in self._op_map.values():
                if ssa not in sop.operands:
                    continue
                if sop.opname == 'tt.store':
                    continue  # terminal — not a post-op
                if sop.opname in alias_ops:
                    # Follow through alias ops transparently
                    for r in sop.results:
                        frontier.add(r)
                    continue
                if sop.opname in cast_ops:
                    # Follow through cast ops (handled by needs_cast)
                    for r in sop.results:
                        frontier.add(r)
                    continue
                # Any other op consuming the acc is a real post-op
                return True
        return False

    def _detect_fused_scale_op(self, acc_result: str) -> dict | None:
        """Detect per-channel scale pattern: result → mulf(result, broadcast(1D_scale)).

        Returns dict with scale info if fusable, None otherwise.
        Pattern: arith.mulf(acc, broadcast(expand_dims(extf?(load(scale_ptrs)))))
        """
        # Find the mulf that uses acc_result
        mulf_op = None
        for sop in self._op_map.values():
            if sop.opname == 'arith.mulf' and acc_result in sop.operands:
                mulf_op = sop
                break
        if mulf_op is None:
            return None

        # Identify which operand is acc and which is scale
        scale_ssa = mulf_op.operands[1] if mulf_op.operands[0] == acc_result else mulf_op.operands[0]

        # Trace scale_ssa back through broadcast -> expand_dims -> extf? -> load
        scale_chain = []  # collect SSA names to skip
        scale_chain.append(mulf_op.results[0])  # mulf result

        cur = scale_ssa
        scale_arg_idx = None
        for _depth in range(10):  # max chain depth
            op = self._op_map.get(cur)
            if op is None:
                break
            scale_chain.append(cur)
            if op.opname == 'tt.broadcast' or op.opname == 'tt.expand_dims' or op.opname in ('arith.extf', 'arith.truncf'):
                cur = op.operands[0]
            elif op.opname == 'tt.load':
                # Found the scale load — trace its pointer to a func_arg
                ptr_ssa = op.operands[0]
                scale_arg_idx = self._trace_ptr_to_func_arg(ptr_ssa)
                break
            else:
                return None  # unexpected op in chain

        if scale_arg_idx is None:
            return None

        # Also collect the store that consumes the mulf result (possibly
        # through a truncf for f32→f16 output), plus all ops building the
        # store pointer. We mark the mulf result and any truncf; the C pointer
        # ops are harmless scalars that cost no threadgroup memory, so we let
        # them run.
        mulf_result = mulf_op.results[0]
        store_search_ssa = mulf_result
        # Follow through truncf and ttg.convert_layout to find the store.
        # TTGIR inserts convert_layout between truncf and tt.store.
        _PASSTHRU = {'arith.truncf', 'arith.extf', 'ttg.convert_layout'}
        for _ in range(5):
            found_next = False
            for sop in self._op_map.values():
                if sop.opname in _PASSTHRU and store_search_ssa in sop.operands:
                    scale_chain.append(sop.results[0])
                    store_search_ssa = sop.results[0]
                    found_next = True
                    break
            if not found_next:
                break
        # tt.store is a void op (no results) so it's not in _op_map.
        # Scan the flat ops list instead.
        for sop in self._all_ops:
            if sop.opname == 'tt.store' and store_search_ssa in sop.operands:
                # The store is already handled by _dot_store_emitted.
                # Collect the store's mask chain — these ops allocate
                # bool tiles that are never read when the store is fused.
                if len(sop.operands) >= 3:
                    self._collect_dead_store_chain(sop.operands[2], scale_chain)
                break

        # Also collect intermediate ops that build the scale pointer (load, addptr, etc.)
        # These all feed into the scale load and should be skipped to avoid allocating tiles
        self._collect_scale_chain_ops(scale_ssa, scale_chain)

        return {
            'scale_arg_idx': scale_arg_idx,
            'skip_ssas': set(scale_chain),
        }

    def _detect_fused_bias_op(self, acc_result: str) -> dict | None:
        """Detect per-column bias add pattern: result → addf(result, broadcast(1D_bias)).

        Optionally detects GELU after the bias add.
        Returns dict with bias info if fusable, None otherwise.
        Pattern: arith.addf(acc, broadcast(expand_dims(extf?(load(bias_ptrs)))))
        Optional GELU: the addf result feeds into a GELU chain → truncf → store
        """
        # Find the addf that uses acc_result (possibly through alias/cast chain)
        addf_op = None
        addf_acc_ssa = acc_result
        _ALIAS_OPS = {'ttg.convert_layout', 'ttg.local_alloc',
                      'ttg.local_load', 'ttg.memdesc_trans'}
        _CAST_OPS = {'arith.truncf', 'arith.extf'}
        visited = set()
        frontier = {acc_result}
        while frontier and addf_op is None:
            ssa = frontier.pop()
            if ssa in visited:
                continue
            visited.add(ssa)
            for sop in self._op_map.values():
                if ssa not in sop.operands:
                    continue
                if sop.opname == 'arith.addf':
                    addf_op = sop
                    addf_acc_ssa = ssa
                    break
                if sop.opname in _ALIAS_OPS | _CAST_OPS:
                    for r in sop.results:
                        frontier.add(r)
        if addf_op is None:
            return None

        # Identify which operand is acc and which is bias
        bias_ssa = addf_op.operands[1] if addf_op.operands[0] == addf_acc_ssa else addf_op.operands[0]

        # Trace bias_ssa back through broadcast -> expand_dims -> extf? -> load
        bias_chain = []
        bias_chain.append(addf_op.results[0])  # addf result

        cur = bias_ssa
        bias_arg_idx = None
        for _depth in range(10):
            op = self._op_map.get(cur)
            if op is None:
                break
            bias_chain.append(cur)
            if op.opname == 'tt.broadcast' or op.opname == 'tt.expand_dims' or op.opname in ('arith.extf', 'arith.truncf'):
                cur = op.operands[0]
            elif op.opname == 'tt.load':
                ptr_ssa = op.operands[0]
                bias_arg_idx = self._trace_ptr_to_func_arg(ptr_ssa)
                break
            else:
                return None

        if bias_arg_idx is None:
            return None

        # Check if there's a GELU after the addf
        # GELU pattern: x * 0.5 * (1 + erf(x * rsqrt2))
        # In TTIR: mulf(x, mulf(const_0.5, addf(const_1, extern_erf(mulf(x, const_rsqrt2)))))
        # We detect: the addf result feeds into mulf ops (GELU) rather than directly to store
        has_gelu = False
        has_silu = False
        gelu_chain = []
        addf_result = addf_op.results[0]

        # Find what consumes the addf result
        addf_consumers = []
        for sop in self._op_map.values():
            if addf_result in sop.operands:
                addf_consumers.append(sop)

        # Check if addf_result feeds into an activation (GELU, SiLU, etc.)
        # rather than directly to store/truncf. Activation starts with an
        # arithmetic op (mulf, subf, negf, divf) or math op (exp).
        _ACTIVATION_STARTERS = {'arith.mulf', 'arith.subf', 'arith.negf',
                                'arith.divf', 'math.exp'}
        activation_consumer = None
        for consumer in addf_consumers:
            if consumer.opname in _ACTIVATION_STARTERS:
                activation_consumer = consumer
                break

        if activation_consumer is not None:
            # GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
            # Collect all ops in the GELU chain for skipping.
            # Also detect SiLU = x / (1 + exp(-x)) or x * sigmoid(x).
            _ACTIVATION_OPS = {'arith.mulf', 'arith.addf', 'arith.divf',
                         'arith.truncf', 'arith.extf', 'arith.constant',
                         'arith.negf', 'arith.subf',
                         'tt.extern_elementwise', 'math.sqrt', 'math.erf',
                         'math.exp', 'math.exp2',
                         'ttg.convert_layout', 'ttg.local_alloc',
                         'ttg.local_load'}
            gelu_frontier = {addf_result}
            gelu_visited = set()
            found_erf = False
            found_exp = False
            while gelu_frontier:
                ssa = gelu_frontier.pop()
                if ssa in gelu_visited:
                    continue
                gelu_visited.add(ssa)
                gelu_chain.append(ssa)
                for sop in self._op_map.values():
                    if ssa not in sop.operands:
                        continue
                    if sop.opname in _ACTIVATION_OPS:
                        if sop.opname in ('math.erf', 'tt.extern_elementwise'):
                            found_erf = True
                        if sop.opname in ('math.exp', 'math.exp2'):
                            found_exp = True
                        for r in sop.results:
                            gelu_frontier.add(r)
            # Only fuse if we found a recognized activation pattern
            if found_erf:
                has_gelu = True  # GELU: contains erf
            elif found_exp:
                has_gelu = True  # SiLU: contains exp (sigmoid-based)
                has_silu = True
            else:
                # Not a recognized activation — fall back to post-ops path
                return None

        # Collect the store that consumes the final result
        if has_gelu:
            # Find the last SSA in the GELU chain that feeds a store
            for ssa in reversed(gelu_chain):
                for sop in self._op_map.values():
                    if sop.opname == 'tt.store' and ssa in sop.operands:
                        break
        else:
            # Collect truncf and ttg alias ops in the chain for skipping
            _follow_ops = {'arith.truncf', 'arith.extf', 'ttg.convert_layout',
                          'ttg.local_alloc', 'ttg.local_load', 'ttg.memdesc_trans'}
            frontier_store = {addf_result}
            visited_store = set()
            while frontier_store:
                ssa = frontier_store.pop()
                if ssa in visited_store:
                    continue
                visited_store.add(ssa)
                for sop2 in self._op_map.values():
                    if ssa not in sop2.operands:
                        continue
                    if sop2.opname in _follow_ops:
                        for r in sop2.results:
                            bias_chain.append(r)
                            frontier_store.add(r)

        # Collect bias pointer chain ops
        self._collect_scale_chain_ops(bias_ssa, bias_chain)
        # Add GELU chain
        bias_chain.extend(gelu_chain)

        return {
            'bias_arg_idx': bias_arg_idx,
            'has_gelu': has_gelu,
            'has_silu': has_silu,
            'skip_ssas': set(bias_chain),
        }

    def _trace_ptr_to_func_arg(self, ptr_ssa: str) -> int | None:
        """Trace a pointer SSA back to its func_arg index."""
        op = self._op_map.get(ptr_ssa)
        if op is None:
            return None
        if op.opname == 'tt.addptr':
            # Trace the base pointer
            return self._trace_ptr_to_func_arg(op.operands[0])
        if op.opname == 'tt.broadcast':
            # broadcast preserves the pointer origin — trace through
            return self._trace_ptr_to_func_arg(op.operands[0])
        if op.opname == 'tt.splat':
            # The inner operand should be a func_arg
            inner = op.operands[0]
            for arg in self.func_args:
                if arg.name == inner:
                    return arg.index
            return None
        return None

    def _collect_dead_store_chain(self, ssa: str, chain: list[str]):
        """Collect SSA names in a dead store's mask/ptr chain for skipping.

        Traces backward through mask construction ops (cmpi, broadcast, andi,
        splat) that only feed into the already-fused store. Stops at ops that
        might be shared with live code (e.g., make_range, expand_dims).
        """
        _MASK_OPS = {'arith.cmpi', 'arith.andi', 'tt.broadcast', 'tt.splat'}
        visited = set()
        stack = [ssa]
        while stack:
            s = stack.pop()
            if s in visited or s in chain:
                visited.add(s)
                continue
            visited.add(s)
            op = self._op_map.get(s)
            if op is None or op.opname not in _MASK_OPS:
                continue
            chain.append(s)
            stack.extend(op.operands)

    def _collect_scale_chain_ops(self, ssa: str, chain: list[str],
                                 _visited: set[str] | None = None):
        """Collect all SSA names in the scale computation chain for skipping."""
        if _visited is None:
            _visited = set()
        if ssa in _visited:
            return
        _visited.add(ssa)
        op = self._op_map.get(ssa)
        if op is None:
            return
        if ssa not in chain:
            chain.append(ssa)
        _SCALE_OPS = ('tt.broadcast', 'tt.expand_dims', 'arith.extf',
                      'arith.truncf', 'tt.load', 'tt.addptr', 'tt.splat',
                      'arith.addi', 'arith.cmpi', 'tt.make_range')
        for operand in op.operands:
            sub_op = self._op_map.get(operand)
            if sub_op and sub_op.opname in _SCALE_OPS:
                if operand not in chain:
                    chain.append(operand)
                self._collect_scale_chain_ops(operand, chain, _visited)

    def _gen_scf_for(self, op: Op):
        """Generate for loop. Detects dot-accumulator pattern for optimization."""
        dot_acc_idx = self._find_dot_acc_iter_arg(op)
        if dot_acc_idx is not None:
            dot_op = self._find_dot_op(op)
            has_reduce = any(o.opname == 'tt.reduce' for o in (op.body_ops or []))
            num_dots = sum(1 for o in (op.body_ops or []) if o.opname == 'tt.dot')
            # Standard matmul pattern requires 3+ iter_args (A_ptrs, B_ptrs, acc)
            # with pointer advancement in the loop body. Non-standard patterns
            # (e.g., attention with pointers computed from loop var) use generic path.
            has_ptr_iter_args = len(op.iter_arg_inits) >= 3
            # Detect dequant pattern: sitofp/uitofp feeding into dot operands.
            # This is the W8A16/W4A16 quantized matmul pattern — the optimized
            # path handles it by loading weights as int and converting in-place.
            dequant_info = self._detect_dequant_pattern(op, dot_op)
            # Check for extra data-modifying ops that are NOT part of a dequant chain
            extra_data_ops = {'arith.sitofp', 'arith.fptosi', 'arith.uitofp'}
            has_data_transform = False
            for o in (op.body_ops or []):
                if o.opname in extra_data_ops and o.results:
                    # Skip if this is the dequant cast that feeds into dot
                    if dequant_info and o.results[0] in (dequant_info.get('cast_result'),):
                        continue
                    has_data_transform = True
            # Also detect element-wise mulf/divf on loaded tiles (not the acc += pattern)
            mulf_on_load = False
            # Collect all cast results from dequant ops for chain detection
            dequant_casts = set()
            if dequant_info:
                dequant_casts.add(dequant_info['cast_result'])
            for o in (op.body_ops or []):
                if o.opname in extra_data_ops and o.results:
                    dequant_casts.add(o.results[0])
            if not has_data_transform:
                for o in (op.body_ops or []):
                    if o.opname == 'arith.mulf' and o.results:
                        # Skip if this mulf is part of a dequant chain (cast → mulf → dot)
                        if any(dc in o.operands for dc in dequant_casts):
                            continue
                        # Check if result feeds into tt.dot (modifying loaded tile)
                        for o2 in (op.body_ops or []):
                            if o2.opname == 'tt.dot' and o.results[0] in o2.operands[:2]:
                                mulf_on_load = True
                                break
            is_pure_matmul = not has_data_transform and not mulf_on_load
            if dot_op and not has_reduce and num_dots == 1 and has_ptr_iter_args and is_pure_matmul:
                # Simple matmul pattern: single dot with accumulation, no reductions
                a_shape = dot_op.result_types[0].shape if dot_op.result_types else None
                if a_shape and len(a_shape) == 2:
                    BM, BN = a_shape
                    # Extract BK and in_dtype from dot's type_str, falling back
                    # to operand type info when the walker leaves type_str empty.
                    bk_match = re.search(
                        r'tensor<(\d+)x(\d+)x\w+>\s*\*\s*tensor<(\d+)x(\d+)x',
                        dot_op.type_str)
                    BK = int(bk_match.group(2)) if bk_match else BM
                    in_dtype = 'f32'
                    dtype_m = re.search(r'tensor<\d+x\d+x(\w+)>\s*\*', dot_op.type_str)
                    if dtype_m:
                        in_dtype = dtype_m.group(1)
                    acc_dtype = dot_op.result_types[0].dtype if dot_op.result_types else 'f32'
                    # Fallback: infer BK and in_dtype from dot operand types
                    if not bk_match and len(dot_op.operands) >= 2:
                        a_opnd = dot_op.operands[0]
                        a_src = self._op_map.get(a_opnd)
                        if a_src and a_src.result_types:
                            a_rt = a_src.result_types[0]
                            if a_rt.shape and len(a_rt.shape) == 2:
                                BK = a_rt.shape[1]
                            if a_rt.dtype:
                                in_dtype = a_rt.dtype
                    # Override accumulator dtype to fp16 for native half-precision ALUs.
                    # This uses mad(half, half, half) instead of mad(float, float, float),
                    # giving ~2x throughput on GPUs with dedicated fp16 pipelines.
                    if self.force_acc_fp16 and in_dtype == 'f16':
                        acc_dtype = 'f16'
                    # Check if the accumulator result has post-loop modifications.
                    # If the result is used by ops other than tt.store, we need to
                    # keep it as a tile for those ops to modify.
                    acc_result = op.results[dot_acc_idx] if dot_acc_idx < len(op.results) else None
                    has_post_ops = False
                    fused_scale_info = None
                    _TTGIR_ALIAS_OPS = {'ttg.convert_layout', 'ttg.local_alloc',
                                        'ttg.local_load', 'ttg.memdesc_trans'}
                    _CAST_OPS = {'arith.truncf', 'arith.extf'}
                    if acc_result:
                        has_post_ops = self._acc_has_post_ops(
                            acc_result, _TTGIR_ALIAS_OPS, _CAST_OPS)
                    can_simd = (self.emitter.supports_simd_matrix()
                                and BM % 8 == 0 and BN % 8 == 0 and BK % 8 == 0)
                    # Detect fusable per-channel scale: result → mulf(result, broadcast(1D))
                    # If detected, fold scale into the store (avoids sC for large tiles)
                    # Detect fusable per-channel scale and per-column bias+activation
                    fused_bias_info = None
                    fused_scale_info = None
                    if has_post_ops and acc_result:
                        fused_scale_info = self._detect_fused_scale_op(acc_result)
                        if fused_scale_info:
                            has_post_ops = False  # Will fuse scale into store
                            self._fused_scale_skip = fused_scale_info['skip_ssas']
                    # Detect fusable per-column bias add: result → addf(result, broadcast(1D))
                    # Optionally followed by GELU or SiLU activation.
                    # Works for both simdgroup and scalar paths.
                    if has_post_ops and acc_result:
                        fused_bias_info = self._detect_fused_bias_op(acc_result)
                        if fused_bias_info:
                            has_post_ops = False  # Will fuse bias (+ optional activation) into store
                            self._fused_scale_skip = fused_bias_info['skip_ssas']
                    b_load_dtype = dequant_info['device_dtype'] if dequant_info else None
                    dequant_scale = None
                    if dequant_info and 'scale_ssa' in dequant_info:
                        try:
                            dequant_scale = self._get_expr(dequant_info['scale_ssa'])
                        except KeyError:
                            pass  # scale not resolvable (e.g. per-column vector)
                    # If dequant has an unresolvable scale, fall back to generic path
                    if dequant_info and 'scale_ssa' in dequant_info and dequant_scale is None:
                        self._gen_scf_for_generic(op)
                        return
                    if can_simd:
                        self._gen_scf_for_dot_optimized(op, dot_acc_idx, BM, BN, BK,
                                                         in_dtype, acc_dtype,
                                                         has_post_ops=has_post_ops,
                                                         b_load_dtype=b_load_dtype,
                                                         fused_scale=fused_scale_info,
                                                         fused_bias=fused_bias_info,
                                                         dequant_scale=dequant_scale)
                    else:
                        self._gen_scf_for_dot_scalar(op, dot_acc_idx, BM, BN, BK,
                                                      in_dtype, acc_dtype,
                                                      has_post_ops=has_post_ops,
                                                      b_load_dtype=b_load_dtype,
                                                      dequant_scale=dequant_scale,
                                                      fused_bias=fused_bias_info,
                                                      fused_scale=fused_scale_info)
                    return

        self._gen_scf_for_generic(op)

    def _gen_scf_for_dot_optimized(self, op: Op, dot_acc_idx: int,
                                    BM: int, BN: int, BK: int,
                                    in_dtype: str, acc_dtype: str,
                                    has_post_ops: bool = False,
                                    b_load_dtype: str | None = None,
                                    fused_scale: dict | None = None,
                                    fused_bias: dict | None = None,
                                    dequant_scale: str | None = None):
        """Optimized matmul loop: register tiling, no sC, direct device store.

        Key optimizations over the previous version:
        1. No sC in threadgroup memory — accumulators stay in simdgroup registers
        2. Register tiling: each simdgroup computes TM*TN 8x8 blocks with A/B reuse
        3. Double buffering for A/B tiles when memory allows
        4. Direct simdgroup_store to device memory (or sC for post-ops)

        b_load_dtype: if set (e.g. 'i8'), B weights are loaded from device as this
            dtype and dequantized (cast) to in_dtype in threadgroup memory. This
            supports W8A16 quantized matmul patterns.
        """
        THREADS = min(self.block_size, self.MAX_THREADS)
        A_ELEMS = BM * BK
        B_ELEMS = BK * BN
        C_ELEMS = BM * BN
        A_LOADS = max(1, (A_ELEMS + THREADS - 1) // THREADS)
        B_LOADS = max(1, (B_ELEMS + THREADS - 1) // THREADS)
        C_STORES = max(1, (C_ELEMS + THREADS - 1) // THREADS)

        tid = self.emitter.thread_id_expr()
        in_type = self.emitter.map_dtype(in_dtype)
        acc_type = self.emitter.map_dtype(acc_dtype)

        _excl = set()
        if fused_scale:
            _excl.add(fused_scale['scale_arg_idx'])
        if fused_bias:
            _excl.add(fused_bias['bias_arg_idx'])
        params = self._extract_matmul_params(op, exclude_ptr_indices=_excl)
        a_base = params['a_base']
        b_base = params['b_base']
        c_base = params['c_base']
        k_param = params['k_param']
        stride_am = params['stride_am']
        stride_bk = params['stride_bk']
        stride_cm = params['stride_cm']

        # L2-aware swizzled block scheduling: reorder threadgroup dispatch
        # so nearby groups share A/B tiles in L2 cache.
        # Only apply when there are no post-ops — post-ops (bias, relu, etc.)
        # use the original program_id values baked into the TTIR, which would
        # be inconsistent with swizzled PIDs.
        tgid_x = self._pid_exprs.get('x', f'(int){self.emitter.threadgroup_id_expr("x")}')
        tgid_y = self._pid_exprs.get('y', f'(int){self.emitter.threadgroup_id_expr("y")}')
        if has_post_ops:
            pid_m = tgid_x
            pid_n = tgid_y
        else:
            SWIZZLE_GROUP = 8
            grid_n_expr = f'(int){self.emitter.grid_dim_expr("y")}'
            pid_m = "_sw_pid_m"
            pid_n = "_sw_pid_n"
            grid_m_expr = f'(int){self.emitter.grid_dim_expr("x")}'
            self._emit(f"int _num_pid_m = {grid_m_expr};")
            self._emit(f"int _num_pid_n = {grid_n_expr};")
            self._emit(f"int {pid_m}, {pid_n};")
            self._emit("{")
            self._emit(f"    int _pid_linear = {tgid_x} * _num_pid_n + {tgid_y};")
            self._emit(f"    int _num_pid_in_group = {SWIZZLE_GROUP} * _num_pid_n;")
            self._emit("    int _group_id = _pid_linear / _num_pid_in_group;")
            self._emit("    int _within = _pid_linear % _num_pid_in_group;")
            self._emit(f"    int _first_pid_m = _group_id * {SWIZZLE_GROUP};")
            self._emit(f"    int _group_sz_m = min(_num_pid_m - _first_pid_m, {SWIZZLE_GROUP});")
            self._emit(f"    {pid_m} = _first_pid_m + _within % _group_sz_m;")
            self._emit(f"    {pid_n} = _within / _group_sz_m;")
            self._emit("}")

        c_out_dtype = params['c_out_dtype']
        c_out_type = self.emitter.map_dtype(c_out_dtype)
        needs_cast = (c_out_dtype != acc_dtype)

        sizeof_in = 2 if in_dtype in ('f16', 'bf16') else 4
        sizeof_acc = 2 if acc_dtype in ('f16', 'bf16') else 4
        TGMEM_LIMIT = 32768
        # vec4 loads: 4 halves at a time, requires f16 and cols divisible by 4
        # Disable vec4 for B when loading from a different device dtype (dequant)
        use_vec4_a = (in_dtype == 'f16' and BK % 4 == 0)
        use_vec4_b = (in_dtype == 'f16' and BN % 4 == 0 and b_load_dtype is None)

        NUM_SG = max(1, THREADS // 32)
        BLOCKS_M = BM // 8
        BLOCKS_N = BN // 8
        num_blocks_total = BLOCKS_M * BLOCKS_N

        # --- Compute register tiling dimensions (TM x TN per simdgroup) ---
        # Constraint: (BLOCKS_M / TM) * (BLOCKS_N / TN) == NUM_SG
        # When NUM_SG > num_blocks_total (e.g. 32x32 tile → 16 blocks, 32 SGs),
        # some SGs will be idle. Use effective_sg = min(NUM_SG, num_blocks_total).
        effective_sg = min(NUM_SG, num_blocks_total)
        blocks_per_sg = max(1, num_blocks_total // effective_sg)
        # Choose TM/TN for balanced register reuse (prefer TM ≈ TN)
        TM, TN = 1, 1  # default: each SG handles one block
        found_tiling = False
        best_ratio = float('inf')
        for tm in range(1, blocks_per_sg + 1):
            if blocks_per_sg % tm == 0:
                tn = blocks_per_sg // tm
                if (BLOCKS_M % tm == 0 and BLOCKS_N % tn == 0
                        and (BLOCKS_M // tm) * (BLOCKS_N // tn) == effective_sg):
                    # Prefer balanced (ratio closest to 1), tiebreak: higher TN
                    ratio = max(tm, tn) / max(min(tm, tn), 1)
                    if not found_tiling or ratio < best_ratio or (ratio == best_ratio and tn > TN):
                        TM, TN = tm, tn
                        best_ratio = ratio
                        found_tiling = True
        if not found_tiling:
            # Fallback: TM=1, TN=1, linear assignment with bounds check
            TM, TN = 1, 1
        SUPER_N = max(1, BLOCKS_N // TN)
        needs_sg_guard = (NUM_SG > num_blocks_total) or (not found_tiling)

        # --- Decide whether to use sC (needed for post-ops or type cast) ---
        # Without sC, double buffering has more room for larger A/B tiles
        use_sC = has_post_ops
        cast_temp_bytes = NUM_SG * 64 * sizeof_acc if (needs_cast or fused_scale or fused_bias) else 0
        fused_scale_bytes = BN * sizeof_acc if fused_scale else 0
        fused_bias_bytes = BN * sizeof_acc if fused_bias else 0
        if use_sC:
            # With sC: double buffer check includes sC
            double_buf_bytes = 2 * (A_ELEMS + B_ELEMS) * sizeof_in + C_ELEMS * sizeof_acc
        else:
            # Without sC: only A/B tiles + cast temp + fused scale buffer
            double_buf_bytes = 2 * (A_ELEMS + B_ELEMS) * sizeof_in + cast_temp_bytes + fused_scale_bytes + fused_bias_bytes
        # Account for pre-existing tile allocations (constant dense tiles, etc.)
        available_tgmem = TGMEM_LIMIT - self._tg_bytes_allocated
        use_double_buf = (not has_post_ops) and (double_buf_bytes <= available_tgmem)

        sg_id = self._fresh_var("sg_id")
        self._emit(f"uint {sg_id} = (uint){tid} / 32u;")

        # --- Shared memory allocation ---
        if use_sC:
            sc = self._fresh_var("sC")
            self._emit(self.emitter.shared_memory_decl(sc, acc_dtype, C_ELEMS))

        if use_double_buf:
            sa0 = self._fresh_var("sA0")
            sa1 = self._fresh_var("sA1")
            sb0 = self._fresh_var("sB0")
            sb1 = self._fresh_var("sB1")
            self._emit(self.emitter.shared_memory_decl(sa0, in_dtype, A_ELEMS))
            self._emit(self.emitter.shared_memory_decl(sa1, in_dtype, A_ELEMS))
            self._emit(self.emitter.shared_memory_decl(sb0, in_dtype, B_ELEMS))
            self._emit(self.emitter.shared_memory_decl(sb1, in_dtype, B_ELEMS))
            self._emit(f"threadgroup {in_type}* _sA_cur = {sa0};")
            self._emit(f"threadgroup {in_type}* _sA_nxt = {sa1};")
            self._emit(f"threadgroup {in_type}* _sB_cur = {sb0};")
            self._emit(f"threadgroup {in_type}* _sB_nxt = {sb1};")
            sa_cur, sb_cur = "_sA_cur", "_sB_cur"
            sa_nxt, sb_nxt = "_sA_nxt", "_sB_nxt"
        else:
            sa = self._fresh_var("sA")
            sb = self._fresh_var("sB")
            self._emit(self.emitter.shared_memory_decl(sa, in_dtype, A_ELEMS))
            self._emit(self.emitter.shared_memory_decl(sb, in_dtype, B_ELEMS))

        # Cast temp buffer (per-SG, only when !use_sC and needs_cast or fused_scale/bias)
        if not use_sC and (needs_cast or fused_scale or fused_bias):
            s_cast = self._fresh_var("sCast")
            self._emit(self.emitter.shared_memory_decl(s_cast, acc_dtype, NUM_SG * 64))

        # Fused scale: allocate small shared buffer for scale vector [BN]
        if fused_scale:
            s_scale = self._fresh_var("sScale")
            self._emit(self.emitter.shared_memory_decl(s_scale, acc_dtype, BN))

        # Fused bias: allocate small shared buffer for bias vector [BN]
        s_bias = None
        if fused_bias:
            s_bias = self._fresh_var("sBias")
            self._emit(self.emitter.shared_memory_decl(s_bias, acc_dtype, BN))

        # --- Map simdgroup to its TM*TN output region ---
        self._emit(f"int _sg_row = (int){sg_id} / {SUPER_N};")
        self._emit(f"int _sg_col = (int){sg_id} % {SUPER_N};")
        self._emit(f"int _base_br = _sg_row * {TM};")
        self._emit(f"int _base_bc = _sg_col * {TN};")
        if needs_sg_guard:
            self._emit(f"bool _sg_active = (_base_br < {BLOCKS_M} && _base_bc < {BLOCKS_N});")

        # --- Register accumulators: TM * TN simdgroup_matrix ---
        acc_mat_type = self.emitter.simd_matrix_type(acc_dtype, 8, 8)
        sg_a_type = self.emitter.simd_matrix_type(in_dtype, 8, 8)
        self._emit(f"{acc_mat_type} _sg_acc[{TM}][{TN}];")
        self._emit(f"for (int _ti = 0; _ti < {TM}; _ti++)")
        self._emit(f"    for (int _tj = 0; _tj < {TN}; _tj++)")
        self._emit(f"        _sg_acc[_ti][_tj] = {acc_mat_type}(0);")

        iv = self._fresh_var("iv")

        if use_double_buf:
            # --- Double-buffered K-loop ---
            self._emit_cooperative_tile_load(
                sa_cur, a_base, stride_am, BM, BK, A_ELEMS, A_LOADS, THREADS,
                row_offset=f"(int){pid_m} * {BM}", col_offset="0",
                vec4=use_vec4_a, in_dtype=in_dtype)
            self._emit_cooperative_tile_load(
                sb_cur, b_base, stride_bk, BK, BN, B_ELEMS, B_LOADS, THREADS,
                row_offset="0", col_offset=f"(int){pid_n} * {BN}",
                vec4=use_vec4_b, in_dtype=in_dtype,
                device_dtype=b_load_dtype or '')
            if dequant_scale:
                self._emit_dequant_scale_pass(sb_cur, B_ELEMS, dequant_scale, in_dtype)
            self._emit(self.emitter.barrier())

            self._emit(f"for (int {iv} = 0; {iv} < {k_param}; {iv} += {BK}) {{")
            self.indent += 1

            # Prefetch next K-block
            self._emit(f"if ({iv} + {BK} < {k_param}) {{")
            self.indent += 1
            self._emit_cooperative_tile_load(
                sa_nxt, a_base, stride_am, BM, BK, A_ELEMS, A_LOADS, THREADS,
                row_offset=f"(int){pid_m} * {BM}", col_offset=f"({iv} + {BK})",
                vec4=use_vec4_a, in_dtype=in_dtype)
            self._emit_cooperative_tile_load(
                sb_nxt, b_base, stride_bk, BK, BN, B_ELEMS, B_LOADS, THREADS,
                row_offset=f"({iv} + {BK})", col_offset=f"(int){pid_n} * {BN}",
                vec4=use_vec4_b, in_dtype=in_dtype,
                device_dtype=b_load_dtype or '')
            if dequant_scale:
                self._emit_dequant_scale_pass(sb_nxt, B_ELEMS, dequant_scale, in_dtype)
            self.indent -= 1
            self._emit("}")

            # MMA with register tiling: load A[TM] and B[TN], do TM*TN MACs
            if needs_sg_guard:
                self._emit("if (_sg_active) {")
                self.indent += 1
            kk = self._fresh_var("kk")
            self._emit(f"for (uint {kk} = 0; {kk} < {BK}u; {kk} += 8u) {{")
            self.indent += 1
            self._emit(f"{sg_a_type} _sg_A[{TM}], _sg_B[{TN}];")
            for tm in range(TM):
                self._emit(self.emitter.simd_load(
                    f"_sg_A[{tm}]",
                    f"&{sa_cur}[(_base_br + {tm}) * {8 * BK}u + {kk}]",
                    f"{BK}ul"))
            for tn in range(TN):
                self._emit(self.emitter.simd_load(
                    f"_sg_B[{tn}]",
                    f"&{sb_cur}[{kk} * {BN}u + (_base_bc + {tn}) * 8u]",
                    f"{BN}ul"))
            for tm in range(TM):
                for tn in range(TN):
                    self._emit(self.emitter.simd_multiply_accumulate(
                        f"_sg_acc[{tm}][{tn}]", f"_sg_A[{tm}]", f"_sg_B[{tn}]",
                        f"_sg_acc[{tm}][{tn}]"))
            self.indent -= 1
            self._emit("}")
            if needs_sg_guard:
                self.indent -= 1
                self._emit("}")

            self._emit(self.emitter.barrier())

            # Swap ping-pong pointers
            self._emit(f"{{ threadgroup {in_type}* _tmp;")
            self._emit(f"  _tmp = {sa_cur}; {sa_cur} = {sa_nxt}; {sa_nxt} = _tmp;")
            self._emit(f"  _tmp = {sb_cur}; {sb_cur} = {sb_nxt}; {sb_nxt} = _tmp; }}")

            self.indent -= 1
            self._emit("}")
        else:
            # --- Single-buffered K-loop ---
            self._emit(f"for (int {iv} = 0; {iv} < {k_param}; {iv} += {BK}) {{")
            self.indent += 1

            self._emit_cooperative_tile_load(
                sa, a_base, stride_am, BM, BK, A_ELEMS, A_LOADS, THREADS,
                row_offset=f"(int){pid_m} * {BM}", col_offset=iv,
                vec4=use_vec4_a, in_dtype=in_dtype)
            self._emit_cooperative_tile_load(
                sb, b_base, stride_bk, BK, BN, B_ELEMS, B_LOADS, THREADS,
                row_offset=iv, col_offset=f"(int){pid_n} * {BN}",
                vec4=use_vec4_b, in_dtype=in_dtype,
                device_dtype=b_load_dtype or '')
            if dequant_scale:
                self._emit_dequant_scale_pass(sb, B_ELEMS, dequant_scale, in_dtype)
            self._emit(self.emitter.barrier())

            # MMA with register tiling
            if needs_sg_guard:
                self._emit("if (_sg_active) {")
                self.indent += 1
            kk = self._fresh_var("kk")
            self._emit(f"for (uint {kk} = 0; {kk} < {BK}u; {kk} += 8u) {{")
            self.indent += 1
            self._emit(f"{sg_a_type} _sg_A[{TM}], _sg_B[{TN}];")
            for tm in range(TM):
                self._emit(self.emitter.simd_load(
                    f"_sg_A[{tm}]",
                    f"&{sa}[(_base_br + {tm}) * {8 * BK}u + {kk}]",
                    f"{BK}ul"))
            for tn in range(TN):
                self._emit(self.emitter.simd_load(
                    f"_sg_B[{tn}]",
                    f"&{sb}[{kk} * {BN}u + (_base_bc + {tn}) * 8u]",
                    f"{BN}ul"))
            for tm in range(TM):
                for tn in range(TN):
                    self._emit(self.emitter.simd_multiply_accumulate(
                        f"_sg_acc[{tm}][{tn}]", f"_sg_A[{tm}]", f"_sg_B[{tn}]",
                        f"_sg_acc[{tm}][{tn}]"))
            self.indent -= 1
            self._emit("}")
            if needs_sg_guard:
                self.indent -= 1
                self._emit("}")
            self._emit(self.emitter.barrier())

            self.indent -= 1
            self._emit("}")

        # --- Load fused scale vector ---
        if fused_scale:
            scale_arg_idx = fused_scale['scale_arg_idx']
            scale_base = self._arg_name(self.func_args[scale_arg_idx])
            tg_size = self.emitter.threads_per_group_expr()
            # Cooperative load: each thread loads one or more elements
            # Scale is 1D with BN elements, offset by pid_n * BN
            SCALE_LOADS = max(1, (BN + THREADS - 1) // THREADS)
            if SCALE_LOADS > 1:
                self._emit(f"for (uint _si = 0; _si < {SCALE_LOADS}u; _si++) {{")
                self._emit(f"    uint _sidx = (uint){tid} + _si * {tg_size};")
                self._emit(f"    if (_sidx < {BN}u) {s_scale}[_sidx] = ({acc_type}){scale_base}[(int){pid_n} * {BN} + (int)_sidx];")
                self._emit("}")
            else:
                self._emit(f"if ((uint){tid} < {BN}u) {s_scale}[(uint){tid}] = ({acc_type}){scale_base}[(int){pid_n} * {BN} + (int)(uint){tid}];")
            self._emit(self.emitter.barrier())

        # --- Load fused bias vector ---
        if fused_bias:
            bias_arg_idx = fused_bias['bias_arg_idx']
            bias_base = self._arg_name(self.func_args[bias_arg_idx])
            tg_size = self.emitter.threads_per_group_expr()
            BIAS_LOADS = max(1, (BN + THREADS - 1) // THREADS)
            if BIAS_LOADS > 1:
                self._emit(f"for (uint _bi = 0; _bi < {BIAS_LOADS}u; _bi++) {{")
                self._emit(f"    uint _bidx = (uint){tid} + _bi * {tg_size};")
                self._emit(f"    if (_bidx < {BN}u) {s_bias}[_bidx] = ({acc_type}){bias_base}[(int){pid_n} * {BN} + (int)_bidx];")
                self._emit("}")
            else:
                self._emit(f"if ((uint){tid} < {BN}u) {s_bias}[(uint){tid}] = ({acc_type}){bias_base}[(int){pid_n} * {BN} + (int)(uint){tid}];")
            self._emit(self.emitter.barrier())

        # --- Store results ---
        if use_sC:
            # Post-ops path: store accumulators to sC for further processing
            # Zero sC first
            if C_STORES > 1:
                ei = self._fresh_var("ei")
                self._emit(f"for (uint {ei} = 0; {ei} < {C_STORES}u; {ei}++) {{")
                self._emit(f"    uint _cidx = (uint){tid} + {ei} * {THREADS}u;")
                self._emit(f"    if (_cidx < {C_ELEMS}u) {sc}[_cidx] = ({acc_type})0;")
                self._emit("}")
            else:
                self._emit(f"{sc}[(uint){tid}] = ({acc_type})0;")
            self._emit(self.emitter.barrier())

            # Store each TM*TN accumulator to its sC position
            if needs_sg_guard:
                self._emit("if (_sg_active) {")
                self.indent += 1
            for tm in range(TM):
                for tn in range(TN):
                    self._emit(self.emitter.simd_store(
                        f"_sg_acc[{tm}][{tn}]",
                        f"&{sc}[(_base_br + {tm}) * {8 * BN}u + (_base_bc + {tn}) * 8u]",
                        f"{BN}ul"))
            if needs_sg_guard:
                self.indent -= 1
                self._emit("}")
            self._emit(self.emitter.barrier())

            sC_tile = TileInfo(shared_name=sc, shape=[BM, BN], dtype=acc_dtype)
            # Save matmul store params so post-ops tt.store can use optimized
            # grid-stride store without materializing a BM×BN index tile
            self._matmul_store_params = {
                'pid_m': pid_m, 'pid_n': pid_n,
                'BM': BM, 'BN': BN,
                'c_base': c_base, 'stride_cm': stride_cm,
                'c_out_dtype': c_out_dtype,
            }
            for i, result_name in enumerate(op.results):
                if i == dot_acc_idx:
                    if needs_cast:
                        cast_tile = self._alloc_tile([BM, BN], c_out_dtype)
                        self._emit_tile_loop(C_ELEMS,
                            f"{cast_tile.shared_name}[_fi] = ({c_out_type}){sc}[_fi];")
                        self._register_tile(result_name, cast_tile)
                    else:
                        self._register_tile(result_name, sC_tile)
                else:
                    self._set_val(result_name, TType(dtype='i32'), "0")
        else:
            # Direct store: simdgroup registers → device memory
            row_off = f"(int){pid_m} * {BM}"
            col_off = f"(int){pid_n} * {BN}"

            if needs_sg_guard:
                self._emit("if (_sg_active) {")
                self.indent += 1
            if needs_cast or fused_scale or fused_bias:
                # Use per-SG temp buffer for type cast, fused scale, and/or fused bias
                scale_expr = ""
                if fused_scale:
                    scale_expr = f" * {s_scale}[(_base_bc + {{tn}}) * 8u + _c]"
                # Build the value expression applied to each element
                # For fused_bias: val + bias[col]
                # For fused_bias + gelu: gelu(val + bias[col])
                has_gelu = fused_bias and fused_bias.get('has_gelu', False)
                has_silu = fused_bias and fused_bias.get('has_silu', False)
                for tm in range(TM):
                    for tn in range(TN):
                        self._emit(self.emitter.simd_store(
                            f"_sg_acc[{tm}][{tn}]",
                            f"&{s_cast}[{sg_id} * 64u]", "8ul"))
                        self._emit(self.emitter.barrier())
                        # Each thread in the SG copies 2 elements (64 / 32)
                        sc_expr = scale_expr.format(tn=tn)
                        self._emit(f"{{ uint _lane = (uint){tid} % 32u;")
                        self._emit("  for (uint _ei = 0; _ei < 2u; _ei++) {")
                        self._emit("    uint _idx = _lane * 2u + _ei;")
                        self._emit("    uint _r = _idx / 8u, _c = _idx % 8u;")
                        if fused_bias:
                            # Apply bias add (and optional activation) per-element
                            self._emit(f"    {acc_type} _val = {s_cast}[{sg_id} * 64u + _idx] + {s_bias}[(_base_bc + {tn}) * 8u + _c];")
                            if has_silu:
                                # Fused SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                                self._emit("    _val = _val / (1.0f + exp(-_val));")
                            elif has_gelu:
                                # Fused GELU: x * 0.5 * (1 + erf(x * rsqrt(2)))
                                self._emit("    _val = _val * 0.5f * (1.0f + _erf_approx(_val * 0.7071067811865476f));")
                            self._emit(f"    {c_base}[({row_off} + (_base_br + {tm}) * 8 + (int)_r) * {stride_cm} + ({col_off} + (_base_bc + {tn}) * 8 + (int)_c)] = ({c_out_type})_val;")
                        else:
                            self._emit(f"    {c_base}[({row_off} + (_base_br + {tm}) * 8 + (int)_r) * {stride_cm} + ({col_off} + (_base_bc + {tn}) * 8 + (int)_c)] = ({c_out_type})({s_cast}[{sg_id} * 64u + _idx]{sc_expr});")
                        self._emit("  } }")
            else:
                # Direct simdgroup_store to device memory (types match)
                for tm in range(TM):
                    for tn in range(TN):
                        self._emit(self.emitter.simd_store(
                            f"_sg_acc[{tm}][{tn}]",
                            f"&{c_base}[({row_off} + (_base_br + {tm}) * 8) * {stride_cm} + ({col_off} + (_base_bc + {tn}) * 8)]",
                            f"(ulong){stride_cm}"))
            if needs_sg_guard:
                self.indent -= 1
                self._emit("}")

            self._dot_store_emitted = True
            for i, result_name in enumerate(op.results):
                self._set_val(result_name, TType(dtype='i32'), "0")
                self._dot_consumed_ssa.add(result_name)

    def _gen_scf_for_dot_scalar(self, op: Op, dot_acc_idx: int,
                                 BM: int, BN: int, BK: int,
                                 in_dtype: str, acc_dtype: str,
                                 has_post_ops: bool = False,
                                 b_load_dtype: str | None = None,
                                 dequant_scale: str | None = None,
                                 fused_bias: dict | None = None,
                                 fused_scale: dict | None = None):
        """Scalar matmul loop with register sub-tiling for high performance.

        Each thread computes a TM×TN sub-tile of the output, reusing A rows
        across TN columns and B columns across TM rows. This reduces shared
        memory reads by (TM+TN)/(TM*TN) compared to naive per-element approach.

        Uses grid-stride loops for cooperative loading so the code works for
        any thread count (critical for Intel GPUs with < 1024 max threads).
        """
        # Choose register sub-tile size (TM×TN per thread).
        # Goal: WORK_ITEMS = (BM/TM)*(BN/TN) should use all available threads.
        # Register sub-tiling reduces shared memory reads by (TM+TN)/(TM*TN).
        # With unroll pragma, Intel GPUs typically get 448 max_threads.
        # We want WORK_ITEMS ≈ 3 × 448 = 1344 so 3 overflow slots suffice,
        # but also WORK_ITEMS ≥ 256 for parallelism.
        # For 32×32: TM=1 (1024 work items)
        # For 64×64: TM=2,TN=2 (1024 work items)
        # For 128×128: TM=4,TN=4 (1024 work items)
        TM, TN = 1, 1
        for tm, tn in [(4, 4), (2, 4), (4, 2), (2, 2)]:
            if BM % tm == 0 and BN % tn == 0:
                work = (BM // tm) * (BN // tn)
                if work >= 256:  # enough parallelism for Intel
                    TM, TN = tm, tn
                    break

        A_ELEMS = BM * BK
        B_ELEMS = BK * BN
        WORK_ITEMS = (BM // TM) * (BN // TN)
        ITEMS_N = BN // TN

        tid = self.emitter.thread_id_expr()
        in_type = self.emitter.map_dtype(in_dtype)
        acc_type = self.emitter.map_dtype(acc_dtype)
        tg_size = "_tg_size.x"  # Runtime thread count

        _excl = set()
        if fused_scale:
            _excl.add(fused_scale['scale_arg_idx'])
        if fused_bias:
            _excl.add(fused_bias['bias_arg_idx'])
        params = self._extract_matmul_params(op, exclude_ptr_indices=_excl)
        a_base = params['a_base']
        b_base = params['b_base']
        c_base = params['c_base']
        k_param = params['k_param']
        stride_am = params['stride_am']
        stride_bk = params['stride_bk']
        stride_cm = params['stride_cm']

        pid_m = self._pid_exprs.get('x', f'(int){self.emitter.threadgroup_id_expr("x")}')
        pid_n = self._pid_exprs.get('y', f'(int){self.emitter.threadgroup_id_expr("y")}')

        c_out_dtype = params['c_out_dtype']
        c_out_type = self.emitter.map_dtype(c_out_dtype)

        sa = self._fresh_var("sA")
        sb = self._fresh_var("sB")
        self._emit(self.emitter.shared_memory_decl(sa, in_dtype, A_ELEMS))
        self._emit(self.emitter.shared_memory_decl(sb, in_dtype, B_ELEMS))

        # Grid-stride work items: each thread handles ceil(WORK_ITEMS/tg_size) items.
        # We statically generate MAX_SLOTS sets of accumulators so that even when
        # max_threads is much less than WORK_ITEMS (e.g., 448 threads for 1024 items
        # due to register pressure from unrolling), all work items are covered.
        # MIN_THREADS=384 is safe for Intel Gen9 (24 EUs × 8 SIMD × 2 HW threads).
        # This gives MAX_SLOTS=3 for 1024 work items (48 acc vs 64 for 4 slots).
        MIN_THREADS = 384
        MAX_SLOTS = (WORK_ITEMS + MIN_THREADS - 1) // MIN_THREADS

        for s in range(MAX_SLOTS):
            self._emit(f"uint _wi{s} = (uint){tid} + {s}u * {tg_size};")
            self._emit(f"uint _wr{s} = (_wi{s} / {ITEMS_N}u) * {TM}u;")
            self._emit(f"uint _wc{s} = (_wi{s} % {ITEMS_N}u) * {TN}u;")
            self._emit(f"bool _active{s} = (_wi{s} < {WORK_ITEMS}u);")

        # Register accumulators for each slot's TM×TN sub-tile
        for s in range(MAX_SLOTS):
            sfx = f"{s}" if s > 0 else ""
            for tm in range(TM):
                for tn in range(TN):
                    self._emit(f"{acc_type} _acc{sfx}_{tm}_{tn} = ({acc_type})0;")

        iv = self._fresh_var("iv")
        self._emit(f"for (int {iv} = 0; {iv} < {k_param}; {iv} += {BK}) {{")
        self.indent += 1

        # Cooperative tile loads using grid-stride loops.
        # Three paths for vec4 device→shared copies:
        # 1. MSL ptr cast: *((threadgroup half4*)&s[i]) = *((device const half4*)&buf[j])
        # 2. HLSL half4 buffer: half4 v = arg[j >> 2u]; s[i]=v.x; s[i+1]=v.y; ...
        # 3. Scalar fallback: s[i] = buf[j] (1 element per thread, good coalescing)
        _ptr_cast = self.emitter.supports_ptr_cast()
        # Vec4 cooperative loads require MSL pointer cast support.
        # The HLSL half4 structured buffer path is disabled — it requires
        # coordinated dispatch-side changes (buffer stride) that are fragile.
        a_dev_dtype = params.get('a_dev_dtype', in_dtype)
        use_vec4_a = (_ptr_cast and a_dev_dtype == 'f16' and BK % 4 == 0
                      and in_dtype == a_dev_dtype)
        use_vec4_b = (_ptr_cast and in_dtype == 'f16' and BN % 4 == 0
                      and not b_load_dtype)

        if use_vec4_a:
            a_vec_total = A_ELEMS // 4
            a_vec_cols = BK // 4
            self._emit(f"for (uint _vf = (uint){tid}; _vf < {a_vec_total}u; _vf += {tg_size}) {{")
            self._emit(f"    uint _ar = _vf / {a_vec_cols}u, _ac4 = _vf % {a_vec_cols}u;")
            s_idx = f"_ar * {BK}u + _ac4 * 4u"
            d_idx = f"((int){pid_m} * {BM} + (int)_ar) * {stride_am} + ({iv} + (int)_ac4 * 4)"
            if _ptr_cast:
                copy_stmt = self.emitter.vec4_copy_device_to_shared('half', sa, s_idx, a_base, d_idx)
                self._emit(f"    {copy_stmt}")
            else:
                # half4 structured buffer: single wide read, decompose to groupshared
                # Cast needed when shared type differs from device (e.g., f32 shared, f16 device)
                cast = f"({in_type})" if in_dtype != a_dev_dtype else ""
                self._emit(f"    half4 _vh = {a_base}[({d_idx}) >> 2u];")
                self._emit(f"    {sa}[{s_idx}] = {cast}_vh.x; {sa}[({s_idx}) + 1u] = {cast}_vh.y; {sa}[({s_idx}) + 2u] = {cast}_vh.z; {sa}[({s_idx}) + 3u] = {cast}_vh.w;")
            self._emit("}")
        else:
            self._emit(f"for (uint _af = (uint){tid}; _af < {A_ELEMS}u; _af += {tg_size}) {{")
            self._emit(f"    uint _ar = _af / {BK}u, _ac = _af % {BK}u;")
            self._emit(f"    {sa}[_af] = {a_base}[((int){pid_m} * {BM} + (int)_ar) * {stride_am} + ({iv} + (int)_ac)];")
            self._emit("}")

        if use_vec4_b:
            b_vec_total = B_ELEMS // 4
            b_vec_cols = BN // 4
            self._emit(f"for (uint _vf = (uint){tid}; _vf < {b_vec_total}u; _vf += {tg_size}) {{")
            self._emit(f"    uint _br = _vf / {b_vec_cols}u, _bc4 = _vf % {b_vec_cols}u;")
            s_idx = f"_br * {BN}u + _bc4 * 4u"
            d_idx = f"({iv} + (int)_br) * {stride_bk} + ((int){pid_n} * {BN} + (int)_bc4 * 4)"
            if _ptr_cast:
                copy_stmt = self.emitter.vec4_copy_device_to_shared('half', sb, s_idx, b_base, d_idx)
                self._emit(f"    {copy_stmt}")
            else:
                self._emit(f"    half4 _vh = {b_base}[({d_idx}) >> 2u];")
                self._emit(f"    {sb}[{s_idx}] = _vh.x; {sb}[({s_idx}) + 1u] = _vh.y; {sb}[({s_idx}) + 2u] = _vh.z; {sb}[({s_idx}) + 3u] = _vh.w;")
            self._emit("}")
        elif b_load_dtype:
            b_dev_type = self.emitter.map_dtype(b_load_dtype)
            self._emit(f"for (uint _bf = (uint){tid}; _bf < {B_ELEMS}u; _bf += {tg_size}) {{")
            self._emit(f"    uint _br = _bf / {BN}u, _bc = _bf % {BN}u;")
            if not _ptr_cast and b_load_dtype == 'i8':
                # HLSL: RWStructuredBuffer<int> packs 4 int8 per element — unpack via shift/mask
                self._emit(f"    int _b_col = (int){pid_n} * {BN} + (int)_bc;")
                self._emit(f"    int _packed = {b_base}[({iv} + (int)_br) * {stride_bk} + _b_col / 4];")
                self._emit("    int _shift = (_b_col & 3) * 8;")
                self._emit(f"    {sb}[_bf] = ({in_type})((_packed << (24 - _shift)) >> 24);")
            else:
                self._emit(f"    {sb}[_bf] = ({in_type})(({b_dev_type}){b_base}[({iv} + (int)_br) * {stride_bk} + ((int){pid_n} * {BN} + (int)_bc)]);")
            self._emit("}")
        else:
            self._emit(f"for (uint _bf = (uint){tid}; _bf < {B_ELEMS}u; _bf += {tg_size}) {{")
            self._emit(f"    uint _br = _bf / {BN}u, _bc = _bf % {BN}u;")
            self._emit(f"    {sb}[_bf] = {b_base}[({iv} + (int)_br) * {stride_bk} + ((int){pid_n} * {BN} + (int)_bc)];")
            self._emit("}")
        if dequant_scale:
            self._emit_dequant_scale_pass(sb, B_ELEMS, dequant_scale, in_dtype)

        self._emit(self.emitter.barrier())

        # Rank-1 update: each thread processes its TM×TN sub-tile(s)
        # Statically unrolled for each slot with unroll pragma on K-loop.
        # NESO_UNROLL controls unroll factor (default 2).
        import os
        _unroll = int(os.environ.get("NESO_UNROLL", "2"))
        kk = self._fresh_var("kk")

        # Vec4 inner loop: batch 4 k-iterations, load A and B as half4.
        # Reduces SLM reads from 8 scalar/iter to 2 vec4/iter (4x fewer).
        # Requires ptr cast for true wide LDS loads. HLSL half4() constructor from
        # scalar reads provides no benefit (DXC can't merge, adds pack/unpack overhead).
        use_vec4_inner = (_ptr_cast and TM == 4 and TN == 4 and BK % 4 == 0
                          and in_dtype == 'f16' and not b_load_dtype)

        for s in range(MAX_SLOTS):
            sfx = f"{s}" if s > 0 else ""
            self._emit(f"if (_active{s}) {{")
            if use_vec4_inner:
                if _unroll > 0:
                    self._emit(f"#pragma clang loop unroll_count({_unroll})")
                self._emit(f"for (int {kk} = 0; {kk} < {BK}; {kk} += 4) {{")
                self.indent += 1
                # Load 4 half4 A values (4 consecutive k-values per row)
                for tm in range(TM):
                    a_idx = f"(_wr{s} + {tm}u) * {BK}u + (uint){kk}"
                    a_load = self.emitter.vec4_load_shared('half', sa, a_idx)
                    self._emit(f"half4 _av{tm} = {a_load};")
                # 4 sub-k iterations with unique B vec4 names
                for dk in range(4):
                    # Load 1 half4 B value (4 consecutive cols for this k)
                    b_idx = f"(uint)({kk}+{dk}) * {BN}u + _wc{s}"
                    b_load = self.emitter.vec4_load_shared('half', sb, b_idx)
                    self._emit(f"half4 _bv{dk} = {b_load};")
                    _fma = self.emitter.math_func('fma')
                    for tm in range(TM):
                        for tn in range(TN):
                            self._emit(f"_acc{sfx}_{tm}_{tn} = {_fma}(({acc_type})_av{tm}[{dk}], ({acc_type})_bv{dk}[{tn}], _acc{sfx}_{tm}_{tn});")
                self.indent -= 1
                self._emit("}")
            else:
                if _unroll > 0:
                    self._emit(f"#pragma clang loop unroll_count({_unroll})")
                self._emit(f"for (int {kk} = 0; {kk} < {BK}; {kk}++) {{")
                self.indent += 1
                for tm in range(TM):
                    self._emit(f"{acc_type} _a{tm} = ({acc_type}){sa}[(_wr{s} + {tm}u) * {BK}u + (uint){kk}];")
                for tn in range(TN):
                    self._emit(f"{acc_type} _b{tn} = ({acc_type}){sb}[(uint){kk} * {BN}u + _wc{s} + {tn}u];")
                _fma = self.emitter.math_func('fma')
                for tm in range(TM):
                    for tn in range(TN):
                        self._emit(f"_acc{sfx}_{tm}_{tn} = {_fma}(_a{tm}, _b{tn}, _acc{sfx}_{tm}_{tn});")
                self.indent -= 1
                self._emit("}")
            self._emit("}")

        self._emit(self.emitter.barrier())
        self.indent -= 1
        self._emit("}")

        if has_post_ops:
            # Need sC tile in shared memory for post-ops to read from
            sC_tile = self._alloc_tile([BM, BN], acc_dtype)
            sc = sC_tile.shared_name
            for s in range(MAX_SLOTS):
                sfx = f"{s}" if s > 0 else ""
                self._emit(f"if (_active{s}) {{")
                for tm in range(TM):
                    for tn in range(TN):
                        self._emit(f"{sc}[(_wr{s} + {tm}u) * {BN}u + _wc{s} + {tn}u] = _acc{sfx}_{tm}_{tn};")
                self._emit("}")
            self._emit(self.emitter.barrier())
            for i, result_name in enumerate(op.results):
                if i == dot_acc_idx:
                    self._register_tile(result_name, sC_tile)
                else:
                    self._set_val(result_name, TType(dtype='i32'), "0")
        else:
            # Store directly from registers to global memory with per-thread cast.
            # Avoids allocating BM×BN shared memory tiles (critical for 128×128).

            # Fused bias: load bias vector [BN] into shared memory
            s_bias = None
            if fused_bias:
                s_bias = self._fresh_var("sBias")
                self._emit(self.emitter.shared_memory_decl(s_bias, acc_dtype, BN))
                bias_arg_idx = fused_bias['bias_arg_idx']
                bias_base = self._arg_name(self.func_args[bias_arg_idx])
                tg_size = "_tg_size.x"
                self._emit(f"for (uint _bi = (uint){tid}; _bi < {BN}u; _bi += {tg_size}) {{")
                self._emit(f"    {s_bias}[_bi] = ({acc_type}){bias_base}[(int){pid_n} * {BN} + (int)_bi];")
                self._emit("}")
                self._emit(self.emitter.barrier())

            # Fused per-column scale: load scale vector [BN] into shared memory
            s_scale = None
            if fused_scale:
                s_scale = self._fresh_var("sScale")
                self._emit(self.emitter.shared_memory_decl(s_scale, acc_dtype, BN))
                scale_arg_idx = fused_scale['scale_arg_idx']
                scale_base = self._arg_name(self.func_args[scale_arg_idx])
                tg_size = "_tg_size.x"
                self._emit(f"for (uint _si = (uint){tid}; _si < {BN}u; _si += {tg_size}) {{")
                self._emit(f"    {s_scale}[_si] = ({acc_type}){scale_base}[(int){pid_n} * {BN} + (int)_si];")
                self._emit("}")
                self._emit(self.emitter.barrier())

            row_off = f"(int){pid_m} * {BM}"
            col_off = f"(int){pid_n} * {BN}"
            has_fused_gelu = fused_bias and fused_bias.get('has_gelu', False)
            has_fused_silu = fused_bias and fused_bias.get('has_silu', False)

            # Vec4 stores when TN==4 and output is f16 (4 consecutive cols → half4)
            # Disable vec4 when fused bias is active (need per-element bias add)
            use_vec4_store = (TN == 4 and c_out_dtype == 'f16' and _ptr_cast
                              and not fused_bias)
            for s in range(MAX_SLOTS):
                sfx = f"{s}" if s > 0 else ""
                self._emit(f"if (_active{s}) {{")
                if use_vec4_store:
                    for tm in range(TM):
                        d_idx = f"({row_off} + (int)(_wr{s} + {tm}u)) * {stride_cm} + ({col_off} + (int)_wc{s})"
                        if _ptr_cast:
                            if fused_scale:
                                comps = [f"(half)(_acc{sfx}_{tm}_{tn} * {s_scale}[_wc{s} + {tn}u])" for tn in range(TN)]
                            else:
                                comps = [f"(half)_acc{sfx}_{tm}_{tn}" for tn in range(TN)]
                            store_stmt = self.emitter.vec4_store_device('half', c_base, d_idx, comps)
                            self._emit(store_stmt)
                        else:
                            # half4 structured buffer: single wide write
                            if fused_scale:
                                comps = ", ".join(f"(half)(_acc{sfx}_{tm}_{tn} * {s_scale}[_wc{s} + {tn}u])" for tn in range(TN))
                            else:
                                comps = ", ".join(f"(half)_acc{sfx}_{tm}_{tn}" for tn in range(TN))
                            self._emit(f"{c_base}[({d_idx}) >> 2u] = half4({comps});")
                elif fused_bias:
                    # Fused bias (+ optional activation) applied per element
                    for tm in range(TM):
                        for tn in range(TN):
                            acc_ref = f"_acc{sfx}_{tm}_{tn}"
                            col_idx = f"(_wc{s} + {tn}u)"
                            self._emit(f"  {{ {acc_type} _val = {acc_ref} + {s_bias}[{col_idx}];")
                            if has_fused_silu:
                                self._emit(f"    _val = _val / (({acc_type})1 + exp(-_val));")
                            elif has_fused_gelu:
                                self._emit(f"    _val = _val * ({acc_type})0.5 * (({acc_type})1 + _erf_approx(_val * ({acc_type})0.7071067811865476));")
                            d_idx = f"({row_off} + (int)(_wr{s} + {tm}u)) * {stride_cm} + ({col_off} + (int){col_idx})"
                            self._emit(f"    {c_base}[{d_idx}] = ({c_out_type})_val; }}")
                elif fused_scale:
                    # Fused per-column scale applied per element
                    for tm in range(TM):
                        for tn in range(TN):
                            acc_ref = f"_acc{sfx}_{tm}_{tn}"
                            col_idx = f"(_wc{s} + {tn}u)"
                            d_idx = f"({row_off} + (int)(_wr{s} + {tm}u)) * {stride_cm} + ({col_off} + (int){col_idx})"
                            self._emit(f"{c_base}[{d_idx}] = ({c_out_type})({acc_ref} * {s_scale}[{col_idx}]);")
                else:
                    for tm in range(TM):
                        for tn in range(TN):
                            self._emit(f"{c_base}[({row_off} + (int)(_wr{s} + {tm}u)) * {stride_cm} + ({col_off} + (int)(_wc{s} + {tn}u))] = ({c_out_type})_acc{sfx}_{tm}_{tn};")
                self._emit("}")
            self._dot_store_emitted = True
            for i, result_name in enumerate(op.results):
                self._set_val(result_name, TType(dtype='i32'), "0")
                self._dot_consumed_ssa.add(result_name)

    def _emit_cooperative_tile_load(self, shared_name: str, base_ptr: str,
                                     stride_expr: str, rows: int, cols: int,
                                     total_elems: int, loads: int, threads: int,
                                     row_offset: str, col_offset: str,
                                     vec4: bool = False, in_dtype: str = '',
                                     device_dtype: str = ''):
        """Emit cooperative tile loading into shared memory.

        vec4: use half4 vectorized loads (4 elements per transaction). Requires
              in_dtype='f16' and cols % 4 == 0.
        device_dtype: if set and different from in_dtype, load from device as this
              type and cast to in_dtype when storing to threadgroup memory. Used for
              W8A16 dequant (load i8 → store f16 to threadgroup).
        """
        tid = self.emitter.thread_id_expr()
        # Dequant path: load from device as device_dtype, store to tgmem as in_dtype
        dequant = (device_dtype and device_dtype != in_dtype)
        dequant_tg_type = self.emitter.map_dtype(in_dtype) if dequant else None

        # All cooperative loads use grid-stride loops with runtime _tg_size.x to
        # handle GPUs where maxTotalThreadsPerThreadgroup < compile-time thread count.

        # Try vec4 path for half data with aligned columns (requires ptr cast for
        # true wide loads — scalar fallback preserves cross-thread coalescing)
        if vec4 and self.emitter.supports_ptr_cast() and in_dtype == 'f16' and cols % 4 == 0 and not dequant:
            vec_total = total_elems // 4
            vec_cols = cols // 4
            self._emit(f"for (uint _vflat = (uint){tid}; _vflat < {vec_total}u; _vflat += _tg_size.x) {{")
            self._emit(f"    uint _r = _vflat / {vec_cols}u, _vc = _vflat % {vec_cols}u;")
            s_idx = f"_r * {cols}u + _vc * 4u"
            d_idx = f"({row_offset} + (int)_r) * {stride_expr} + ({col_offset} + (int)_vc * 4)"
            copy_stmt = self.emitter.vec4_copy_device_to_shared('half', shared_name, s_idx, base_ptr, d_idx)
            self._emit(f"    {copy_stmt}")
            self._emit("}")
            return

        # Vectorized dequant: load char4 from device, convert to half4 for threadgroup
        # (requires ptr cast — MSL only, HLSL falls through to scalar)
        if self.emitter.supports_ptr_cast() and dequant and device_dtype == 'i8' and in_dtype == 'f16' and cols % 4 == 0:
            vec_total = total_elems // 4
            vec_cols = cols // 4
            self._emit(f"for (uint _vflat = (uint){tid}; _vflat < {vec_total}u; _vflat += _tg_size.x) {{")
            self._emit(f"    uint _r = _vflat / {vec_cols}u, _vc = _vflat % {vec_cols}u;")
            self._emit(f"    char4 _cv = *((device const char4*)&{base_ptr}[({row_offset} + (int)_r) * {stride_expr} + ({col_offset} + (int)_vc * 4)]);")
            self._emit(f"    *((threadgroup half4*)&{shared_name}[_r * {cols}u + _vc * 4u]) = half4(float4(int4(_cv)));")
            self._emit("}")
            return

        # HLSL scalar int8 dequant: unpack individual bytes from int32 elements
        if not self.emitter.supports_ptr_cast() and dequant and device_dtype == 'i8' and in_dtype == 'f16':
            col_expr = f"({col_offset} + (int)_c)"
            self._emit(f"for (uint _flat = (uint){tid}; _flat < {total_elems}u; _flat += _tg_size.x) {{")
            self._emit(f"    uint _r = _flat / {cols}u, _c = _flat % {cols}u;")
            self._emit(f"    int _b_col = {col_expr};")
            self._emit(f"    int _packed = {base_ptr}[({row_offset} + (int)_r) * {stride_expr} + _b_col / 4];")
            self._emit("    int _shift = (_b_col & 3) * 8;")
            self._emit(f"    {shared_name}[_flat] = ({dequant_tg_type})((_packed << (24 - _shift)) >> 24);")
            self._emit("}")
            return

        addr_expr = f"{base_ptr}[({row_offset} + (int)_r) * {stride_expr} + ({col_offset} + (int)_c)]"
        if dequant:
            store_expr = f"{shared_name}[_flat] = ({dequant_tg_type})({addr_expr});"
        else:
            store_expr = f"{shared_name}[_flat] = {addr_expr};"
        self._emit(f"for (uint _flat = (uint){tid}; _flat < {total_elems}u; _flat += _tg_size.x) {{")
        self._emit(f"    uint _r = _flat / {cols}u, _c = _flat % {cols}u;")
        self._emit(f"    {store_expr}")
        self._emit("}")

    def _emit_dequant_scale_pass(self, shared_name: str, total_elems: int,
                                  scale_expr: str, in_dtype: str):
        """Multiply every element of a threadgroup tile by a scalar scale factor.

        Used after dequant cooperative tile loads (int → float) to apply the
        per-tensor scale from the original arith.mulf in the dequant chain.
        """
        tid = self.emitter.thread_id_expr()
        tg_type = self.emitter.map_dtype(in_dtype)
        self._emit(f"for (uint _sf = (uint){tid}; _sf < {total_elems}u; _sf += _tg_size.x)")
        self._emit(f"    {shared_name}[_sf] = ({tg_type})((float){shared_name}[_sf] * (float)({scale_expr}));")

    def _gen_scf_for_generic(self, op: Op):
        """Generic scf.for codegen — handles tile iter_args and tile body ops."""
        start_expr = self._get_expr(op.loop_start)
        end_expr = self._get_expr(op.loop_end)
        step_expr = self._get_expr(op.loop_step)
        iv_name = self._fresh_var("iv")
        self._set_val(op.loop_var, TType(dtype='i32'), iv_name)

        # Detect register accumulator candidates (dot accumulators that can
        # stay in simdgroup registers, eliminating shared memory for O tile).
        reg_acc_candidates = {}  # arg_idx -> chain info
        if self.emitter.supports_simd_matrix():
            for i, (arg_name, init_name) in enumerate(
                    zip(op.iter_arg_names, op.iter_arg_inits)):
                if not self._is_tile(init_name):
                    continue
                # Get shape from deferred or materialized tile
                if init_name in self._deferred_tiles:
                    shape = list(self._deferred_tiles[init_name][0])
                elif init_name in self._tiles:
                    shape = list(self._tiles[init_name].shape)
                else:
                    continue
                if len(shape) != 2 or shape[0] % 8 != 0 or shape[1] % 8 != 0:
                    continue
                chain = self._detect_reg_acc_chain(op, arg_name, i)
                if chain:
                    reg_acc_candidates[i] = chain

        iter_var_names = []
        iter_tiles = []  # Track which iter_args are tiles
        for i, (arg_name, init_name) in enumerate(
                zip(op.iter_arg_names, op.iter_arg_inits)):
            init_val = self._get_val(init_name)
            if self._is_tile(init_name) and i in reg_acc_candidates:
                # Register accumulator: keep in simdgroup registers, no shared tile
                if init_name in self._deferred_tiles:
                    shape, dtype = self._deferred_tiles[init_name][:2]
                else:
                    t = self._get_tile(init_name)
                    shape, dtype = list(t.shape), t.dtype
                BM_r, d_r = shape
                # Use minimum of 16 SGs (512 threads) for bps calculation so the
                # kernel works correctly when dispatched with fewer threads than
                # block_size.  The contiguous mapping (blk = sg*bps + bi) naturally
                # leaves higher SGs idle when dispatched with more threads.
                threads = min(self.block_size, self.MAX_THREADS)
                NUM_SG = max(1, threads // 32)
                nbn = d_r // 8
                nbt = (BM_r // 8) * nbn
                # Cap effective SGs to avoid exceeding optimal thread count.
                # For high block counts (bps≥3), empirically 16 SGs is optimal
                # on Apple Silicon to balance register pressure vs utilization.
                MIN_REG_SG = 16
                effective_sg = min(NUM_SG, MIN_REG_SG)
                bps = max(1, (nbt + effective_sg - 1) // effective_sg)
                min_sgs_needed = (nbt + bps - 1) // bps
                sg_var = self._fresh_var("sg")
                reg_name = self._fresh_var("sgO")
                acc_type = self.emitter.simd_matrix_type(dtype, 8, 8)
                tid = self.emitter.thread_id_expr()
                self._emit(f"uint {sg_var} = (uint){tid} / 32u;")
                self._emit(f"{acc_type} {reg_name}[{bps}];")
                self._emit(f"for (uint _bi = 0; _bi < {bps}u; _bi++)")
                self._emit(f"    {reg_name}[_bi] = {acc_type}(0.0f);")
                reg = RegAccInfo(reg_name=reg_name, shape=shape, dtype=dtype,
                                 blocks_per_sg=bps, num_blocks_n=nbn,
                                 num_blocks_total=nbt, sg_var=sg_var)
                self._reg_tiles[arg_name] = reg
                self._reg_acc_bps = max(self._reg_acc_bps, bps)
                self._reg_acc_min_sgs = max(self._reg_acc_min_sgs, min_sgs_needed)
                # Set a dummy val so _get_val works (won't be used for per-element access)
                self._set_val(arg_name, init_val.ttype, "0 /*reg_acc*/")
                iter_var_names.append((None, init_val.ttype))
                iter_tiles.append(None)  # No shared tile
            elif self._is_tile(init_name):
                # Optimization: if init is a deferred constant, fill iter_arg
                # directly instead of materializing + copying (saves one tile).
                if init_name in self._deferred_tiles:
                    shape, dtype, fill_val, target_type = self._deferred_tiles[init_name]
                    iter_tile = self._alloc_tile(list(shape), dtype)
                    self._emit_tile_loop(iter_tile.total,
                        f"{self._tile_write(iter_tile, '_fi')} = ({target_type}){fill_val};")
                else:
                    # Non-deferred tile: copy init tile into a fresh tile so we
                    # don't corrupt the original.
                    init_tile = self._get_tile(init_name)
                    iter_tile = self._alloc_tile(list(init_tile.shape), init_tile.dtype)
                    self._emit_tile_copy(iter_tile, init_tile)
                self._tiles[arg_name] = iter_tile
                if iter_tile.is_register:
                    self._set_val(arg_name, init_val.ttype, iter_tile.shared_name)
                else:
                    tid = self.emitter.thread_id_expr()
                    self._set_val(arg_name, init_val.ttype, f"{iter_tile.shared_name}[(uint){tid}]")
                iter_var_names.append((None, init_val.ttype))  # None = tile, no scalar var
                iter_tiles.append(iter_tile)
                # Pin: iter_arg tiles must never be freed
                self._pinned_backing.add(self._real_backing(iter_tile))
            else:
                var = self._fresh_var("iter")
                t = self.emitter.map_dtype(init_val.ttype.dtype)
                if init_val.ttype.is_ptr:
                    self._emit(f"auto {var} = {init_val.expr};")
                else:
                    self._emit(f"{t} {var} = {init_val.expr};")
                self._set_val(arg_name, init_val.ttype, var)
                iter_var_names.append((var, init_val.ttype))
                iter_tiles.append(None)

        # Release init tiles early — they've been copied to iter_arg tiles,
        # so their backing can be reused by loop body allocations.
        for init_name in op.iter_arg_inits:
            if (init_name in self._last_use and self._last_use[init_name] <= self._current_op_idx
                    and (init_name in self._tiles or init_name in self._deferred_tiles)):
                self._release_tile(init_name)

        # --- Auto-unroll detection ---
        unroll = self._detect_wave_reduce_unroll(op)

        if unroll > 1:
            self._gen_scf_for_wave_unrolled(
                op, unroll, iv_name, start_expr, end_expr, step_expr,
                iter_var_names, iter_tiles, reg_acc_candidates)
        else:
            simple_unroll = self._detect_simple_loop_unroll(op)
            if simple_unroll > 1:
                self._gen_scf_for_simple_unrolled(
                    op, simple_unroll, iv_name, start_expr, end_expr, step_expr,
                    iter_var_names, iter_tiles)
            else:
                self._gen_scf_for_standard_loop(
                    op, iv_name, start_expr, end_expr, step_expr,
                    iter_var_names, iter_tiles)

        # Map results
        for i, result_name in enumerate(op.results):
            if i < len(iter_var_names):
                var, ttype = iter_var_names[i]
                if var is not None:
                    self._set_val(result_name, ttype, var)
                elif i in reg_acc_candidates:
                    # Register accumulator result: keep in registers for
                    # post-loop ops (normalization via diag multiply, then
                    # direct store to device memory — no shared tile needed).
                    arg_name = op.iter_arg_names[i]
                    reg = self._reg_tiles.get(arg_name)
                    if reg:
                        self._reg_tiles[result_name] = reg
                        init_val = self._get_val(arg_name)
                        self._set_val(result_name, init_val.ttype, "0 /*reg_acc*/")
                elif i < len(iter_tiles) and iter_tiles[i]:
                    # Tile result: register it
                    self._register_tile(result_name, iter_tiles[i])

    def _start_may_be_negative(self, op) -> bool:
        """Check if a scf.for loop start could be negative at runtime.

        Returns False only for known non-negative compile-time constants
        (e.g. literal 0).  Returns True for all runtime-computed starts,
        since we can't statically prove they're >= 0 (e.g. flash attention's
        sliding window: off_m - window_left).
        """
        start_val = self._try_eval_int_expr(op.loop_start)
        if start_val is not None:
            return start_val < 0
        # Unknown at compile time — conservatively assume it could be negative
        return True

    def _gen_scf_for_standard_loop(self, op, iv_name, start_expr, end_expr, step_expr,
                                     iter_var_names, iter_tiles):
        """Emit a standard (non-unrolled) for loop."""
        self._flush_fused_loops()
        # Clamp loop start to >= 0 only when it could be negative (e.g. flash
        # attention K-loop with sliding window before position 0).  For loops
        # starting at 0 or a known non-negative value, skip the clamp so the
        # compiler can constant-fold and optimize the loop freely.
        if self._start_may_be_negative(op):
            init = f"max({start_expr}, 0)"
        else:
            init = start_expr
        self._emit(f"for (int {iv_name} = {init}; {iv_name} < {end_expr}; {iv_name} += {step_expr}) {{")
        self.indent += 1

        if op.body_ops:
            self._gen_ops_with_liveness(op.body_ops)

        self._process_loop_yields(op, iter_var_names, iter_tiles)
        self._flush_fused_loops()
        self.indent -= 1
        self._emit("}")

    def _gen_scf_for_wave_unrolled(self, op, unroll, iv_name, start_expr, end_expr,
                                     step_expr, iter_var_names, iter_tiles, reg_acc_candidates):
        """Emit an auto-unrolled for loop with batched wave reductions.

        The loop body is partitioned into:
          Phase 1+2: ops independent of reduce results (loads, muls, reductions)
          Phase 3: ops dependent on reduce results (softmax, V accumulate)

        For N unrolled copies per iteration:
          - Phase 1+2 for copies 0..N-1 (reductions batched, 1 barrier)
          - Phase 3 for copies 0..N-1 (sequential, iter_arg dependent)
        Plus a scalar remainder loop for the non-divisible tail.
        """
        step_val = self._try_eval_int_expr(op.loop_step)
        phase12_ops, phase3_ops, reduce_ssas, cross_phase_ssas = \
            self._classify_body_for_unroll(op)

        self._flush_fused_loops()
        self._in_unrolled_loop = True

        # Declare IV before loops so it's shared between main + remainder.
        # Only clamp to >= 0 when start could be negative (flash attention
        # sliding window).  For non-negative starts, use start directly so
        # the compiler can constant-fold the aligned-end calculation.
        if self._start_may_be_negative(op):
            self._emit(f"int {iv_name} = max({start_expr}, 0);")
            step_n = step_val * unroll
            aligned = self._fresh_var("end_a")
            self._emit(f"int {aligned} = (({end_expr} - {iv_name}) / {step_n}) * {step_n} + {iv_name};")
        else:
            self._emit(f"int {iv_name} = {start_expr};")
            step_n = step_val * unroll
            aligned = self._fresh_var("end_a")
            self._emit(f"int {aligned} = (({end_expr} - {start_expr}) / {step_n}) * {step_n} + {start_expr};")

        # === Main loop (N-way unrolled) ===
        self._emit(f"for (; {iv_name} < {aligned}; {iv_name} += {step_n}) {{")
        self.indent += 1

        saved_cross_phase: dict[int, dict] = {}

        # Phase 1+2 for all N copies
        for k in range(unroll):
            if k == 0:
                self._set_val(op.loop_var, TType(dtype='i32'), iv_name)
            else:
                iv_k = self._fresh_var("iv")
                self._emit(f"int {iv_k} = {iv_name} + {k * step_val};")
                self._set_val(op.loop_var, TType(dtype='i32'), iv_k)

            self._unroll_copy_idx = k
            self._gen_ops_no_liveness(phase12_ops)
            self._unroll_copy_idx = None

            # Save cross-phase SSA values for this copy
            saved_cross_phase[k] = {}
            for ssa in cross_phase_ssas:
                if ssa in self.ssa_map:
                    v = self.ssa_map[ssa]
                    saved_cross_phase[k][ssa] = (v.ttype, v.expr)

        # Force flush all pending wave reductions (1 barrier for all N copies)
        self._flush_wave_reductions()

        # Phase 3 for all N copies (sequential due to iter_arg dependencies)
        for k in range(unroll):
            # Restore cross-phase SSAs for this copy
            for ssa, (ttype, expr) in saved_cross_phase[k].items():
                self._set_val(ssa, ttype, expr)

            # Restore reduce results for this copy
            for rssa in reduce_ssas:
                copy_ssa = f"{rssa}_u{k}"
                if copy_ssa in self.ssa_map:
                    v = self.ssa_map[copy_ssa]
                    self._set_val(rssa, v.ttype, v.expr)

            # Restore loop variable (Phase 3 ops may reference it)
            if k == 0:
                self._set_val(op.loop_var, TType(dtype='i32'), iv_name)
            else:
                iv_k = self._fresh_var("iv")
                self._emit(f"int {iv_k} = {iv_name} + {k * step_val};")
                self._set_val(op.loop_var, TType(dtype='i32'), iv_k)

            self._gen_ops_no_liveness(phase3_ops)
            self._process_loop_yields(op, iter_var_names, iter_tiles)

        self._flush_fused_loops()
        self.indent -= 1
        self._emit("}")

        # === Remainder loop (scalar, 1 iteration at a time) ===
        # Keep _in_unrolled_loop=True: _gen_counter has advanced past original
        # _last_use values, so _try_reuse_in_place would incorrectly alias registers.
        self._set_val(op.loop_var, TType(dtype='i32'), iv_name)
        self._emit(f"for (; {iv_name} < {end_expr}; {iv_name} += {step_expr}) {{")
        self.indent += 1

        if op.body_ops:
            self._gen_ops_no_liveness(op.body_ops)

        self._process_loop_yields(op, iter_var_names, iter_tiles)
        self._flush_fused_loops()
        self.indent -= 1
        self._emit("}")
        # Keep _in_unrolled_loop=True permanently: gen_counter has advanced
        # past original _last_use values, making liveness stale for all
        # subsequent code (post-loop ops, not just loop body).

    def _gen_scf_for_simple_unrolled(self, op, unroll, iv_name, start_expr, end_expr,
                                       step_expr, iter_var_names, iter_tiles):
        """Emit an unrolled for loop for simple accumulator loops (no wave reductions).

        Uses phase separation like wave_unrolled to enable load pipelining:
          Phase 1+2: loads and iter_arg-independent ops (all copies back-to-back)
          Phase 3:   accumulates and iter_arg-dependent ops (sequential)
        This groups memory loads together, giving the GPU shader compiler
        maximum opportunity to pipeline loads across the accumulate chain.
        """
        step_val = self._try_eval_int_expr(op.loop_step)
        phase12_ops, phase3_ops, reduce_ssas, cross_phase_ssas = \
            self._classify_body_for_unroll(op)

        self._flush_fused_loops()
        self._in_unrolled_loop = True

        # Declare IV, compute aligned end.  Only clamp when start could be
        # negative (flash attention sliding window).
        if self._start_may_be_negative(op):
            self._emit(f"int {iv_name} = max({start_expr}, 0);")
            step_n = step_val * unroll
            aligned = self._fresh_var("end_a")
            self._emit(f"int {aligned} = (({end_expr} - {iv_name}) / {step_n}) * {step_n} + {iv_name};")
        else:
            self._emit(f"int {iv_name} = {start_expr};")
            step_n = step_val * unroll
            aligned = self._fresh_var("end_a")
            self._emit(f"int {aligned} = (({end_expr} - {start_expr}) / {step_n}) * {step_n} + {start_expr};")

        # Main unrolled loop
        self._emit(f"for (; {iv_name} < {aligned}; {iv_name} += {step_n}) {{")
        self.indent += 1

        saved_cross_phase: dict[int, dict] = {}

        # Phase 1+2: loads + independent ops for all copies
        for k in range(unroll):
            if k == 0:
                self._set_val(op.loop_var, TType(dtype='i32'), iv_name)
            else:
                iv_k = self._fresh_var("iv")
                self._emit(f"int {iv_k} = {iv_name} + {k * step_val};")
                self._set_val(op.loop_var, TType(dtype='i32'), iv_k)

            self._unroll_copy_idx = k
            self._gen_ops_no_liveness(phase12_ops)
            self._unroll_copy_idx = None

            saved_cross_phase[k] = {}
            for ssa in cross_phase_ssas:
                if ssa in self.ssa_map:
                    v = self.ssa_map[ssa]
                    saved_cross_phase[k][ssa] = (v.ttype, v.expr)

        # Phase 3: accumulates for all copies (sequential, iter_arg dependent)
        for k in range(unroll):
            for ssa, (ttype, expr) in saved_cross_phase[k].items():
                self._set_val(ssa, ttype, expr)

            for rssa in reduce_ssas:
                copy_ssa = f"{rssa}_u{k}"
                if copy_ssa in self.ssa_map:
                    v = self.ssa_map[copy_ssa]
                    self._set_val(rssa, v.ttype, v.expr)

            if k == 0:
                self._set_val(op.loop_var, TType(dtype='i32'), iv_name)
            else:
                iv_k = self._fresh_var("iv")
                self._emit(f"int {iv_k} = {iv_name} + {k * step_val};")
                self._set_val(op.loop_var, TType(dtype='i32'), iv_k)

            self._gen_ops_no_liveness(phase3_ops)
            self._process_loop_yields(op, iter_var_names, iter_tiles)

        self._flush_fused_loops()
        self.indent -= 1
        self._emit("}")

        # Remainder loop (scalar, no phase separation)
        body_ops = [bop for bop in (op.body_ops or []) if bop.opname != 'scf.yield']
        self._set_val(op.loop_var, TType(dtype='i32'), iv_name)
        self._emit(f"for (; {iv_name} < {end_expr}; {iv_name} += {step_expr}) {{")
        self.indent += 1
        self._gen_ops_no_liveness(body_ops)
        self._process_loop_yields(op, iter_var_names, iter_tiles)
        self._flush_fused_loops()
        self.indent -= 1
        self._emit("}")

    def _find_dot_acc_iter_arg(self, for_op: Op) -> int | None:
        if not for_op.body_ops:
            return None
        for body_op in for_op.body_ops:
            if body_op.opname == 'tt.dot' and len(body_op.operands) >= 3:
                acc_name = body_op.operands[2]
                for i, arg_name in enumerate(for_op.iter_arg_names):
                    if arg_name == acc_name:
                        return i
        return None

    def _find_dot_op(self, for_op: Op) -> Op | None:
        for body_op in (for_op.body_ops or []):
            if body_op.opname == 'tt.dot':
                return body_op
        return None

    def _detect_dequant_pattern(self, for_op: Op, dot_op: Op) -> dict | None:
        """Detect W8A16/W4A16 dequant pattern: load(int) -> sitofp/uitofp -> [mulf] -> dot.

        Handles both direct (cast → dot) and indirect (cast → mulf(scale) → dot) chains.
        Returns dict with 'device_dtype' (e.g. 'i8'), 'cast_result' (SSA name
        of the sitofp result), 'operand_idx' (0=A, 1=B) if detected, else None.
        """
        if not dot_op or not for_op.body_ops:
            return None
        cast_ops = {'arith.sitofp', 'arith.uitofp'}
        _ALIAS_OPS = {'ttg.convert_layout', 'ttg.local_alloc',
                       'ttg.local_load', 'ttg.memdesc_trans'}
        # Build result → op map for tracing
        result_to_op = {}
        for body_op in for_op.body_ops:
            for r in body_op.results:
                result_to_op[r] = body_op

        def trace_back(ssa):
            """Trace through TTGIR alias/layout ops to find real source."""
            for _ in range(5):
                if ssa in result_to_op:
                    src_op = result_to_op[ssa]
                    if src_op.opname in _ALIAS_OPS and src_op.operands:
                        ssa = src_op.operands[0]
                        continue
                break
            return ssa

        for body_op in for_op.body_ops:
            if body_op.opname not in cast_ops or not body_op.results:
                continue
            cast_result = body_op.results[0]
            # Check if the cast result feeds into dot (directly or via mulf).
            # TTGIR may insert ttg.convert_layout between cast/mulf and dot,
            # so trace back through alias ops to find the real source.
            for idx in (0, 1):
                if idx >= len(dot_op.operands):
                    continue
                dot_input = dot_op.operands[idx]
                dot_src = trace_back(dot_input)
                scale_ssa = None
                # Direct: cast → [convert_layout →] dot
                if dot_src == cast_result:
                    pass  # match
                # Indirect: cast → mulf(scale) → [convert_layout →] dot
                elif dot_src in result_to_op:
                    mulf_op = result_to_op[dot_src]
                    if mulf_op.opname == 'arith.mulf' and cast_result in mulf_op.operands:
                        # Capture the scale operand (the one that isn't the cast result)
                        for mop in mulf_op.operands:
                            if mop != cast_result:
                                scale_ssa = mop
                                break
                    else:
                        continue
                else:
                    continue
                # Extract the source dtype from the cast's type annotation or operand type
                src_dtype = None
                if body_op.type_str:
                    m = re.search(r'tensor<[\dx]+x(\w+)>\s+to\s+', body_op.type_str)
                    if m:
                        src_dtype = m.group(1)
                # Fallback: infer from operand's result type (walker may not set type_str)
                if src_dtype is None and body_op.operands:
                    src_ssa = body_op.operands[0]
                    src_op = result_to_op.get(src_ssa)
                    if src_op and src_op.result_types:
                        dt = src_op.result_types[0].dtype
                        if dt and dt.startswith('i'):
                            src_dtype = dt
                if src_dtype and src_dtype.startswith('i'):
                    info = {
                        'device_dtype': src_dtype,
                        'cast_result': cast_result,
                        'operand_idx': idx,
                    }
                    if scale_ssa is not None:
                        # Trace through tt.splat/tt.broadcast to get
                        # the underlying scalar (defined outside the loop)
                        s = scale_ssa
                        for _ in range(5):  # max depth
                            if s in result_to_op:
                                s_op = result_to_op[s]
                                if s_op.opname in ('tt.splat', 'tt.broadcast') and s_op.operands:
                                    s = s_op.operands[0]
                                    continue
                            break
                        info['scale_ssa'] = s
                    return info
        return None

    def _detect_reg_acc_chain(self, for_op: Op, arg_name: str, arg_idx: int):
        """Detect if iter_arg is a dot accumulator eligible for register storage.

        Returns dict with 'mulf_result' (or None) and 'dot_result' if detected,
        else None.  Pattern: arg → [mulf(arg, broadcast)] → dot(A, B, acc) → yield.
        """
        if not for_op.body_ops:
            return None
        # Build result→op map for body
        result_to_op = {}
        for body_op in for_op.body_ops:
            for r in body_op.results:
                result_to_op[r] = body_op

        # Case 1: direct — dot(A, B, arg_name)
        for body_op in for_op.body_ops:
            if (body_op.opname == 'tt.dot' and len(body_op.operands) >= 3
                    and body_op.operands[2] == arg_name):
                # Check yield
                dot_result = body_op.results[0]
                if self._chain_to_yield(dot_result, for_op, arg_idx, result_to_op):
                    return {'mulf_result': None, 'dot_result': dot_result}

        # Case 2: via mulf — mulf(arg, broadcast) → dot(A, B, mulf_result)
        for body_op in for_op.body_ops:
            if body_op.opname == 'arith.mulf' and len(body_op.operands) >= 2:
                if arg_name not in body_op.operands:
                    continue
                mulf_result = body_op.results[0]
                for body_op2 in for_op.body_ops:
                    if (body_op2.opname == 'tt.dot' and len(body_op2.operands) >= 3
                            and body_op2.operands[2] == mulf_result):
                        dot_result = body_op2.results[0]
                        if self._chain_to_yield(dot_result, for_op, arg_idx, result_to_op):
                            return {'mulf_result': mulf_result, 'dot_result': dot_result}
        return None

    def _chain_to_yield(self, ssa_name: str, for_op: Op, yield_idx: int,
                         result_to_op: dict) -> bool:
        """Check if ssa_name chains (possibly through element-wise ops) to yield[yield_idx]."""
        if not hasattr(for_op, 'yield_operands') or not for_op.yield_operands:
            return False
        if yield_idx >= len(for_op.yield_operands):
            return False
        target = for_op.yield_operands[yield_idx]
        # Direct match
        if ssa_name == target:
            return True
        # Follow single-use chain through element-wise ops
        visited = {ssa_name}
        current = ssa_name
        for _ in range(20):  # limit depth
            found_next = False
            for body_op in for_op.body_ops:
                if current in body_op.operands and body_op.results:
                    next_name = body_op.results[0]
                    if next_name not in visited:
                        visited.add(next_name)
                        current = next_name
                        found_next = True
                        if current == target:
                            return True
                        break
            if not found_next:
                break
        return False

    def _gen_scf_if(self, op: Op):
        self._flush_fused_loops()
        cond_expr = self._get_expr(op.operands[0])

        # Check if scf.if produces results (has yields)
        has_yields = len(op.results) > 0 and len(op.yield_operands) > 0
        if_result_vars = []

        if has_yields:
            # Declare a temp variable for each scf.if result
            for res_ssa, res_type in zip(op.results, op.result_types):
                var = self._fresh_var("if_res")
                # For tensor types (register tiles), use the element dtype
                if res_type.shape:
                    c_type = self.emitter.map_dtype(res_type.dtype)
                else:
                    c_type = self.emitter.map_dtype(res_type.dtype)
                if_result_vars.append((var, res_type))
                self._tile_decls.append(f"{c_type} {var};")

        self._emit(f"if ({cond_expr}) {{")
        self.indent += 1

        # Process then-body ops (scf.yield already excluded by walker)
        for body_op in (op.body_ops or []):
            self._gen_op(body_op)

        # Assign then-branch yields to temp vars
        if has_yields and op.yield_operands:
            self._flush_fused_loops()
            for i, yield_ssa in enumerate(op.yield_operands):
                var, _ = if_result_vars[i]
                yield_expr = self._get_expr(yield_ssa)
                self._emit(f"{var} = {yield_expr};")

        self.indent -= 1

        # Else block
        if op.else_ops or (has_yields and op.else_yield_operands):
            self._emit("} else {")
            self.indent += 1

            # Process else-body ops (scf.yield already excluded by walker)
            for else_op in (op.else_ops or []):
                self._gen_op(else_op)

            # Assign else-branch yields to temp vars
            if has_yields and op.else_yield_operands:
                for i, yield_ssa in enumerate(op.else_yield_operands):
                    var, _ = if_result_vars[i]
                    yield_expr = self._get_expr(yield_ssa)
                    self._emit(f"{var} = {yield_expr};")

            self.indent -= 1
        self._emit("}")

        # Map scf.if results to temp vars in SSA map
        if has_yields:
            for i, (res_ssa, (var, res_type)) in enumerate(
                    zip(op.results, if_result_vars)):
                # For tensor types, use element dtype for the scalar SSA value
                if res_type.shape:
                    scalar_type = TType(dtype=res_type.dtype)
                else:
                    scalar_type = res_type
                self._set_val(res_ssa, scalar_type, var)

    def _gen_scf_yield(self, op: Op):
        # Yields inside scf.if are handled by _gen_scf_if.
        # Yields inside scf.for are handled by _process_loop_yields.
        pass

    # --- GPU barrier ops ---

    def _gen_gpu_barrier(self, op: Op):
        self._emit(self.emitter.barrier())

    # --- Ignored ops ---

    def _gen_ub_poison(self, op: Op):
        pass
