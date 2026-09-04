#!/usr/bin/env python3
"""Benchmark matmul kernel execution on Metal, with data already in GPU buffers.

Compares:
  1. Codegen-generated kernels (TTIR -> MSL via codegen.py template)
  2. MPS baseline (PyTorch's torch.matmul on MPS device)
"""
import sys, os, struct, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Metal
from neso.backend.codegen import ttir_to_msl

device = Metal.MTLCreateSystemDefaultDevice()
queue = device.newCommandQueue()


def make_buffer(data, dtype='f'):
    """Create a Metal shared buffer from a list of values."""
    raw = struct.pack(f'{len(data)}{dtype}', *data)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)


def make_zero_buffer(n, dtype='f'):
    """Create a zeroed Metal shared buffer."""
    sz = n * struct.calcsize(dtype)
    return device.newBufferWithLength_options_(sz, Metal.MTLResourceStorageModeShared)


def scalar_buf(val, dtype='i'):
    raw = struct.pack(dtype, val)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)


def read_buffer(buf, n, dtype='f'):
    raw = buf.contents().as_buffer(n * struct.calcsize(dtype))
    return list(struct.unpack(f'{n}{dtype}', raw))


def compile_kernel(msl_source, name):
    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)
    lib, err = device.newLibraryWithSource_options_error_(msl_source, options, None)
    if err:
        raise RuntimeError(f"MSL compile error:\n{err.localizedDescription()}\n\n{msl_source}")
    fn = lib.newFunctionWithName_(name)
    pipe, err = device.newComputePipelineStateWithFunction_error_(fn, None)
    if err:
        raise RuntimeError(f"Pipeline error: {err.localizedDescription()}")
    return pipe


def dispatch(pipe, grid, threads_per_group, buffers):
    cb = queue.commandBuffer()
    enc = cb.computeCommandEncoder()
    enc.setComputePipelineState_(pipe)
    for i, buf in enumerate(buffers):
        enc.setBuffer_offset_atIndex_(buf, 0, i)
    enc.dispatchThreadgroups_threadsPerThreadgroup_(
        Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(threads_per_group, 1, 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return cb


def bench_matmul(M, N, K, BM, BN, BK, iters=20, use_simdgroup=True, use_half=False):
    """Benchmark a tiled matmul kernel via codegen (TTIR -> MSL template)."""

    # fp16: half inputs, float accumulator, half output (mixed precision)
    # fp32: float everything
    in_t = 'f16' if use_half else 'f32'
    acc_t = 'f32'  # always accumulate in f32
    c_t = 'f16' if use_half else 'f32'

    ttir = f"""
    module {{
      tt.func public @matmul_kernel(
        %a_ptr: !tt.ptr<{in_t}>, %b_ptr: !tt.ptr<{in_t}>, %c_ptr: !tt.ptr<{c_t}>,
        %K_param: i32, %stride_am: i32, %stride_bk: i32, %stride_cm: i32
      ) {{
        %c0 = arith.constant 0 : i32
        %cst_BK = arith.constant {BK} : i32
        %cst_BM = arith.constant {BM} : i32
        %cst_BN = arith.constant {BN} : i32
        %cst_acc = arith.constant dense<0.000000e+00> : tensor<{BM}x{BN}x{acc_t}>

        %pid_m = tt.get_program_id x : i32
        %pid_n = tt.get_program_id y : i32

        %off_m_base = arith.muli %pid_m, %cst_BM : i32
        %range_m = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %off_m_splat = tt.splat %off_m_base : i32 -> tensor<{BM}xi32>
        %offs_m = arith.addi %off_m_splat, %range_m : tensor<{BM}xi32>

        %off_n_base = arith.muli %pid_n, %cst_BN : i32
        %range_n = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %off_n_splat = tt.splat %off_n_base : i32 -> tensor<{BN}xi32>
        %offs_n = arith.addi %off_n_splat, %range_n : tensor<{BN}xi32>

        %range_k = tt.make_range {{start = 0 : i32, end = {BK} : i32}} : tensor<{BK}xi32>

        %row = tt.expand_dims %offs_m {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %stride_am_splat = tt.splat %stride_am : i32 -> tensor<{BM}x1xi32>
        %row_off = arith.muli %row, %stride_am_splat : tensor<{BM}x1xi32>
        %col_k = tt.expand_dims %range_k {{axis = 0 : i32}} : tensor<{BK}xi32> -> tensor<1x{BK}xi32>
        %row_off_bc = tt.broadcast %row_off : tensor<{BM}x1xi32> -> tensor<{BM}x{BK}xi32>
        %col_k_bc = tt.broadcast %col_k : tensor<1x{BK}xi32> -> tensor<{BM}x{BK}xi32>
        %a_idx = arith.addi %row_off_bc, %col_k_bc : tensor<{BM}x{BK}xi32>
        %a_base = tt.splat %a_ptr : !tt.ptr<{in_t}> -> tensor<{BM}x{BK}x!tt.ptr<{in_t}>>
        %a_ptrs_init = tt.addptr %a_base, %a_idx : tensor<{BM}x{BK}x!tt.ptr<{in_t}>>, tensor<{BM}x{BK}xi32>

        %k_row = tt.expand_dims %range_k {{axis = 1 : i32}} : tensor<{BK}xi32> -> tensor<{BK}x1xi32>
        %stride_bk_splat = tt.splat %stride_bk : i32 -> tensor<{BK}x1xi32>
        %k_row_off = arith.muli %k_row, %stride_bk_splat : tensor<{BK}x1xi32>
        %col_n = tt.expand_dims %offs_n {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %k_row_off_bc = tt.broadcast %k_row_off : tensor<{BK}x1xi32> -> tensor<{BK}x{BN}xi32>
        %col_n_bc = tt.broadcast %col_n : tensor<1x{BN}xi32> -> tensor<{BK}x{BN}xi32>
        %b_idx = arith.addi %k_row_off_bc, %col_n_bc : tensor<{BK}x{BN}xi32>
        %b_base = tt.splat %b_ptr : !tt.ptr<{in_t}> -> tensor<{BK}x{BN}x!tt.ptr<{in_t}>>
        %b_ptrs_init = tt.addptr %b_base, %b_idx : tensor<{BK}x{BN}x!tt.ptr<{in_t}>>, tensor<{BK}x{BN}xi32>

        %cst_a_adv = arith.constant dense<{BK}> : tensor<{BM}x{BK}xi32>
        %b_adv_scalar = arith.muli %stride_bk, %cst_BK : i32
        %b_adv_splat = tt.splat %b_adv_scalar : i32 -> tensor<{BK}x{BN}xi32>

        %results:3 = scf.for %iv = %c0 to %K_param step %cst_BK
            iter_args(%a_ptrs = %a_ptrs_init, %b_ptrs = %b_ptrs_init, %acc = %cst_acc)
            -> (tensor<{BM}x{BK}x!tt.ptr<{in_t}>>, tensor<{BK}x{BN}x!tt.ptr<{in_t}>>, tensor<{BM}x{BN}x{acc_t}>) : i32 {{
          %a = tt.load %a_ptrs : tensor<{BM}x{BK}x!tt.ptr<{in_t}>>
          %b = tt.load %b_ptrs : tensor<{BK}x{BN}x!tt.ptr<{in_t}>>
          %d = tt.dot %a, %b, %acc : tensor<{BM}x{BK}x{in_t}> * tensor<{BK}x{BN}x{in_t}> -> tensor<{BM}x{BN}x{acc_t}>
          %a_ptrs_next = tt.addptr %a_ptrs, %cst_a_adv : tensor<{BM}x{BK}x!tt.ptr<{in_t}>>, tensor<{BM}x{BK}xi32>
          %b_ptrs_next = tt.addptr %b_ptrs, %b_adv_splat : tensor<{BK}x{BN}x!tt.ptr<{in_t}>>, tensor<{BK}x{BN}xi32>
          scf.yield %a_ptrs_next, %b_ptrs_next, %d : tensor<{BM}x{BK}x!tt.ptr<{in_t}>>, tensor<{BK}x{BN}x!tt.ptr<{in_t}>>, tensor<{BM}x{BN}x{acc_t}>
        }}

        %c_row = tt.expand_dims %offs_m {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %stride_cm_splat = tt.splat %stride_cm : i32 -> tensor<{BM}x1xi32>
        %c_row_off = arith.muli %c_row, %stride_cm_splat : tensor<{BM}x1xi32>
        %c_col = tt.expand_dims %offs_n {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %c_row_off_bc = tt.broadcast %c_row_off : tensor<{BM}x1xi32> -> tensor<{BM}x{BN}xi32>
        %c_col_bc = tt.broadcast %c_col : tensor<1x{BN}xi32> -> tensor<{BM}x{BN}xi32>
        %c_idx = arith.addi %c_row_off_bc, %c_col_bc : tensor<{BM}x{BN}xi32>
        %c_base = tt.splat %c_ptr : !tt.ptr<{c_t}> -> tensor<{BM}x{BN}x!tt.ptr<{c_t}>>
        %c_ptrs = tt.addptr %c_base, %c_idx : tensor<{BM}x{BN}x!tt.ptr<{c_t}>>, tensor<{BM}x{BN}xi32>
        {"" if acc_t == c_t else f"%results_cast = arith.truncf %results#2 : tensor<{BM}x{BN}x{acc_t}> to tensor<{BM}x{BN}x{c_t}>"}
        tt.store %c_ptrs, {"%" + "results_cast" if acc_t != c_t else "%results#2"} : tensor<{BM}x{BN}x!tt.ptr<{c_t}>>
        tt.return
      }}
    }}
    """

    msl, name = ttir_to_msl(ttir, block_size=BM * BN, use_simdgroup=use_simdgroup)
    pipe = compile_kernel(msl, name)

    threads = min(BM * BN, 1024)
    grid = (M // BM, N // BN, 1)

    buf_dtype = 'e' if use_half else 'f'  # 'e' = IEEE 754 half

    # Random data in Metal buffers (no MPS involved)
    import random
    random.seed(42)
    a_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    b_data = [random.gauss(0, 0.5) for _ in range(K * N)]
    a_buf = make_buffer(a_data, buf_dtype)
    b_buf = make_buffer(b_data, buf_dtype)
    c_buf = make_zero_buffer(M * N, buf_dtype)
    k_buf = scalar_buf(K)
    sam_buf = scalar_buf(K)      # stride_am = K (row-major A)
    sbk_buf = scalar_buf(N)      # stride_bk = N (row-major B)
    scm_buf = scalar_buf(N)      # stride_cm = N (row-major C)

    bufs = [a_buf, b_buf, c_buf, k_buf, sam_buf, sbk_buf, scm_buf]

    # Warmup
    for _ in range(3):
        dispatch(pipe, grid, threads, bufs)

    # Timed runs
    gpu_times = []
    for _ in range(iters):
        # Zero C between runs
        c_buf = make_zero_buffer(M * N, buf_dtype)
        bufs[2] = c_buf
        cb = dispatch(pipe, grid, threads, bufs)
        gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
        gpu_times.append(gpu_ms)

    avg = sum(gpu_times) / len(gpu_times)
    mn = min(gpu_times)
    flops = 2.0 * M * N * K
    gflops_avg = flops / (avg / 1000.0) / 1e9 if avg > 0 else 0
    gflops_peak = flops / (mn / 1000.0) / 1e9 if mn > 0 else 0

    # Verify correctness (spot check)
    c_vals = read_buffer(c_buf, M * N, buf_dtype)
    # Compute expected C[0][0] = sum_k A[0][k] * B[k][0]
    expected_00 = sum(a_data[k] * b_data[k * N] for k in range(K))
    err = abs(c_vals[0] - expected_00)
    tol = K * 5e-3 if use_half else K * 1e-4
    ok = err < tol

    return avg, mn, gflops_avg, gflops_peak, ok, err


def bench_mps(M, N, K, iters=20, use_half=False):
    """Benchmark PyTorch MPS matmul as a reference baseline."""
    import torch
    torch.mps.synchronize()
    dtype = torch.float16 if use_half else torch.float32
    a = torch.randn(M, K, device='mps', dtype=dtype)
    b = torch.randn(K, N, device='mps', dtype=dtype)
    # Warmup
    for _ in range(5):
        c = a @ b
    torch.mps.synchronize()

    times = []
    for _ in range(iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg = sum(times) / len(times)
    mn = min(times)
    flops = 2.0 * M * N * K
    return avg, mn, flops / (avg / 1000.0) / 1e9, flops / (mn / 1000.0) / 1e9


import os
use_simdgroup = os.environ.get("NESO_SIMDGROUP", "").lower() not in ("0", "false", "no")
mode_str = "simdgroup_matrix" if use_simdgroup else "scalar"

print(f"Metal Device: {device.name()}")
print(f"Mode: {mode_str}")
print()

has_torch = True
try:
    import torch
except ImportError:
    has_torch = False

has_mps = has_torch and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

sizes = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]

tile_configs = [
    ( 512,  512,  512, 32, 32, 32),
    (1024, 1024, 1024, 32, 32, 32),
    (2048, 2048, 2048, 32, 32, 32),
    ( 512,  512,  512, 64, 64, 16),
    (1024, 1024, 1024, 64, 64, 16),
    (2048, 2048, 2048, 64, 64, 16),
    (1024, 1024, 1024, 64, 64, 32),
    (2048, 2048, 2048, 64, 64, 32),
    (4096, 4096, 4096, 64, 64, 16),
    # Larger tiles — disabled: these crash the Metal compiler XPC service
    # after many compilations in a single process.  Re-enable individually
    # when benchmarking 128x128 in isolation.
    # (1024, 1024, 1024, 128, 128, 32),
    # (2048, 2048, 2048, 128, 128, 32),
    # (4096, 4096, 4096, 128, 128, 32),
    # (2048, 2048, 2048, 128, 128, 16),
    # (4096, 4096, 4096, 128, 128, 16),
]

for use_half in [False, True]:
    dtype_str = "fp16" if use_half else "fp32"

    # --- MPS baseline ---
    mps_results = {}
    if has_mps:
        print(f"=== MPS baseline (torch.matmul, {dtype_str}) ===")
        print(f"{'Size':>14s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'Avg GFLOP/s':>11s}  {'Peak GFLOP/s':>12s}")
        print("-" * 60)

        for M, N, K in sizes:
            avg, mn, gf_avg, gf_peak = bench_mps(M, N, K, use_half=use_half)
            mps_results[(M, N, K)] = gf_peak
            print(f"  {M}x{N}x{K:>4d}  {avg:>8.3f}  {mn:>8.3f}  {gf_avg:>11.1f}  {gf_peak:>12.1f}")
    else:
        print(f"(MPS not available, skipping torch baseline for {dtype_str})")

    # --- Codegen-generated kernels ---
    print()
    print(f"=== Triton codegen ({mode_str}, {dtype_str}) ===")
    print(f"{'Size':>14s}  {'Tile':>10s}  {'Grid':>7s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}  {'vs MPS':>7s}  {'Err':>8s}  {'OK':>3s}")
    print("-" * 85)

    for M, N, K, BM, BN, BK in tile_configs:
        grid_str = f"{M//BM}x{N//BN}"
        tile_str = f"{BM}x{BN}x{BK}"
        try:
            avg, mn, gf_avg, gf_peak, ok, err = bench_matmul(
                M, N, K, BM, BN, BK, use_simdgroup=use_simdgroup, use_half=use_half)
        except RuntimeError as e:
            msg = str(e)
            if 'XPC_ERROR' in msg or 'interrupted' in msg.lower():
                print(f"  {M}x{N}x{K:>4d}  {tile_str:>10s}  {grid_str:>7s}  (Metal compiler service crashed, skipping)")
                continue
            if 'threadgroup memory' in msg.lower() or 'exceeds the maximum' in msg.lower():
                print(f"  {M}x{N}x{K:>4d}  {tile_str:>10s}  {grid_str:>7s}  (exceeds 32KB threadgroup memory, skipping)")
                continue
            raise
        mps_peak = mps_results.get((M, N, K), 0)
        ratio = f"{gf_peak/mps_peak:.0%}" if mps_peak > 0 else "N/A"
        print(f"  {M}x{N}x{K:>4d}  {tile_str:>10s}  {grid_str:>7s}  {avg:>8.3f}  {mn:>8.3f}  {gf_peak:>9.1f}  {ratio:>7s}  {err:>8.4f}  {'Y' if ok else 'N'}")
    print()
