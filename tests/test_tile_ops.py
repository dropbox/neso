#!/usr/bin/env python3
"""Test tile-based operations needed for Flash Attention 2.

Tests: 2D load/store, reduce (rowmax, rowsum), broadcast, element-wise on tiles,
       tt.trans (transpose), tt.dot with transposed input.
"""
import sys, os, struct, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Metal
from neso.backend.codegen import ttir_to_msl

device = Metal.MTLCreateSystemDefaultDevice()
queue = device.newCommandQueue()


def make_buffer(data, dtype='f'):
    raw = struct.pack(f'{len(data)}{dtype}', *data)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)


def make_zero_buffer(n, dtype='f'):
    sz = n * struct.calcsize(dtype)
    return device.newBufferWithLength_options_(sz, Metal.MTLResourceStorageModeShared)


def scalar_buf(val, dtype='i'):
    raw = struct.pack(dtype, val)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)


def float_buf(val):
    raw = struct.pack('f', val)
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


# ============================================================
# Test 1: 2D Softmax (load tile, rowmax, sub, exp, rowsum, div, store)
# ============================================================
def test_2d_softmax():
    """Tests: 2D load/store, reduce(max,axis=1), reduce(add,axis=1),
    broadcast 1D->2D, element-wise sub/exp/div on tiles."""
    BM, BN = 8, 32  # BN=32 = SIMD width for fast reduce
    ttir = f"""
    module {{
      tt.func public @softmax_2d(
        %in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>,
        %stride: i32
      ) {{
        %c0 = arith.constant 0 : i32
        %cBM = arith.constant {BM} : i32

        %pid = tt.get_program_id x : i32
        %off_m_base = arith.muli %pid, %cBM : i32

        // Build 2D pointers for input [BM, BN]
        %range_m = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %off_m_splat = tt.splat %off_m_base : i32 -> tensor<{BM}xi32>
        %offs_m = arith.addi %off_m_splat, %range_m : tensor<{BM}xi32>

        %range_n = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>

        %row = tt.expand_dims %offs_m {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %stride_splat = tt.splat %stride : i32 -> tensor<{BM}x1xi32>
        %row_off = arith.muli %row, %stride_splat : tensor<{BM}x1xi32>
        %col = tt.expand_dims %range_n {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %row_off_bc = tt.broadcast %row_off : tensor<{BM}x1xi32> -> tensor<{BM}x{BN}xi32>
        %col_bc = tt.broadcast %col : tensor<1x{BN}xi32> -> tensor<{BM}x{BN}xi32>
        %idx = arith.addi %row_off_bc, %col_bc : tensor<{BM}x{BN}xi32>
        %in_base = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<{BM}x{BN}x!tt.ptr<f32>>
        %in_ptrs = tt.addptr %in_base, %idx : tensor<{BM}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xi32>
        %x = tt.load %in_ptrs : tensor<{BM}x{BN}x!tt.ptr<f32>>

        // Row-wise max: [BM, BN] -> [BM]
        %row_max = "tt.reduce"(%x) <{{axis = 1 : i32}}> ({{
        ^bb0(%lhs: f32, %rhs: f32):
          %max = arith.maxnumf %lhs, %rhs : f32
          tt.reduce.return %max : f32
        }}) : (tensor<{BM}x{BN}xf32>) -> tensor<{BM}xf32>

        // Broadcast max back: [BM] -> [BM, 1] -> [BM, BN]
        %max_exp = tt.expand_dims %row_max {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
        %max_bc = tt.broadcast %max_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{BN}xf32>

        // x - max
        %shifted = arith.subf %x, %max_bc : tensor<{BM}x{BN}xf32>

        // exp(x - max)
        %ex = math.exp %shifted : tensor<{BM}x{BN}xf32>

        // Row-wise sum: [BM, BN] -> [BM]
        %row_sum = "tt.reduce"(%ex) <{{axis = 1 : i32}}> ({{
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          tt.reduce.return %sum : f32
        }}) : (tensor<{BM}x{BN}xf32>) -> tensor<{BM}xf32>

        // Broadcast sum: [BM] -> [BM, BN]
        %sum_exp = tt.expand_dims %row_sum {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
        %sum_bc = tt.broadcast %sum_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{BN}xf32>

        // exp(x-max) / sum
        %result = arith.divf %ex, %sum_bc : tensor<{BM}x{BN}xf32>

        // Store
        %out_base = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<{BM}x{BN}x!tt.ptr<f32>>
        %out_ptrs = tt.addptr %out_base, %idx : tensor<{BM}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xi32>
        tt.store %out_ptrs, %result : tensor<{BM}x{BN}x!tt.ptr<f32>>
        tt.return
      }}
    }}
    """

    msl, name = ttir_to_msl(ttir, block_size=BM * BN)
    # print("--- 2D Softmax MSL ---")
    # print(msl)

    pipe = compile_kernel(msl, name)

    import random
    random.seed(42)
    M, N = 16, BN  # 2 blocks of BM=8
    data = [random.gauss(0, 1.0) for _ in range(M * N)]
    in_buf = make_buffer(data)
    out_buf = make_zero_buffer(M * N)
    stride_buf = scalar_buf(N)

    dispatch(pipe, (M // BM, 1, 1), min(BM * BN, 1024), [in_buf, out_buf, stride_buf])

    result = read_buffer(out_buf, M * N)

    # Reference softmax
    errors = 0
    max_err = 0
    for row in range(M):
        row_data = data[row * N:(row + 1) * N]
        mx = max(row_data)
        exps = [math.exp(v - mx) for v in row_data]
        s = sum(exps)
        ref = [e / s for e in exps]
        for col in range(N):
            err = abs(result[row * N + col] - ref[col])
            max_err = max(max_err, err)
            if err > 1e-4:
                errors += 1
                if errors <= 3:
                    print(f"  Mismatch [{row},{col}]: got {result[row*N+col]:.6f}, expected {ref[col]:.6f}, err={err:.6f}")

    ok = errors == 0
    print(f"  2D Softmax: {'PASS' if ok else 'FAIL'} (max_err={max_err:.6f}, {M}x{N})")
    return ok


# ============================================================
# Test 2: Transpose + Dot (Q @ K^T via tile path)
# ============================================================
def test_transpose_dot():
    """Tests: tt.trans on tiles, tt.dot with transposed B input."""
    BM, BK, BN = 8, 16, 8  # Small for testing
    # Q[BM, BK] @ K^T[BK, BN] where K is [BN, BK]
    # So we load K[BN, BK], transpose to K^T[BK, BN], then dot Q * K^T -> C[BM, BN]
    ttir = f"""
    module {{
      tt.func public @trans_dot_kernel(
        %q_ptr: !tt.ptr<f32>, %k_ptr: !tt.ptr<f32>, %c_ptr: !tt.ptr<f32>,
        %stride_q: i32, %stride_k: i32, %stride_c: i32
      ) {{
        %pid_m = tt.get_program_id x : i32
        %pid_n = tt.get_program_id y : i32
        %cBM = arith.constant {BM} : i32
        %cBN = arith.constant {BN} : i32

        // -- Load Q tile [BM, BK] --
        %off_m_base = arith.muli %pid_m, %cBM : i32
        %range_m = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %off_m_splat = tt.splat %off_m_base : i32 -> tensor<{BM}xi32>
        %offs_m = arith.addi %off_m_splat, %range_m : tensor<{BM}xi32>
        %range_k = tt.make_range {{start = 0 : i32, end = {BK} : i32}} : tensor<{BK}xi32>

        %q_row = tt.expand_dims %offs_m {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %q_stride_splat = tt.splat %stride_q : i32 -> tensor<{BM}x1xi32>
        %q_row_off = arith.muli %q_row, %q_stride_splat : tensor<{BM}x1xi32>
        %q_col = tt.expand_dims %range_k {{axis = 0 : i32}} : tensor<{BK}xi32> -> tensor<1x{BK}xi32>
        %q_row_off_bc = tt.broadcast %q_row_off : tensor<{BM}x1xi32> -> tensor<{BM}x{BK}xi32>
        %q_col_bc = tt.broadcast %q_col : tensor<1x{BK}xi32> -> tensor<{BM}x{BK}xi32>
        %q_idx = arith.addi %q_row_off_bc, %q_col_bc : tensor<{BM}x{BK}xi32>
        %q_base = tt.splat %q_ptr : !tt.ptr<f32> -> tensor<{BM}x{BK}x!tt.ptr<f32>>
        %q_ptrs = tt.addptr %q_base, %q_idx : tensor<{BM}x{BK}x!tt.ptr<f32>>, tensor<{BM}x{BK}xi32>
        %Q = tt.load %q_ptrs : tensor<{BM}x{BK}x!tt.ptr<f32>>

        // -- Load K tile [BN, BK] --
        %off_n_base = arith.muli %pid_n, %cBN : i32
        %range_n = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %off_n_splat = tt.splat %off_n_base : i32 -> tensor<{BN}xi32>
        %offs_n = arith.addi %off_n_splat, %range_n : tensor<{BN}xi32>

        %k_row = tt.expand_dims %offs_n {{axis = 1 : i32}} : tensor<{BN}xi32> -> tensor<{BN}x1xi32>
        %k_stride_splat = tt.splat %stride_k : i32 -> tensor<{BN}x1xi32>
        %k_row_off = arith.muli %k_row, %k_stride_splat : tensor<{BN}x1xi32>
        %k_col = tt.expand_dims %range_k {{axis = 0 : i32}} : tensor<{BK}xi32> -> tensor<1x{BK}xi32>
        %k_row_off_bc = tt.broadcast %k_row_off : tensor<{BN}x1xi32> -> tensor<{BN}x{BK}xi32>
        %k_col_bc = tt.broadcast %k_col : tensor<1x{BK}xi32> -> tensor<{BN}x{BK}xi32>
        %k_idx = arith.addi %k_row_off_bc, %k_col_bc : tensor<{BN}x{BK}xi32>
        %k_base = tt.splat %k_ptr : !tt.ptr<f32> -> tensor<{BN}x{BK}x!tt.ptr<f32>>
        %k_ptrs = tt.addptr %k_base, %k_idx : tensor<{BN}x{BK}x!tt.ptr<f32>>, tensor<{BN}x{BK}xi32>
        %K = tt.load %k_ptrs : tensor<{BN}x{BK}x!tt.ptr<f32>>

        // -- Transpose K: [BN, BK] -> [BK, BN] --
        %KT = tt.trans %K {{order = array<i32: 1, 0>}} : tensor<{BN}x{BK}xf32> -> tensor<{BK}x{BN}xf32>

        // -- Dot: Q[BM, BK] @ K^T[BK, BN] -> C[BM, BN] --
        %zero = arith.constant dense<0.000000e+00> : tensor<{BM}x{BN}xf32>
        %C = tt.dot %Q, %KT, %zero : tensor<{BM}x{BK}xf32> * tensor<{BK}x{BN}xf32> -> tensor<{BM}x{BN}xf32>

        // -- Store C tile [BM, BN] --
        %off_m_c_base = arith.muli %pid_m, %cBM : i32
        %c_range_m = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
        %c_off_m_splat = tt.splat %off_m_c_base : i32 -> tensor<{BM}xi32>
        %c_offs_m = arith.addi %c_off_m_splat, %c_range_m : tensor<{BM}xi32>
        %off_n_c_base = arith.muli %pid_n, %cBN : i32
        %c_range_n = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
        %c_off_n_splat = tt.splat %off_n_c_base : i32 -> tensor<{BN}xi32>
        %c_offs_n = arith.addi %c_off_n_splat, %c_range_n : tensor<{BN}xi32>

        %c_row = tt.expand_dims %c_offs_m {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
        %c_stride_splat = tt.splat %stride_c : i32 -> tensor<{BM}x1xi32>
        %c_row_off = arith.muli %c_row, %c_stride_splat : tensor<{BM}x1xi32>
        %c_col = tt.expand_dims %c_offs_n {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
        %c_row_off_bc = tt.broadcast %c_row_off : tensor<{BM}x1xi32> -> tensor<{BM}x{BN}xi32>
        %c_col_bc = tt.broadcast %c_col : tensor<1x{BN}xi32> -> tensor<{BM}x{BN}xi32>
        %c_idx = arith.addi %c_row_off_bc, %c_col_bc : tensor<{BM}x{BN}xi32>
        %c_base = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<{BM}x{BN}x!tt.ptr<f32>>
        %c_ptrs = tt.addptr %c_base, %c_idx : tensor<{BM}x{BN}x!tt.ptr<f32>>, tensor<{BM}x{BN}xi32>
        tt.store %c_ptrs, %C : tensor<{BM}x{BN}x!tt.ptr<f32>>
        tt.return
      }}
    }}
    """

    msl, name = ttir_to_msl(ttir, block_size=max(BM * BK, BM * BN))
    # print("--- Trans+Dot MSL ---")
    # print(msl)

    pipe = compile_kernel(msl, name)

    import random
    random.seed(123)
    M, K, N = BM, BK, BN  # Single block for simplicity
    q_data = [random.gauss(0, 0.5) for _ in range(M * K)]
    k_data = [random.gauss(0, 0.5) for _ in range(N * K)]  # K is [N, K]

    q_buf = make_buffer(q_data)
    k_buf = make_buffer(k_data)
    c_buf = make_zero_buffer(M * N)
    stride_q_buf = scalar_buf(K)
    stride_k_buf = scalar_buf(K)
    stride_c_buf = scalar_buf(N)

    threads = min(BM * BN, 1024)  # Must match block_size from dot result shape
    dispatch(pipe, (1, 1, 1), threads,
             [q_buf, k_buf, c_buf, stride_q_buf, stride_k_buf, stride_c_buf])

    result = read_buffer(c_buf, M * N)

    # Reference: C = Q @ K^T where Q[M,K], K[N,K], so C[i,j] = sum_k Q[i,k]*K[j,k]
    max_err = 0
    errors = 0
    for i in range(M):
        for j in range(N):
            expected = sum(q_data[i * K + kk] * k_data[j * K + kk] for kk in range(K))
            err = abs(result[i * N + j] - expected)
            max_err = max(max_err, err)
            if err > 1e-3:
                errors += 1
                if errors <= 3:
                    print(f"  Mismatch [{i},{j}]: got {result[i*N+j]:.6f}, expected {expected:.6f}")

    ok = errors == 0
    print(f"  Trans+Dot: {'PASS' if ok else 'FAIL'} (max_err={max_err:.6f}, Q[{M},{K}] @ K^T[{K},{N}])")
    return ok


if __name__ == "__main__":
    print("=== Tile Operations Tests ===\n")

    tests = [
        ("2D Softmax", test_2d_softmax),
        ("Transpose + Dot", test_transpose_dot),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"[TEST] {name}")
        try:
            if fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
