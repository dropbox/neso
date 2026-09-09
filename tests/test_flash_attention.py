#!/usr/bin/env python3
"""Flash Attention 2 forward pass through the generic codegen pipeline.

This tests the full TTIR -> MSL -> Metal execution path for FA2,
using only the generic tile infrastructure (no handwritten MSL).

Algorithm (online softmax, Dao et al.):
  For each query block [BM, d]:
    Load Q once
    For each K/V block [BN, d]:
      QK = Q @ K^T                    [BM, BN]
      QK *= scale
      m_new = max(m, rowmax(QK))
      alpha = exp(m - m_new)
      P = exp(QK - m_new)
      l = l * alpha + rowsum(P)
      O = O * alpha + P @ V
      m = m_new
    O = O / l
"""
import sys, os, struct, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Metal
from neso.backend.codegen import ttir_to_msl
from neso.backend.codegen import ttir_to_msl_with_metadata

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


def generate_fa2_ttir(BM, BN, d, qkv_dtype='f32'):
    """Generate TTIR for Flash Attention 2 forward pass.

    Args: Q_ptr, K_ptr, V_ptr, O_ptr, N (seq length), stride_qo, stride_kv, scale

    Q, K, V are [N, d] row-major. O is [N, d] row-major.
    stride_qo = stride_kv = d (row stride).

    qkv_dtype: 'f32' for all-f32, 'f16' for mixed precision (f16 Q/K/V, f32 accum).
    """
    dt = qkv_dtype  # shorthand for Q/K/V data type
    mixed = (dt != 'f32')  # True if using mixed precision

    def load_tile_ttir(prefix, ptr_arg, stride_arg, row_offset, rows, cols,
                       range_col_name=None, dtype='f32'):
        rc = range_col_name or f"%range_d_{prefix}"
        lines = []
        lines.append(f"%range_r_{prefix} = tt.make_range {{start = 0 : i32, end = {rows} : i32}} : tensor<{rows}xi32>")
        lines.append(f"%off_r_splat_{prefix} = tt.splat {row_offset} : i32 -> tensor<{rows}xi32>")
        lines.append(f"%offs_r_{prefix} = arith.addi %off_r_splat_{prefix}, %range_r_{prefix} : tensor<{rows}xi32>")
        if range_col_name is None:
            lines.append(f"{rc} = tt.make_range {{start = 0 : i32, end = {cols} : i32}} : tensor<{cols}xi32>")
        lines.append(f"%row_exp_{prefix} = tt.expand_dims %offs_r_{prefix} {{axis = 1 : i32}} : tensor<{rows}xi32> -> tensor<{rows}x1xi32>")
        lines.append(f"%stride_splat_{prefix} = tt.splat {stride_arg} : i32 -> tensor<{rows}x1xi32>")
        lines.append(f"%row_off_{prefix} = arith.muli %row_exp_{prefix}, %stride_splat_{prefix} : tensor<{rows}x1xi32>")
        lines.append(f"%col_exp_{prefix} = tt.expand_dims {rc} {{axis = 0 : i32}} : tensor<{cols}xi32> -> tensor<1x{cols}xi32>")
        lines.append(f"%row_off_bc_{prefix} = tt.broadcast %row_off_{prefix} : tensor<{rows}x1xi32> -> tensor<{rows}x{cols}xi32>")
        lines.append(f"%col_bc_{prefix} = tt.broadcast %col_exp_{prefix} : tensor<1x{cols}xi32> -> tensor<{rows}x{cols}xi32>")
        lines.append(f"%idx_{prefix} = arith.addi %row_off_bc_{prefix}, %col_bc_{prefix} : tensor<{rows}x{cols}xi32>")
        lines.append(f"%base_{prefix} = tt.splat {ptr_arg} : !tt.ptr<{dtype}> -> tensor<{rows}x{cols}x!tt.ptr<{dtype}>>")
        lines.append(f"%ptrs_{prefix} = tt.addptr %base_{prefix}, %idx_{prefix} : tensor<{rows}x{cols}x!tt.ptr<{dtype}>>, tensor<{rows}x{cols}xi32>")
        lines.append(f"%tile_{prefix} = tt.load %ptrs_{prefix} : tensor<{rows}x{cols}x!tt.ptr<{dtype}>>")
        return '\n        '.join(lines), f"%tile_{prefix}"

    def store_tile_ttir(prefix, ptr_arg, stride_arg, row_offset, val_ssa, rows, cols,
                        range_col_name=None, dtype='f32'):
        rc = range_col_name or f"%range_d_st_{prefix}"
        lines = []
        lines.append(f"%range_r_st_{prefix} = tt.make_range {{start = 0 : i32, end = {rows} : i32}} : tensor<{rows}xi32>")
        lines.append(f"%off_r_st_splat_{prefix} = tt.splat {row_offset} : i32 -> tensor<{rows}xi32>")
        lines.append(f"%offs_r_st_{prefix} = arith.addi %off_r_st_splat_{prefix}, %range_r_st_{prefix} : tensor<{rows}xi32>")
        if range_col_name is None:
            lines.append(f"{rc} = tt.make_range {{start = 0 : i32, end = {cols} : i32}} : tensor<{cols}xi32>")
        lines.append(f"%row_exp_st_{prefix} = tt.expand_dims %offs_r_st_{prefix} {{axis = 1 : i32}} : tensor<{rows}xi32> -> tensor<{rows}x1xi32>")
        lines.append(f"%stride_splat_st_{prefix} = tt.splat {stride_arg} : i32 -> tensor<{rows}x1xi32>")
        lines.append(f"%row_off_st_{prefix} = arith.muli %row_exp_st_{prefix}, %stride_splat_st_{prefix} : tensor<{rows}x1xi32>")
        lines.append(f"%col_exp_st_{prefix} = tt.expand_dims {rc} {{axis = 0 : i32}} : tensor<{cols}xi32> -> tensor<1x{cols}xi32>")
        lines.append(f"%row_off_bc_st_{prefix} = tt.broadcast %row_off_st_{prefix} : tensor<{rows}x1xi32> -> tensor<{rows}x{cols}xi32>")
        lines.append(f"%col_bc_st_{prefix} = tt.broadcast %col_exp_st_{prefix} : tensor<1x{cols}xi32> -> tensor<{rows}x{cols}xi32>")
        lines.append(f"%idx_st_{prefix} = arith.addi %row_off_bc_st_{prefix}, %col_bc_st_{prefix} : tensor<{rows}x{cols}xi32>")
        lines.append(f"%base_st_{prefix} = tt.splat {ptr_arg} : !tt.ptr<{dtype}> -> tensor<{rows}x{cols}x!tt.ptr<{dtype}>>")
        lines.append(f"%ptrs_st_{prefix} = tt.addptr %base_st_{prefix}, %idx_st_{prefix} : tensor<{rows}x{cols}x!tt.ptr<{dtype}>>, tensor<{rows}x{cols}xi32>")
        lines.append(f"tt.store %ptrs_st_{prefix}, {val_ssa} : tensor<{rows}x{cols}x!tt.ptr<{dtype}>>")
        return '\n        '.join(lines)

    q_load, q_ssa = load_tile_ttir("q", "%Q_ptr", "%stride_qo", "%off_m", BM, d, dtype=dt)
    k_load, k_ssa = load_tile_ttir("k", "%K_ptr", "%stride_kv", "%iv", BN, d, dtype=dt)
    v_load, v_ssa = load_tile_ttir("v", "%V_ptr", "%stride_kv", "%iv", BN, d, dtype=dt)
    o_store = store_tile_ttir("o", "%O_ptr", "%stride_qo", "%off_m", "%O_final", BM, d, dtype='f32')

    # For mixed precision, P needs truncf to match Q/K/V dtype for the PV dot
    if mixed:
        p_for_dot = "%P_lo"
        truncf_line = f"%P_lo = arith.truncf %P : tensor<{BM}x{BN}xf32> to tensor<{BM}x{BN}x{dt}>"
    else:
        p_for_dot = "%P"
        truncf_line = ""

    ttir = f"""
    module {{
      tt.func public @flash_attention_fwd(
        %Q_ptr: !tt.ptr<{dt}>, %K_ptr: !tt.ptr<{dt}>, %V_ptr: !tt.ptr<{dt}>,
        %O_ptr: !tt.ptr<f32>,
        %N_param: i32, %stride_qo: i32, %stride_kv: i32, %scale_param: f32
      ) {{
        %c0 = arith.constant 0 : i32
        %cBN = arith.constant {BN} : i32
        %cBM = arith.constant {BM} : i32

        %pid = tt.get_program_id x : i32
        %off_m = arith.muli %pid, %cBM : i32

        // Load Q tile [BM, d] — persists across all iterations
        {q_load}

        // Initialize accumulators
        %m_init = arith.constant dense<0xFF800000> : tensor<{BM}xf32>
        %l_init = arith.constant dense<0.000000e+00> : tensor<{BM}xf32>
        %o_init = arith.constant dense<0.000000e+00> : tensor<{BM}x{d}xf32>

        // Main loop over K/V blocks
        %results:3 = scf.for %iv = %c0 to %N_param step %cBN
            iter_args(%m_i = %m_init, %l_i = %l_init, %acc_o = %o_init)
            -> (tensor<{BM}xf32>, tensor<{BM}xf32>, tensor<{BM}x{d}xf32>) : i32 {{

          // Load K[kv_start:kv_start+BN, 0:d]
          {k_load}

          // Transpose K: [BN, d] -> [d, BN]
          %KT = tt.trans {k_ssa} {{order = array<i32: 1, 0>}} : tensor<{BN}x{d}x{dt}> -> tensor<{d}x{BN}x{dt}>

          // QK = Q @ K^T: [BM, d] * [d, BN] -> [BM, BN] (mixed precision if f16)
          %zero_qk = arith.constant dense<0.000000e+00> : tensor<{BM}x{BN}xf32>
          %QK = tt.dot {q_ssa}, %KT, %zero_qk : tensor<{BM}x{d}x{dt}> * tensor<{d}x{BN}x{dt}> -> tensor<{BM}x{BN}xf32>

          // QK *= scale
          %scale_splat = tt.splat %scale_param : f32 -> tensor<{BM}x{BN}xf32>
          %QK_scaled = arith.mulf %QK, %scale_splat : tensor<{BM}x{BN}xf32>

          // Row-wise max: [BM, BN] -> [BM]
          %row_max = "tt.reduce"(%QK_scaled) <{{axis = 1 : i32}}> ({{
          ^bb0(%lhs: f32, %rhs: f32):
            %max = arith.maxnumf %lhs, %rhs : f32
            tt.reduce.return %max : f32
          }}) : (tensor<{BM}x{BN}xf32>) -> tensor<{BM}xf32>

          // m_new = max(m_i, row_max)
          %m_new = arith.maximumf %m_i, %row_max : tensor<{BM}xf32>

          // alpha = exp(m_i - m_new)
          %diff_m = arith.subf %m_i, %m_new : tensor<{BM}xf32>
          %alpha = math.exp %diff_m : tensor<{BM}xf32>

          // P = exp(QK_scaled - broadcast(m_new))
          %m_new_exp = tt.expand_dims %m_new {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
          %m_new_bc = tt.broadcast %m_new_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{BN}xf32>
          %qk_shifted = arith.subf %QK_scaled, %m_new_bc : tensor<{BM}x{BN}xf32>
          %P = math.exp %qk_shifted : tensor<{BM}x{BN}xf32>

          // Row-wise sum: [BM, BN] -> [BM]
          %row_sum = "tt.reduce"(%P) <{{axis = 1 : i32}}> ({{
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            tt.reduce.return %sum : f32
          }}) : (tensor<{BM}x{BN}xf32>) -> tensor<{BM}xf32>

          // l_new = l_i * alpha + row_sum
          %l_scaled = arith.mulf %l_i, %alpha : tensor<{BM}xf32>
          %l_new = arith.addf %l_scaled, %row_sum : tensor<{BM}xf32>

          // O = O * broadcast(alpha, [BM, d]) + P @ V
          %alpha_exp = tt.expand_dims %alpha {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
          %alpha_bc = tt.broadcast %alpha_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{d}xf32>
          %o_scaled = arith.mulf %acc_o, %alpha_bc : tensor<{BM}x{d}xf32>

          // Truncate P for PV dot (f16 path only)
          {truncf_line}

          // Load V[kv_start:kv_start+BN, 0:d]
          {v_load}

          // PV = P @ V: [BM, BN] * [BN, d] -> [BM, d]
          %o_new = tt.dot {p_for_dot}, {v_ssa}, %o_scaled : tensor<{BM}x{BN}x{dt}> * tensor<{BN}x{d}x{dt}> -> tensor<{BM}x{d}xf32>

          scf.yield %m_new, %l_new, %o_new : tensor<{BM}xf32>, tensor<{BM}xf32>, tensor<{BM}x{d}xf32>
        }}

        // Final normalization: O = O / broadcast(l)
        %l_exp = tt.expand_dims %results#1 {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
        %l_bc = tt.broadcast %l_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{d}xf32>
        %O_final = arith.divf %results#2, %l_bc : tensor<{BM}x{d}xf32>

        // Store O
        {o_store}
        tt.return
      }}
    }}
    """
    return ttir


def test_fa2(N, d, BM=8, BN=32, qkv_dtype='f32'):
    """Test FA2 forward pass through codegen pipeline."""
    ttir = generate_fa2_ttir(BM, BN, d, qkv_dtype=qkv_dtype)

    try:
        msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    except Exception as e:
        print(f"  Codegen failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        pipe = compile_kernel(msl, name)
    except Exception as e:
        print(f"  Metal compile failed: {e}")
        return False

    import random
    random.seed(42)
    scale = 1.0 / math.sqrt(d)

    # Generate data in f64 for reference, pack as appropriate dtype
    q_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    k_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    v_data = [random.gauss(0, 0.5) for _ in range(N * d)]

    pack_fmt = 'e' if qkv_dtype == 'f16' else 'f'
    q_buf = make_buffer(q_data, dtype=pack_fmt)
    k_buf = make_buffer(k_data, dtype=pack_fmt)
    v_buf = make_buffer(v_data, dtype=pack_fmt)
    o_buf = make_zero_buffer(N * d)  # output always f32
    n_buf = scalar_buf(N)
    stride_qo_buf = scalar_buf(d)
    stride_kv_buf = scalar_buf(d)
    scale_buf = float_buf(scale)

    bufs = [q_buf, k_buf, v_buf, o_buf, n_buf, stride_qo_buf, stride_kv_buf, scale_buf]

    threads = rec_threads
    grid = (N // BM, 1, 1)

    dispatch(pipe, grid, threads, bufs)

    o_vals = read_buffer(o_buf, N * d)

    # Reference: softmax(Q[i,:] @ K^T * scale) @ V for first row
    q_row = q_data[0:d]
    scores = []
    for j in range(N):
        k_row = k_data[j * d:(j + 1) * d]
        s = sum(q_row[kk] * k_row[kk] for kk in range(d)) * scale
        scores.append(s)
    max_s = max(scores)
    exp_scores = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_scores)
    attn_weights = [e / sum_exp for e in exp_scores]

    ref_o = [0.0] * d
    for j in range(N):
        v_row = v_data[j * d:(j + 1) * d]
        for kk in range(d):
            ref_o[kk] += attn_weights[j] * v_row[kk]

    max_err = max(abs(o_vals[kk] - ref_o[kk]) for kk in range(d))
    # Scalar fallback (non-simdgroup) has higher FP accumulation error than HW MMA
    use_simdgroup = os.environ.get("NESO_SIMDGROUP", "").lower() not in ("0", "false", "no")
    tol = 0.05 if (qkv_dtype == 'f16' or not use_simdgroup) else 0.01
    ok = max_err < tol

    # Also check middle row
    mid = N // 2
    q_row2 = q_data[mid * d:(mid + 1) * d]
    scores2 = []
    for j in range(N):
        k_row = k_data[j * d:(j + 1) * d]
        s = sum(q_row2[kk] * k_row[kk] for kk in range(d)) * scale
        scores2.append(s)
    max_s2 = max(scores2)
    exp_scores2 = [math.exp(s - max_s2) for s in scores2]
    sum_exp2 = sum(exp_scores2)
    attn_weights2 = [e / sum_exp2 for e in exp_scores2]
    ref_o2 = [0.0] * d
    for j in range(N):
        v_row = v_data[j * d:(j + 1) * d]
        for kk in range(d):
            ref_o2[kk] += attn_weights2[j] * v_row[kk]
    err_mid = max(abs(o_vals[mid * d + kk] - ref_o2[kk]) for kk in range(d))
    max_err = max(max_err, err_mid)
    ok = ok and err_mid < tol

    return ok, max_err


def bench_fa2(N, d, BM=8, BN=32, iters=20, qkv_dtype='f32'):
    """Benchmark FA2 through codegen pipeline."""
    import time
    ttir = generate_fa2_ttir(BM, BN, d, qkv_dtype=qkv_dtype)
    msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    pipe = compile_kernel(msl, name)

    scale = 1.0 / math.sqrt(d)
    import random
    random.seed(42)
    q_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    k_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    v_data = [random.gauss(0, 0.5) for _ in range(N * d)]

    pack_fmt = 'e' if qkv_dtype == 'f16' else 'f'
    q_buf = make_buffer(q_data, dtype=pack_fmt)
    k_buf = make_buffer(k_data, dtype=pack_fmt)
    v_buf = make_buffer(v_data, dtype=pack_fmt)
    n_buf = scalar_buf(N)
    stride_qo_buf = scalar_buf(d)
    stride_kv_buf = scalar_buf(d)
    scale_buf = float_buf(scale)

    threads = rec_threads
    grid = (N // BM, 1, 1)

    # Warmup
    for _ in range(3):
        o_buf = make_zero_buffer(N * d)
        bufs = [q_buf, k_buf, v_buf, o_buf, n_buf, stride_qo_buf, stride_kv_buf, scale_buf]
        dispatch(pipe, grid, threads, bufs)

    # Timed
    gpu_times = []
    for _ in range(iters):
        o_buf = make_zero_buffer(N * d)
        bufs = [q_buf, k_buf, v_buf, o_buf, n_buf, stride_qo_buf, stride_kv_buf, scale_buf]
        cb = dispatch(pipe, grid, threads, bufs)
        gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
        gpu_times.append(gpu_ms)

    avg = sum(gpu_times) / len(gpu_times)
    mn = min(gpu_times)
    flops = 4.0 * N * N * d  # 2 matmuls: QK^T and PV
    gflops = flops / (mn / 1000.0) / 1e9
    return avg, mn, gflops


if __name__ == "__main__":
    print(f"Metal Device: {device.name()}")
    print()

    # === Correctness tests ===
    print("=== Flash Attention 2 Correctness (codegen pipeline) ===")
    configs = [
        (64, 32, 8, 32),    # N=64, d=32, BM=8, BN=32
        (128, 64, 8, 32),   # N=128, d=64
        (256, 64, 8, 32),   # N=256, d=64
        (128, 64, 16, 32),  # N=128, d=64, BM=16
        (256, 64, 16, 32),  # N=256, d=64, BM=16
        (128, 64, 32, 32),  # N=128, d=64, BM=32
        (256, 64, 32, 32),  # N=256, d=64, BM=32
        # d=128 configs
        (128, 128, 8, 32),  # N=128, d=128, BM=8
        (256, 128, 8, 32),  # N=256, d=128, BM=8
        (512, 128, 8, 32),  # N=512, d=128, BM=8
        (128, 128, 16, 16), # N=128, d=128, BM=16, BN=16
        (256, 128, 16, 16), # N=256, d=128, BM=16
        (512, 128, 16, 16), # N=512, d=128, BM=16
        # d=128 BM=16 BN=32 (enabled by register accumulator saving 8KB)
        (128, 128, 16, 32), # N=128, d=128, BM=16, BN=32
        (256, 128, 16, 32), # N=256, d=128, BM=16, BN=32
        (512, 128, 16, 32), # N=512, d=128, BM=16, BN=32
    ]

    all_ok = True
    for N, d, BM, BN in configs:
        result = test_fa2(N, d, BM, BN)
        if result is False:
            print(f"  N={N}, d={d}, BM={BM}, BN={BN}: FAIL (codegen/compile error)")
            all_ok = False
        else:
            ok, err = result
            status = "PASS" if ok else "FAIL"
            print(f"  N={N}, d={d}, BM={BM}, BN={BN}: {status} (max_err={err:.6f})")
            if not ok:
                all_ok = False

    # === f16 mixed precision correctness tests ===
    print()
    print("=== Flash Attention 2 Correctness (f16 mixed precision) ===")
    f16_configs = [
        (64, 32, 8, 32),
        (128, 64, 8, 32),
        (256, 64, 16, 32),
        (256, 64, 32, 32),
        # d=128
        (128, 128, 8, 32),
        (256, 128, 16, 32),
        (512, 128, 16, 32),
        # d=128 BM=32 (f16 enables this: Q+K+V fit in ~24KB)
        (128, 128, 32, 32),
        (256, 128, 32, 32),
        (512, 128, 32, 32),
    ]
    for N, d, BM, BN in f16_configs:
        result = test_fa2(N, d, BM, BN, qkv_dtype='f16')
        if result is False:
            print(f"  N={N}, d={d}, BM={BM}, BN={BN} f16: FAIL (codegen/compile error)")
            all_ok = False
        else:
            ok, err = result
            status = "PASS" if ok else "FAIL"
            print(f"  N={N}, d={d}, BM={BM}, BN={BN} f16: {status} (max_err={err:.6f})")
            if not ok:
                all_ok = False

    if not all_ok:
        print("\nSome tests failed!")
        sys.exit(1)

    if "test" in sys.argv:
        print("\nAll tests passed!")
        sys.exit(0)

    # === Performance benchmark ===
    print()
    print("=== Flash Attention 2 Performance (codegen pipeline) ===")
    print(f"{'N':>6s}  {'d':>4s}  {'Tile':>7s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}")
    print("-" * 55)

    bench_configs = [
        (256, 64, 8, 32),
        (512, 64, 8, 32),
        (1024, 64, 8, 32),
        (2048, 64, 8, 32),
        (4096, 64, 8, 32),
        (256, 64, 16, 32),
        (512, 64, 16, 32),
        (1024, 64, 16, 32),
        (2048, 64, 16, 32),
        (4096, 64, 16, 32),
        (256, 64, 32, 32),
        (512, 64, 32, 32),
        (1024, 64, 32, 32),
        (2048, 64, 32, 32),
        (4096, 64, 32, 32),
        # d=128
        (256, 128, 8, 32),
        (512, 128, 8, 32),
        (1024, 128, 8, 32),
        (2048, 128, 8, 32),
        (4096, 128, 8, 32),
        (256, 128, 16, 16),
        (512, 128, 16, 16),
        (1024, 128, 16, 16),
        (2048, 128, 16, 16),
        (4096, 128, 16, 16),
        # d=128 BM=16 BN=32 (register acc enables this)
        (256, 128, 16, 32),
        (512, 128, 16, 32),
        (1024, 128, 16, 32),
        (2048, 128, 16, 32),
        (4096, 128, 16, 32),
        # d=128 BM=24 BN=32 (single-block diag scratch + bps=3 reg acc)
        (192, 128, 24, 32),
        (384, 128, 24, 32),
        (768, 128, 24, 32),
        (1536, 128, 24, 32),
        (3072, 128, 24, 32),
        (4080, 128, 24, 32),
    ]

    for N, d, BM, BN in bench_configs:
        tile_str = f"{BM}x{BN}"
        avg, mn, gf = bench_fa2(N, d, BM, BN)
        print(f"  {N:>4d}  {d:>4d}  {tile_str:>7s}  {avg:>8.3f}  {mn:>8.3f}  {gf:>9.1f}")

    # === f16 Performance benchmark ===
    print()
    print("=== Flash Attention 2 Performance (f16 mixed precision) ===")
    print(f"{'N':>6s}  {'d':>4s}  {'Tile':>7s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}")
    print("-" * 55)

    f16_bench_configs = [
        # d=64 f16
        (256, 64, 16, 32),
        (512, 64, 16, 32),
        (1024, 64, 16, 32),
        (2048, 64, 16, 32),
        (4096, 64, 16, 32),
        (256, 64, 32, 32),
        (512, 64, 32, 32),
        (1024, 64, 32, 32),
        (2048, 64, 32, 32),
        (4096, 64, 32, 32),
        # d=128 f16
        (256, 128, 16, 32),
        (512, 128, 16, 32),
        (1024, 128, 16, 32),
        (2048, 128, 16, 32),
        (4096, 128, 16, 32),
        (256, 128, 24, 32),
        (512, 128, 24, 32),
        (1024, 128, 24, 32),
        (2048, 128, 24, 32),
        (4096, 128, 24, 32),
        # d=128 BM=32 (f16 enables this)
        (256, 128, 32, 32),
        (512, 128, 32, 32),
        (1024, 128, 32, 32),
        (2048, 128, 32, 32),
        (4096, 128, 32, 32),
    ]

    for N, d, BM, BN in f16_bench_configs:
        tile_str = f"{BM}x{BN}"
        avg, mn, gf = bench_fa2(N, d, BM, BN, qkv_dtype='f16')
        print(f"  {N:>4d}  {d:>4d}  {tile_str:>7s}  {avg:>8.3f}  {mn:>8.3f}  {gf:>9.1f}")

    print()
