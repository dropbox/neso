#!/usr/bin/env python3
"""Flash Attention 2 with relative position bias (multi-head).

Fuses: scores = Q_u @ K^T * scale + rel_bias, softmax, attn @ V
into a single kernel. Multi-head via program_id.y.

Kernel arguments:
  Q: [nh, T, d] F16 (Q + bias_u already added)
  K: [nh, T, d] F16
  V: [nh, T, d] F16
  bias: [nh, T, T] F32 (precomputed relative position bias, already scaled)
  O: [nh, T, d] F16 output
  T: i32 sequence length
  stride: i32 = d (row stride within head)
  scale: f32 = 1/sqrt(d)
  stride_h: i32 = T*d (stride between heads for Q/K/V/O)
  bias_stride_h: i32 = T*T (stride between heads for bias)

Grid: (cdiv(T, BM), nh, 1)
"""
import sys, os, struct, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Metal
from neso.backend.codegen import ttir_to_msl_with_metadata

device = Metal.MTLCreateSystemDefaultDevice()
queue = device.newCommandQueue()


def make_f16_buffer(data):
    raw = struct.pack(f'{len(data)}e', *data)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)

def make_zero_f16_buffer(n):
    return device.newBufferWithLength_options_(n * 2, Metal.MTLResourceStorageModeShared)

def make_f32_buffer(data):
    raw = struct.pack(f'{len(data)}f', *data)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)

def scalar_buf(val, dtype='i'):
    raw = struct.pack(dtype, val)
    return device.newBufferWithBytes_length_options_(raw, len(raw), Metal.MTLResourceStorageModeShared)

def float_buf(val):
    return scalar_buf(val, 'f')

def read_f16_buffer(buf, n):
    raw = buf.contents().as_buffer(n * 2)
    return list(struct.unpack(f'{n}e', raw))

def compile_kernel(msl_source, name):
    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(True)
    options.setLanguageVersion_(Metal.MTLLanguageVersion3_1)
    lib, err = device.newLibraryWithSource_options_error_(msl_source, options, None)
    if err:
        raise RuntimeError(f"MSL compile error:\n{err.localizedDescription()}\n\nSource:\n{msl_source[:2000]}")
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


def generate_rel_pos_fa2_ttir(BM, BN, d):
    """Generate multi-head FA2 with additive bias.

    Uses program_id.y as head index. All pointers offset by head_idx * stride_h.
    """
    dt = 'f16'

    def load_tile(prefix, ptr_arg, stride_arg, row_offset, rows, cols,
                  range_col_name=None, dtype='f16'):
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

    def store_tile(prefix, ptr_arg, stride_arg, row_offset, val_ssa, rows, cols,
                   range_col_name=None, dtype='f16'):
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

    # Loads use head-offset pointers (Q_h, K_h, V_h, bias_h, O_h)
    q_load, q_ssa = load_tile("q", "%Q_h", "%stride", "%off_m", BM, d, dtype=dt)
    k_load, k_ssa = load_tile("k", "%K_h", "%stride", "%iv", BN, d, dtype=dt)
    v_load, v_ssa = load_tile("v", "%V_h", "%stride", "%iv", BN, d, dtype=dt)
    o_store = store_tile("o", "%O_h", "%stride", "%off_m", "%O_f16", BM, d, dtype=dt)

    # Bias tile load: [BM, BN] from bias_h (F16) with row stride = T, upcast to F32
    bias_load = f"""
          %range_r_bias = tt.make_range {{start = 0 : i32, end = {BM} : i32}} : tensor<{BM}xi32>
          %off_r_bias_splat = tt.splat %off_m : i32 -> tensor<{BM}xi32>
          %offs_r_bias = arith.addi %off_r_bias_splat, %range_r_bias : tensor<{BM}xi32>
          %range_c_bias = tt.make_range {{start = 0 : i32, end = {BN} : i32}} : tensor<{BN}xi32>
          %off_c_bias_splat = tt.splat %iv : i32 -> tensor<{BN}xi32>
          %offs_c_bias = arith.addi %off_c_bias_splat, %range_c_bias : tensor<{BN}xi32>
          %row_exp_bias = tt.expand_dims %offs_r_bias {{axis = 1 : i32}} : tensor<{BM}xi32> -> tensor<{BM}x1xi32>
          %stride_bias_splat = tt.splat %T_param : i32 -> tensor<{BM}x1xi32>
          %row_off_bias = arith.muli %row_exp_bias, %stride_bias_splat : tensor<{BM}x1xi32>
          %col_exp_bias = tt.expand_dims %offs_c_bias {{axis = 0 : i32}} : tensor<{BN}xi32> -> tensor<1x{BN}xi32>
          %row_off_bc_bias = tt.broadcast %row_off_bias : tensor<{BM}x1xi32> -> tensor<{BM}x{BN}xi32>
          %col_bc_bias = tt.broadcast %col_exp_bias : tensor<1x{BN}xi32> -> tensor<{BM}x{BN}xi32>
          %idx_bias = arith.addi %row_off_bc_bias, %col_bc_bias : tensor<{BM}x{BN}xi32>
          %base_bias = tt.splat %bias_h : !tt.ptr<{dt}> -> tensor<{BM}x{BN}x!tt.ptr<{dt}>>
          %ptrs_bias = tt.addptr %base_bias, %idx_bias : tensor<{BM}x{BN}x!tt.ptr<{dt}>>, tensor<{BM}x{BN}xi32>
          %bias_tile_f16 = tt.load %ptrs_bias : tensor<{BM}x{BN}x!tt.ptr<{dt}>>
          %bias_tile = arith.extf %bias_tile_f16 : tensor<{BM}x{BN}x{dt}> to tensor<{BM}x{BN}xf32>"""

    ttir = f"""
    module {{
      tt.func public @rel_pos_fa2_fwd(
        %Q_ptr: !tt.ptr<{dt}>, %K_ptr: !tt.ptr<{dt}>, %V_ptr: !tt.ptr<{dt}>,
        %bias_ptr: !tt.ptr<{dt}>,
        %O_ptr: !tt.ptr<{dt}>,
        %T_param: i32, %stride: i32, %scale_param: f32,
        %stride_h: i32, %bias_stride_h: i32
      ) {{
        %c0 = arith.constant 0 : i32
        %cBN = arith.constant {BN} : i32
        %cBM = arith.constant {BM} : i32

        // Block indices
        %pid_m = tt.get_program_id x : i32
        %pid_h = tt.get_program_id y : i32
        %off_m = arith.muli %pid_m, %cBM : i32

        // Offset pointers by head index
        %h_off = arith.muli %pid_h, %stride_h : i32
        %Q_h = tt.addptr %Q_ptr, %h_off : !tt.ptr<{dt}>, i32
        %K_h = tt.addptr %K_ptr, %h_off : !tt.ptr<{dt}>, i32
        %V_h = tt.addptr %V_ptr, %h_off : !tt.ptr<{dt}>, i32
        %O_h = tt.addptr %O_ptr, %h_off : !tt.ptr<{dt}>, i32
        %bh_off = arith.muli %pid_h, %bias_stride_h : i32
        %bias_h = tt.addptr %bias_ptr, %bh_off : !tt.ptr<{dt}>, i32

        // Load Q tile [BM, d]
        {q_load}

        // Initialize accumulators
        %m_init = arith.constant dense<0xFF800000> : tensor<{BM}xf32>
        %l_init = arith.constant dense<0.000000e+00> : tensor<{BM}xf32>
        %o_init = arith.constant dense<0.000000e+00> : tensor<{BM}x{d}xf32>

        // Main loop over K/V blocks
        %results:3 = scf.for %iv = %c0 to %T_param step %cBN
            iter_args(%m_i = %m_init, %l_i = %l_init, %acc_o = %o_init)
            -> (tensor<{BM}xf32>, tensor<{BM}xf32>, tensor<{BM}x{d}xf32>) : i32 {{

          {k_load}
          %KT = tt.trans {k_ssa} {{order = array<i32: 1, 0>}} : tensor<{BN}x{d}x{dt}> -> tensor<{d}x{BN}x{dt}>

          // Content scores: Q @ K^T
          %zero_qk = arith.constant dense<0.000000e+00> : tensor<{BM}x{BN}xf32>
          %QK = tt.dot {q_ssa}, %KT, %zero_qk : tensor<{BM}x{d}x{dt}> * tensor<{d}x{BN}x{dt}> -> tensor<{BM}x{BN}xf32>

          // Add unscaled relative position bias, then scale both together
          {bias_load}
          %QK_plus_bias = arith.addf %QK, %bias_tile : tensor<{BM}x{BN}xf32>
          %scale_splat = tt.splat %scale_param : f32 -> tensor<{BM}x{BN}xf32>
          %scores = arith.mulf %QK_plus_bias, %scale_splat : tensor<{BM}x{BN}xf32>

          // Online softmax
          %row_max = "tt.reduce"(%scores) <{{axis = 1 : i32}}> ({{
          ^bb0(%lhs: f32, %rhs: f32):
            %max = arith.maxnumf %lhs, %rhs : f32
            tt.reduce.return %max : f32
          }}) : (tensor<{BM}x{BN}xf32>) -> tensor<{BM}xf32>
          %m_new = arith.maximumf %m_i, %row_max : tensor<{BM}xf32>
          %diff_m = arith.subf %m_i, %m_new : tensor<{BM}xf32>
          %alpha = math.exp %diff_m : tensor<{BM}xf32>
          %m_new_exp = tt.expand_dims %m_new {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
          %m_new_bc = tt.broadcast %m_new_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{BN}xf32>
          %qk_shifted = arith.subf %scores, %m_new_bc : tensor<{BM}x{BN}xf32>
          %P = math.exp %qk_shifted : tensor<{BM}x{BN}xf32>
          %row_sum = "tt.reduce"(%P) <{{axis = 1 : i32}}> ({{
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            tt.reduce.return %sum : f32
          }}) : (tensor<{BM}x{BN}xf32>) -> tensor<{BM}xf32>
          %l_scaled = arith.mulf %l_i, %alpha : tensor<{BM}xf32>
          %l_new = arith.addf %l_scaled, %row_sum : tensor<{BM}xf32>

          // Rescale O
          %alpha_exp = tt.expand_dims %alpha {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
          %alpha_bc = tt.broadcast %alpha_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{d}xf32>
          %o_scaled = arith.mulf %acc_o, %alpha_bc : tensor<{BM}x{d}xf32>

          // P @ V (truncate P to f16)
          %P_lo = arith.truncf %P : tensor<{BM}x{BN}xf32> to tensor<{BM}x{BN}x{dt}>
          {v_load}
          %o_new = tt.dot %P_lo, {v_ssa}, %o_scaled : tensor<{BM}x{BN}x{dt}> * tensor<{BN}x{d}x{dt}> -> tensor<{BM}x{d}xf32>

          scf.yield %m_new, %l_new, %o_new : tensor<{BM}xf32>, tensor<{BM}xf32>, tensor<{BM}x{d}xf32>
        }}

        // Final: O = O / l, convert to f16
        %l_exp = tt.expand_dims %results#1 {{axis = 1 : i32}} : tensor<{BM}xf32> -> tensor<{BM}x1xf32>
        %l_bc = tt.broadcast %l_exp : tensor<{BM}x1xf32> -> tensor<{BM}x{d}xf32>
        %O_f32 = arith.divf %results#2, %l_bc : tensor<{BM}x{d}xf32>
        %O_f16 = arith.truncf %O_f32 : tensor<{BM}x{d}xf32> to tensor<{BM}x{d}x{dt}>

        {o_store}
        tt.return
      }}
    }}
    """
    return ttir


def reference_rel_pos_attention(Q, K, V, K_rel, bias_u, bias_v, T, d, nh, scale):
    """Multi-head reference implementation."""
    import numpy as np
    hd = d
    Q = np.array(Q).reshape(nh, T, hd)
    K = np.array(K).reshape(nh, T, hd)
    V = np.array(V).reshape(nh, T, hd)
    K_rel = np.array(K_rel).reshape(nh, 2*T-1, hd)
    bias_u = np.array(bias_u).reshape(nh, hd)
    bias_v = np.array(bias_v).reshape(nh, hd)

    out = np.zeros((nh, T, hd))
    for h in range(nh):
        Q_u = Q[h] + bias_u[h]
        Q_v = Q[h] + bias_v[h]
        content = Q_u @ K[h].T
        rel = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                idx = T - 1 - i + j
                if 0 <= idx < 2*T-1:
                    rel[i, j] = Q_v[i] @ K_rel[h, idx]
        scores = (content + rel) * scale
        mx = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - mx)
        w = e / e.sum(axis=-1, keepdims=True)
        out[h] = w @ V[h]
    return out


def precompute_bias_and_qu(Q, K_rel, bias_u, bias_v, T, d, nh, scale):
    """Precompute Q_u and unscaled rel_bias for the fused kernel.

    Returns bias WITHOUT scale applied — kernel applies (QK + bias) * scale.
    """
    import numpy as np
    hd = d
    Q = np.array(Q, dtype=np.float32).reshape(nh, T, hd)
    K_rel = np.array(K_rel, dtype=np.float32).reshape(nh, 2*T-1, hd)
    bias_u = np.array(bias_u, dtype=np.float32).reshape(nh, hd)
    bias_v = np.array(bias_v, dtype=np.float32).reshape(nh, hd)

    Q_u = Q + bias_u[:, None, :]  # [nh, T, hd]

    rel_bias = np.zeros((nh, T, T), dtype=np.float32)
    for h in range(nh):
        Q_v = Q[h] + bias_v[h]
        for i in range(T):
            for j in range(T):
                idx = T - 1 - i + j
                if 0 <= idx < 2*T-1:
                    rel_bias[h, i, j] = float(np.dot(
                        Q_v[i].astype(np.float64),
                        K_rel[h, idx].astype(np.float64)
                    ))
    return Q_u, rel_bias


def test_rel_pos_fa2(T, d, nh, BM=16, BN=32):
    """Test multi-head fused FA2 with precomputed relative position bias."""
    import numpy as np
    np.random.seed(42)

    hd = d
    scale = 1.0 / math.sqrt(hd)

    Q = np.random.randn(nh, T, hd).astype(np.float32) * 0.5
    K = np.random.randn(nh, T, hd).astype(np.float32) * 0.5
    V = np.random.randn(nh, T, hd).astype(np.float32) * 0.5
    K_rel = np.random.randn(nh, 2*T-1, hd).astype(np.float32) * 0.5
    bias_u = np.random.randn(nh, hd).astype(np.float32) * 0.1
    bias_v = np.random.randn(nh, hd).astype(np.float32) * 0.1

    ref_out = reference_rel_pos_attention(
        Q.flatten().tolist(), K.flatten().tolist(), V.flatten().tolist(),
        K_rel.flatten().tolist(), bias_u.flatten().tolist(), bias_v.flatten().tolist(),
        T, hd, nh, scale
    )

    Q_u, rel_bias = precompute_bias_and_qu(
        Q.flatten().tolist(), K_rel.flatten().tolist(),
        bias_u.flatten().tolist(), bias_v.flatten().tolist(),
        T, hd, nh, scale
    )

    ttir = generate_rel_pos_fa2_ttir(BM, BN, hd)
    try:
        msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    except Exception as e:
        print(f"  Codegen failed: {e}")
        import traceback; traceback.print_exc()
        return False, 0.0

    try:
        pipe = compile_kernel(msl, name)
    except Exception as e:
        print(f"  Metal compile failed: {e}")
        return False, 0.0

    q_buf = make_f16_buffer(Q_u.astype(np.float16).flatten().tolist())
    k_buf = make_f16_buffer(K.astype(np.float16).flatten().tolist())
    v_buf = make_f16_buffer(V.astype(np.float16).flatten().tolist())
    bias_buf = make_f16_buffer(rel_bias.astype(np.float16).flatten().tolist())
    o_buf = make_zero_f16_buffer(nh * T * hd)

    stride_h = T * hd
    bias_stride_h = T * T

    bufs = [q_buf, k_buf, v_buf, bias_buf, o_buf,
            scalar_buf(T), scalar_buf(hd), float_buf(scale),
            scalar_buf(stride_h), scalar_buf(bias_stride_h)]

    grid = (T // BM, nh, 1)
    dispatch(pipe, grid, rec_threads, bufs)

    out_vals = np.array(read_f16_buffer(o_buf, nh * T * hd)).reshape(nh, T, hd)
    max_err = np.max(np.abs(out_vals - ref_out))
    ok = max_err < 0.1
    return ok, max_err


def bench_rel_pos_fa2(T, d, nh, BM=16, BN=32, iters=20):
    """Benchmark multi-head FA2."""
    import numpy as np
    np.random.seed(42)

    hd = d
    scale = 1.0 / math.sqrt(hd)

    Q_u = np.random.randn(nh, T, hd).astype(np.float16)
    K = np.random.randn(nh, T, hd).astype(np.float16)
    V = np.random.randn(nh, T, hd).astype(np.float16)
    rel_bias = np.random.randn(nh, T, T).astype(np.float16) * 0.1

    ttir = generate_rel_pos_fa2_ttir(BM, BN, hd)
    msl, name, _, rec_threads = ttir_to_msl_with_metadata(ttir, block_size=BM * BN)
    pipe = compile_kernel(msl, name)

    q_buf = make_f16_buffer(Q_u.flatten().tolist())
    k_buf = make_f16_buffer(K.flatten().tolist())
    v_buf = make_f16_buffer(V.flatten().tolist())
    bias_buf = make_f16_buffer(rel_bias.flatten().tolist())

    stride_h = T * hd
    bias_stride_h = T * T
    grid = (T // BM, nh, 1)

    # Warmup
    for _ in range(3):
        o_buf = make_zero_f16_buffer(nh * T * hd)
        bufs = [q_buf, k_buf, v_buf, bias_buf, o_buf,
                scalar_buf(T), scalar_buf(hd), float_buf(scale),
                scalar_buf(stride_h), scalar_buf(bias_stride_h)]
        dispatch(pipe, grid, rec_threads, bufs)

    gpu_times = []
    for _ in range(iters):
        o_buf = make_zero_f16_buffer(nh * T * hd)
        bufs = [q_buf, k_buf, v_buf, bias_buf, o_buf,
                scalar_buf(T), scalar_buf(hd), float_buf(scale),
                scalar_buf(stride_h), scalar_buf(bias_stride_h)]
        cb = dispatch(pipe, grid, rec_threads, bufs)
        gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
        gpu_times.append(gpu_ms)

    avg = sum(gpu_times) / len(gpu_times)
    mn = min(gpu_times)
    # FLOPs per head: QK^T = 2*T*T*d, PV = 2*T*T*d, bias load = T*T
    flops = nh * (4.0 * T * T * hd + T * T)
    gflops = flops / (mn / 1000.0) / 1e9
    return avg, mn, gflops


if __name__ == "__main__":
    print(f"Metal Device: {device.name()}")
    print()

    print("=== Multi-Head Relative Position FA2 Correctness ===")
    configs = [
        (32, 64, 2, 16, 32),    # T=32, d=64, nh=2
        (64, 64, 4, 16, 32),    # T=64, d=64, nh=4
        (64, 128, 4, 16, 32),   # T=64, d=128, nh=4
        (128, 128, 8, 16, 32),  # T=128, d=128, nh=8
        (256, 128, 8, 16, 32),  # Larger T
        (448, 128, 8, 16, 32),  # Parakeet-like (padded to 448=16*28)
    ]

    all_ok = True
    for T, d, nh, BM, BN in configs:
        ok, err = test_rel_pos_fa2(T, d, nh, BM, BN)
        status = "PASS" if ok else "FAIL"
        print(f"  T={T}, d={d}, nh={nh}, BM={BM}: {status} (max_err={err:.6f})")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nSome tests failed!")
        sys.exit(1)

    if "test" in sys.argv:
        print("\nAll tests passed!")
        sys.exit(0)

    print()
    print("=== Multi-Head FA2 Performance (Parakeet: d=128, nh=8) ===")
    print(f"{'T':>6s}  {'nh':>3s}  {'Tile':>7s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}")
    print("-" * 55)

    bench_configs = [
        (128, 128, 8, 16, 32),
        (256, 128, 8, 16, 32),
        (384, 128, 8, 16, 32),
        (448, 128, 8, 16, 32),  # Parakeet 35s audio
        (512, 128, 8, 16, 32),
    ]
    for T, d, nh, BM, BN in bench_configs:
        try:
            avg, mn, gf = bench_rel_pos_fa2(T, d, nh, BM, BN)
            print(f"  {T:>4d}  {nh:>3d}  {BM}x{BN}  {avg:>8.3f}  {mn:>8.3f}  {gf:>9.1f}")
        except Exception as e:
            print(f"  {T:>4d}  {nh:>3d}  {BM}x{BN}  ERROR: {e}")

    print()
