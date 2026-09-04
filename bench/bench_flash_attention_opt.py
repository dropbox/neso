#!/usr/bin/env python3
"""Optimized Flash Attention 2 forward pass on Metal.

Key optimizations over bench_flash_attention.py:
1. Fewer barriers: combine operations, remove redundant syncs
2. vec4 loads for fp16: halves global memory transactions
3. Better O rescaling: avoid full sO store/reload cycle where possible
4. Larger BM for d=128 fp16 by tighter memory layout
"""
import sys, os, struct, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import Metal

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


def generate_fa2_opt_msl(BM, BN, d, THREADS=1024, use_half=False):
    """Optimized FA2 with fewer barriers and vec4 loads."""
    assert BN == 32, "BN must be 32 (SIMD width)"
    assert BM % 8 == 0 and d % 8 == 0

    io_type = "half" if use_half else "float"
    smem_type = "half" if use_half else "float"
    qk_mat_type = "half" if use_half else "float"
    pv_load_type = "half" if use_half else "float"

    NUM_SG = THREADS // 32
    QK_BLOCKS_M = BM // 8
    QK_BLOCKS_N = BN // 8
    QK_BLOCKS = QK_BLOCKS_M * QK_BLOCKS_N
    QK_BPS = max(1, (QK_BLOCKS + NUM_SG - 1) // NUM_SG)

    PV_BLOCKS_M = BM // 8
    PV_BLOCKS_N = d // 8
    PV_BLOCKS = PV_BLOCKS_M * PV_BLOCKS_N
    PV_BPS = max(1, (PV_BLOCKS + NUM_SG - 1) // NUM_SG)

    Q_SIZE = BM * d
    K_SIZE = BN * d
    QK_SIZE = BM * BN
    O_SIZE = BM * d

    Q_LOADS = max(1, (Q_SIZE + THREADS - 1) // THREADS)
    K_LOADS = max(1, (K_SIZE + THREADS - 1) // THREADS)
    O_STORES = max(1, (O_SIZE + THREADS - 1) // THREADS)

    smem_elem = 2 if use_half else 4
    shmem_bytes = Q_SIZE * smem_elem + K_SIZE * smem_elem + QK_SIZE * 4 + O_SIZE * 4 + BM * 4

    # vec4 load support for half
    use_vec4 = use_half and d % 4 == 0

    # Generate vec4 K/V load
    if use_vec4:
        vec_K_SIZE = K_SIZE // 4
        vec_cols = d // 4
        vec_K_LOADS = max(1, (vec_K_SIZE + THREADS - 1) // THREADS)
        kv_load = f"""        for (uint _i = 0; _i < {vec_K_LOADS}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {vec_K_SIZE}u) {{
                uint row = idx / {vec_cols}u, vcol = idx % {vec_cols}u;
                *((threadgroup half4*)&sKV[row * {d}u + vcol * 4u]) =
                    *((device const half4*)&KV_SRC[(kv_start + (int)row) * d_head + (int)vcol * 4]);
            }}
        }}"""
        vec_Q_SIZE = Q_SIZE // 4
        q_vec_cols = d // 4
        vec_Q_LOADS = max(1, (vec_Q_SIZE + THREADS - 1) // THREADS)
        q_load = f"""    for (uint _i = 0; _i < {vec_Q_LOADS}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {vec_Q_SIZE}u) {{
            uint row = idx / {q_vec_cols}u, vcol = idx % {q_vec_cols}u;
            *((threadgroup half4*)&sQ[row * {d}u + vcol * 4u]) =
                *((device const half4*)&Q[(off_m + (int)row) * d_head + (int)vcol * 4]);
        }}
    }}"""
    else:
        kv_load = f"""        for (uint _i = 0; _i < {K_LOADS}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {K_SIZE}u) {{
                uint row = idx / {d}u, col = idx % {d}u;
                sKV[idx] = KV_SRC[(kv_start + (int)row) * d_head + (int)col];
            }}
        }}"""
        q_load = f"""    for (uint _i = 0; _i < {Q_LOADS}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {Q_SIZE}u) {{
            uint row = idx / {d}u, col = idx % {d}u;
            sQ[idx] = Q[(off_m + (int)row) * d_head + (int)col];
        }}
    }}"""

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void flash_attention_opt(
    device const {io_type}* Q [[buffer(0)]],
    device const {io_type}* K [[buffer(1)]],
    device const {io_type}* V [[buffer(2)]],
    device {io_type}* O [[buffer(3)]],
    constant int& N       [[buffer(4)]],
    constant int& d_head  [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]]
) {{
    const uint t = tid.x;
    const uint sg = t / 32u;
    const uint lane = t % 32u;
    int off_m = (int)tgid.x * {BM};

    threadgroup {smem_type} sQ[{Q_SIZE}];
    threadgroup {smem_type} sKV[{K_SIZE}];
    threadgroup float sQK[{QK_SIZE}];
    threadgroup float sO[{O_SIZE}];

    // Load Q (persistent, vec4 if half)
{q_load}

    // Init sO to zero
    for (uint _i = 0; _i < {O_STORES}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {O_SIZE}u) sO[idx] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_i = -HUGE_VALF;
    float l_i = 0.0f;

    simdgroup_matrix<float, 8, 8> sg_QK[{QK_BPS}];
    simdgroup_matrix<float, 8, 8> sg_O[{PV_BPS}];
    for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
        uint blk = sg * {PV_BPS}u + bi;
        if (blk < {PV_BLOCKS}u) {{
            uint br = blk / {PV_BLOCKS_N}u, bc = blk % {PV_BLOCKS_N}u;
            simdgroup_load(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
        }}
    }}

    for (int kv_start = 0; kv_start < N; kv_start += {BN}) {{

        // Load K (reuse sKV)
        device const {io_type}* KV_SRC = K;
{kv_load}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK = Q @ K^T
        for (uint bi = 0; bi < {QK_BPS}u; bi++)
            sg_QK[bi] = simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint bi = 0; bi < {QK_BPS}u; bi++) {{
            uint blk = sg * {QK_BPS}u + bi;
            if (blk < {QK_BLOCKS}u) {{
                uint br = blk / {QK_BLOCKS_N}u, bc = blk % {QK_BLOCKS_N}u;
                for (uint kk = 0; kk < {d}u; kk += 8u) {{
                    simdgroup_matrix<{qk_mat_type}, 8, 8> sg_A, sg_B;
                    simdgroup_load(sg_A, &sQ[br * {8 * d}u + kk], {d}ul);
                    simdgroup_load(sg_B, &sKV[bc * {8 * d}u + kk], {d}ul, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sg_QK[bi], sg_A, sg_B, sg_QK[bi]);
                }}
            }}
        }}
        for (uint bi = 0; bi < {QK_BPS}u; bi++) {{
            uint blk = sg * {QK_BPS}u + bi;
            if (blk < {QK_BLOCKS}u) {{
                uint br = blk / {QK_BLOCKS_N}u, bc = blk % {QK_BLOCKS_N}u;
                simdgroup_store(sg_QK[bi], &sQK[br * {8 * BN}u + bc * 8u], {BN}ul);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        float qk_val = (t < {QK_SIZE}u) ? sQK[t] * scale : 0.0f;
        float row_max = simd_max(qk_val);
        float m_new = max(m_i, row_max);
        float alpha = exp(m_i - m_new);
        float p_val = exp(qk_val - m_new);
        float row_sum = simd_sum(p_val);
        l_i = l_i * alpha + row_sum;
        if (t < {QK_SIZE}u) sQK[t] = p_val;
        m_i = m_new;

        // Rescale O: O = O * alpha
        // Store O regs → sO, scale, reload
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u, bc = blk % {PV_BLOCKS_N}u;
                simdgroup_store(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scale sO rows by alpha and store alpha for later
        threadgroup float s_alpha[{BM}];
        if (lane == 0u && sg < {BM}u) s_alpha[sg] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint _i = 0; _i < {O_STORES}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {O_SIZE}u) {{
                uint row = idx / {d}u;
                sO[idx] *= s_alpha[row];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reload scaled O
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u, bc = blk % {PV_BLOCKS_N}u;
                simdgroup_load(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
            }}
        }}

        // Load V (reuse sKV)
        KV_SRC = V;
{kv_load}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P @ V
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u, bc = blk % {PV_BLOCKS_N}u;
                for (uint kk = 0; kk < {BN}u; kk += 8u) {{
                    simdgroup_matrix<float, 8, 8> sg_P;
                    simdgroup_matrix<{pv_load_type}, 8, 8> sg_V;
                    simdgroup_load(sg_P, &sQK[br * {8 * BN}u + kk], {BN}ul);
                    simdgroup_load(sg_V, &sKV[kk * {d}u + bc * 8u], {d}ul);
                    simdgroup_multiply_accumulate(sg_O[bi], sg_P, sg_V, sg_O[bi]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Final: store O regs → sO, normalize, write to global
    for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
        uint blk = sg * {PV_BPS}u + bi;
        if (blk < {PV_BLOCKS}u) {{
            uint br = blk / {PV_BLOCKS_N}u, bc = blk % {PV_BLOCKS_N}u;
            simdgroup_store(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float s_l[{BM}];
    if (lane == 0u && sg < {BM}u) s_l[sg] = l_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint _i = 0; _i < {O_STORES}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {O_SIZE}u) {{
            uint row = idx / {d}u;
            float inv_l = 1.0f / s_l[row];
            O[(off_m + (int)row) * d_head + (int)(idx % {d}u)] = ({io_type})(sO[idx] * inv_l);
        }}
    }}
}}
"""
    return msl, shmem_bytes


def pick_tile(d, use_half):
    BN = 32
    smem_elem = 2 if use_half else 4
    for BM in [64, 32, 16, 8]:
        total = BM * d * smem_elem + BN * d * smem_elem + BM * BN * 4 + BM * d * 4 + BM * 4
        if total <= 32768 and BM % 8 == 0:
            return BM
    return 8


def bench(N, d, BM=32, BN=32, iters=20, use_half=False):
    THREADS = 1024
    msl, shmem = generate_fa2_opt_msl(BM, BN, d, THREADS, use_half=use_half)
    pipe = compile_kernel(msl, "flash_attention_opt")
    grid = (N // BM, 1, 1)
    scale = 1.0 / math.sqrt(d)
    buf_dtype = 'e' if use_half else 'f'

    import random
    random.seed(42)
    q_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    k_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    v_data = [random.gauss(0, 0.5) for _ in range(N * d)]

    q_buf = make_buffer(q_data, buf_dtype)
    k_buf = make_buffer(k_data, buf_dtype)
    v_buf = make_buffer(v_data, buf_dtype)
    o_buf = make_zero_buffer(N * d, buf_dtype)
    bufs = [q_buf, k_buf, v_buf, o_buf, scalar_buf(N), scalar_buf(d), float_buf(scale)]

    for _ in range(3):
        o_buf = make_zero_buffer(N * d, buf_dtype)
        bufs[3] = o_buf
        dispatch(pipe, grid, THREADS, bufs)

    # Verify
    o_vals = read_buffer(o_buf, N * d, buf_dtype)
    q_row = q_data[0:d]
    scores = []
    for j in range(N):
        k_row = k_data[j*d:(j+1)*d]
        s = sum(q_row[k] * k_row[k] for k in range(d)) * scale
        scores.append(s)
    max_s = max(scores)
    exp_scores = [math.exp(s - max_s) for s in scores]
    sum_exp = sum(exp_scores)
    attn_weights = [e / sum_exp for e in exp_scores]
    ref_o = [0.0] * d
    for j in range(N):
        v_row = v_data[j*d:(j+1)*d]
        for k in range(d):
            ref_o[k] += attn_weights[j] * v_row[k]
    err = max(abs(o_vals[k] - ref_o[k]) for k in range(d))
    tol = 0.05 if use_half else 0.01
    ok = err < tol

    gpu_times = []
    for _ in range(iters):
        o_buf = make_zero_buffer(N * d, buf_dtype)
        bufs[3] = o_buf
        cb = dispatch(pipe, grid, THREADS, bufs)
        gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
        gpu_times.append(gpu_ms)

    gpu_times.sort()
    mn = gpu_times[len(gpu_times) // 2]
    flops = 4.0 * N * N * d
    gf = flops / (mn / 1000.0) / 1e9 if mn > 0 else 0
    return mn, gf, ok, err, shmem


# Run comparison: original vs optimized
print(f"Metal Device: {device.name()}")
print()

# Also run original for comparison
from bench_flash_attention import bench_flash_attention, _pick_tile

for use_half in [False, True]:
    dtype_str = "fp16" if use_half else "fp32"
    configs = [(256, 64), (512, 64), (1024, 64), (2048, 64), (4096, 64),
               (512, 128), (1024, 128), (2048, 128), (4096, 128)]

    print(f"=== Flash Attention 2 Comparison ({dtype_str}) ===")
    header = f"{'N':>6s}  {'d':>4s}  {'BM':>4s}  {'Orig ms':>8s}  {'Orig GF':>9s}  {'Opt ms':>8s}  {'Opt GF':>9s}  {'Speedup':>7s}  {'OK':>3s}"
    print(header)
    print('-' * len(header))

    BN = 32
    for N, d in configs:
        BM_orig = _pick_tile(d, use_half)
        BM_opt = pick_tile(d, use_half)
        if N % BM_orig != 0 or N % BM_opt != 0:
            continue
        try:
            _, _, _, gf_orig, ok_orig, _ = bench_flash_attention(N, d, BM_orig, BN, use_half=use_half)
            t_opt, gf_opt, ok_opt, err_opt, shmem = bench(N, d, BM_opt, BN, use_half=use_half)
            speedup = f"{gf_opt/gf_orig:.2f}x" if gf_orig > 0 else "N/A"
            t_orig = 4.0 * N * N * d / (gf_orig * 1e9) * 1000 if gf_orig > 0 else 0
            print(f"  {N:>4d}  {d:>4d}  {BM_opt:>4d}  {t_orig:>8.3f}  {gf_orig:>9.1f}  {t_opt:>8.3f}  {gf_opt:>9.1f}  {speedup:>7s}  {'Y' if ok_opt else 'N'}")
        except Exception as e:
            print(f"  {N:>4d}  {d:>4d}  {BM_opt:>4d}  ERROR: {e}")
    print()
