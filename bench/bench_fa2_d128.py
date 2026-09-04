#!/usr/bin/env python3
"""Flash Attention 2 optimized for d=128: eliminate sO via in-register rescaling.

Key insight: sO[BM*d] dominates shared memory for d=128. By keeping O in
simdgroup registers and using diagonal matrix multiply for per-row rescaling,
we eliminate sO entirely. This allows larger BM tiles:
  - fp16: BM 16→32 (2x), fp32: BM 8→16 (2x)

The diagonal trick: to scale row i of an 8x8 O block by alpha_i, left-multiply
by diag(alpha_0..alpha_7): simdgroup_multiply(result, diag_matrix, O_block).
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


def generate_fa2_d128_msl(BM, BN, d, THREADS=1024, use_half=False):
    """Generate FA2 kernel with NO sO in shared memory.

    O is kept entirely in simdgroup registers. Rescaling uses diagonal
    matrix multiply: diag(alpha_per_row) @ O_block.
    """
    assert BN == 32 and BM % 8 == 0 and d % 8 == 0

    io_type = "half" if use_half else "float"
    smem_type = "half" if use_half else "float"
    smem_elem = 2 if use_half else 4

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

    Q_LOADS = max(1, (Q_SIZE + THREADS - 1) // THREADS)
    K_LOADS = max(1, (K_SIZE + THREADS - 1) // THREADS)

    NUM_DIAGS = QK_BLOCKS_M
    DIAG_TOTAL = NUM_DIAGS * 64  # 64 floats per 8x8 diagonal matrix

    # For half: QK matmul uses half×half→float, P@V uses half(P_stored_as_float)×half→float
    qk_mat_type = "half" if use_half else "float"

    # Memory: sQ + sKV + sQK(float) + s_alpha + s_diags
    shmem_bytes = (Q_SIZE * smem_elem + K_SIZE * smem_elem +
                   QK_SIZE * 4 + BM * 4 + DIAG_TOTAL * 4)

    # Final store scratch: reuse sQK as float scratch for fp16→device conversion
    # sQK has QK_SIZE floats. We need 8*d floats per chunk. Check: QK_SIZE >= 8*d.
    CHUNK_SIZE = 8 * d  # floats needed for one 8-row output chunk
    assert QK_SIZE >= CHUNK_SIZE or not use_half, \
        f"sQK ({QK_SIZE}) too small for output chunk ({CHUNK_SIZE})"

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void flash_attention_d128(
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

    // Shared memory — NO sO!
    threadgroup {smem_type} sQ[{Q_SIZE}];
    threadgroup {smem_type} sKV[{K_SIZE}];
    threadgroup float sQK[{QK_SIZE}];
    threadgroup float s_alpha[{BM}];
    threadgroup float s_diags[{DIAG_TOTAL}];

    // Load Q (persistent)
    for (uint _i = 0; _i < {Q_LOADS}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {Q_SIZE}u) {{
            uint row = idx / {d}u, col = idx % {d}u;
            sQ[idx] = Q[(off_m + (int)row) * d_head + (int)col];
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O accumulators in simdgroup registers — never touches shared memory!
    simdgroup_matrix<float, 8, 8> sg_O[{PV_BPS}];
    for (uint bi = 0; bi < {PV_BPS}u; bi++)
        sg_O[bi] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Per-row running max and sum (in registers, one row per SG)
    float m_i = -HUGE_VALF;
    float l_i = 0.0f;

    // QK register accumulators
    simdgroup_matrix<float, 8, 8> sg_QK[{QK_BPS}];

    // Main K/V loop
    for (int kv_start = 0; kv_start < N; kv_start += {BN}) {{

        // Load K
        for (uint _i = 0; _i < {K_LOADS}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {K_SIZE}u) {{
                uint row = idx / {d}u, col = idx % {d}u;
                sKV[idx] = K[(kv_start + (int)row) * d_head + (int)col];
            }}
        }}
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
                    simdgroup_load(sg_B, &sKV[bc * {8 * d}u + kk], {d}ul, ulong2(0,0), true);
                    simdgroup_multiply_accumulate(sg_QK[bi], sg_A, sg_B, sg_QK[bi]);
                }}
            }}
        }}
        // Store QK to shared memory (float for softmax precision)
        for (uint bi = 0; bi < {QK_BPS}u; bi++) {{
            uint blk = sg * {QK_BPS}u + bi;
            if (blk < {QK_BLOCKS}u) {{
                uint br = blk / {QK_BLOCKS_N}u, bc = blk % {QK_BLOCKS_N}u;
                simdgroup_store(sg_QK[bi], &sQK[br * {8 * BN}u + bc * 8u], {BN}ul);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax (BN=32 = SIMD width, 1 element per thread)
        float qk_val = 0.0f;
        if (t < {QK_SIZE}u) qk_val = sQK[t] * scale;

        float row_max = simd_max(qk_val);
        float m_new = max(m_i, row_max);
        float alpha = exp(m_i - m_new);
        float p_val = exp(qk_val - m_new);
        float row_sum = simd_sum(p_val);
        l_i = l_i * alpha + row_sum;
        m_i = m_new;

        if (t < {QK_SIZE}u) sQK[t] = p_val;
        if (lane == 0u && sg < {BM}u) s_alpha[sg] = alpha;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Build diagonal matrices from s_alpha for O rescaling
        for (uint idx = t; idx < {DIAG_TOTAL}u; idx += {THREADS}u) {{
            uint which_br = idx / 64u;
            uint local = idx % 64u;
            uint r = local / 8u, c = local % 8u;
            s_diags[idx] = (r == c) ? s_alpha[which_br * 8u + r] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale O in registers: O = diag(alpha) @ O
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u;
                simdgroup_matrix<float, 8, 8> sg_diag, sg_tmp;
                simdgroup_load(sg_diag, &s_diags[br * 64u], 8ul);
                simdgroup_multiply(sg_tmp, sg_diag, sg_O[bi]);
                sg_O[bi] = sg_tmp;
            }}
        }}
        // No barrier needed — each SG only touches its own registers

        // Load V (reuse sKV)
        for (uint _i = 0; _i < {K_LOADS}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {K_SIZE}u) {{
                uint row = idx / {d}u, col = idx % {d}u;
                sKV[idx] = V[(kv_start + (int)row) * d_head + (int)col];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O += P @ V
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u, bc = blk % {PV_BLOCKS_N}u;
                for (uint kk = 0; kk < {BN}u; kk += 8u) {{
                    simdgroup_matrix<float, 8, 8> sg_P;
                    simdgroup_matrix<{qk_mat_type}, 8, 8> sg_V;
                    simdgroup_load(sg_P, &sQK[br * {8 * BN}u + kk], {BN}ul);
                    simdgroup_load(sg_V, &sKV[kk * {d}u + bc * 8u], {d}ul);
                    simdgroup_multiply_accumulate(sg_O[bi], sg_P, sg_V, sg_O[bi]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // === Final normalization: O = diag(1/l) @ O, then store ===

    // Write l_i to s_alpha (reuse)
    if (lane == 0u && sg < {BM}u) s_alpha[sg] = l_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build inverse-l diagonal matrices (reuse s_diags)
    for (uint idx = t; idx < {DIAG_TOTAL}u; idx += {THREADS}u) {{
        uint which_br = idx / 64u;
        uint local = idx % 64u;
        uint r = local / 8u, c = local % 8u;
        s_diags[idx] = (r == c) ? (1.0f / s_alpha[which_br * 8u + r]) : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize O in registers
    for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
        uint blk = sg * {PV_BPS}u + bi;
        if (blk < {PV_BLOCKS}u) {{
            uint br = blk / {PV_BLOCKS_N}u;
            simdgroup_matrix<float, 8, 8> sg_inv_l, sg_norm;
            simdgroup_load(sg_inv_l, &s_diags[br * 64u], 8ul);
            simdgroup_multiply(sg_norm, sg_inv_l, sg_O[bi]);
            sg_O[bi] = sg_norm;
        }}
    }}
"""

    if not use_half:
        # fp32: store directly from registers to device memory
        msl += f"""
    // Store O directly to device memory (fp32)
    for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
        uint blk = sg * {PV_BPS}u + bi;
        if (blk < {PV_BLOCKS}u) {{
            uint br = blk / {PV_BLOCKS_N}u, bc = blk % {PV_BLOCKS_N}u;
            simdgroup_store(sg_O[bi], &O[(off_m + (int)(br * 8u)) * d_head + (int)(bc * 8u)], (ulong)d_head);
        }}
    }}
}}
"""
    else:
        # fp16: store via shared scratch (reuse sQK), convert float→half
        msl += f"""
    // Store O to device (fp16): chunk by row-block, convert via sQK scratch
    threadgroup float* s_scratch = (threadgroup float*)sQK;  // reuse sQK as float scratch

    for (uint br_idx = 0; br_idx < {QK_BLOCKS_M}u; br_idx++) {{
        // SGs with this br store their O blocks to scratch
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u;
                uint bc = blk % {PV_BLOCKS_N}u;
                if (br == br_idx) {{
                    simdgroup_store(sg_O[bi], &s_scratch[bc * 8u], {d}ul);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Convert float→half and write to device
        for (uint idx = t; idx < {CHUNK_SIZE}u; idx += {THREADS}u) {{
            uint row = idx / {d}u, col = idx % {d}u;
            O[(off_m + (int)(br_idx * 8u + row)) * d_head + (int)col] = (half)s_scratch[idx];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
}}
"""

    return msl, shmem_bytes


def pick_BM_d128(d, use_half):
    """Pick optimal BM for d=128 with NO sO."""
    BN = 32
    smem_elem = 2 if use_half else 4
    for BM in [32, 24, 16, 8]:
        if BM % 8 != 0:
            continue
        QK_BLOCKS_M = BM // 8
        # sQ + sKV + sQK(float) + s_alpha + s_diags
        total = (BM * d * smem_elem + BN * d * smem_elem +
                 BM * BN * 4 + BM * 4 + QK_BLOCKS_M * 64 * 4)
        # Also need QK_SIZE >= CHUNK_SIZE for fp16 store
        if use_half and BM * BN < 8 * d:
            continue
        # QK_SIZE must fit in single softmax pass (≤ THREADS)
        if BM * BN > 1024:
            continue
        if total <= 32768:
            return BM
    return 8


def bench(N, d, BM, BN=32, iters=20, use_half=False):
    THREADS = 1024
    msl, shmem = generate_fa2_d128_msl(BM, BN, d, THREADS, use_half)
    pipe = compile_kernel(msl, "flash_attention_d128")
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

    o_vals = read_buffer(o_buf, N * d, buf_dtype)
    q_row = q_data[0:d]
    scores = [sum(q_row[k] * k_data[j*d+k] for k in range(d)) * scale for j in range(N)]
    max_s = max(scores)
    exp_s = [math.exp(s - max_s) for s in scores]
    sum_e = sum(exp_s)
    weights = [e / sum_e for e in exp_s]
    ref_o = [sum(weights[j] * v_data[j*d+k] for j in range(N)) for k in range(d)]
    err = max(abs(o_vals[k] - ref_o[k]) for k in range(d))
    tol = 0.05 if use_half else 0.01
    ok = err < tol

    gpu_times = []
    for _ in range(iters):
        o_buf = make_zero_buffer(N * d, buf_dtype)
        bufs[3] = o_buf
        cb = dispatch(pipe, grid, THREADS, bufs)
        gpu_times.append((cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0)

    gpu_times.sort()
    mn = gpu_times[len(gpu_times) // 2]
    flops = 4.0 * N * N * d
    gf = flops / (mn / 1000.0) / 1e9 if mn > 0 else 0
    return mn, gf, ok, err, shmem


# Also import original for comparison
from bench_flash_attention import bench_flash_attention, _pick_tile

print(f"Metal Device: {device.name()}")
print()

configs = [(512, 128), (1024, 128), (2048, 128), (4096, 128)]

for use_half in [False, True]:
    dtype_str = "fp16" if use_half else "fp32"
    print(f"=== d=128 Flash Attention 2 ({dtype_str}): Original vs No-sO ===")
    header = (f"{'N':>6s}  {'d':>4s}  {'BM_old':>6s}  {'BM_new':>6s}  "
              f"{'Old ms':>8s}  {'Old GF':>9s}  {'New ms':>8s}  {'New GF':>9s}  "
              f"{'Speedup':>7s}  {'Err':>8s}  {'OK':>3s}  {'shmem':>7s}")
    print(header)
    print('-' * len(header))

    for N, d in configs:
        BM_old = _pick_tile(d, use_half)
        BM_new = pick_BM_d128(d, use_half)
        if N % BM_old != 0 or N % BM_new != 0:
            continue
        try:
            avg_o, mn_o, gf_avg_o, gf_o, ok_o, err_o = bench_flash_attention(N, d, BM_old, 32, use_half=use_half)
            mn_n, gf_n, ok_n, err_n, shmem_n = bench(N, d, BM_new, use_half=use_half)
            speedup = f"{gf_n/gf_o:.2f}x" if gf_o > 0 else "N/A"
            print(f"  {N:>4d}  {d:>4d}  {BM_old:>6d}  {BM_new:>6d}  "
                  f"{mn_o:>8.3f}  {gf_o:>9.1f}  {mn_n:>8.3f}  {gf_n:>9.1f}  "
                  f"{speedup:>7s}  {err_n:>8.5f}  {'Y' if ok_n else 'N'}  {shmem_n:>5d}B")
        except Exception as e:
            print(f"  {N:>4d}  {d:>4d}  {BM_old:>6d}  {BM_new:>6d}  ERROR: {e}")
    print()
