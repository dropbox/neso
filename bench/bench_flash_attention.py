#!/usr/bin/env python3
"""Flash Attention 2 forward pass on Metal.

Implements the online softmax algorithm from Dao et al. using
simdgroup_matrix hardware acceleration on Apple Silicon.

Compares against MPS scaled_dot_product_attention as baseline.
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


def generate_flash_attention_msl(BM, BN, d, THREADS=1024, use_half=False):
    """Generate a Flash Attention 2 forward kernel in MSL.

    Algorithm (online softmax from Dao et al.):
      For each query block [BM, d]:
        Load Q once into shared memory
        For each K/V block [BN, d]:
          QK = Q @ K^T                    [BM, BN]
          m_new = max(m_old, rowmax(QK * scale))
          alpha = exp(m_old - m_new)
          P = exp(QK * scale - m_new)
          l = l * alpha + rowsum(P)
          O = O * alpha + P @ V
          m = m_new
        O = O / l

    Thread mapping:
      - 1024 threads, 32 SIMD groups of 32 threads
      - For BM×BN (e.g. 32×32 = 1024): 1 element per thread, row = tid/BN
      - Row-wise reductions use simd_max/simd_sum when BN = SIMD width (32)

    Uses simdgroup_matrix for Q@K^T and P@V matmuls.

    When use_half=True:
      - Q/K/V/O device buffers are half precision
      - sQ, sKV shared memory is half (halves bandwidth)
      - QK matmul: half x half -> float accumulator (mixed precision)
      - Softmax computed in float for numerical stability
      - P stored as half for P@V matmul
      - sO accumulator kept in float, converted to half at output
    """
    assert BN == 32, "BN must be 32 (SIMD width) for simd reductions"
    assert BM % 8 == 0 and BN % 8 == 0 and d % 8 == 0

    # Type strings for MSL
    io_type = "half" if use_half else "float"
    smem_type = "half" if use_half else "float"

    NUM_SG = THREADS // 32
    # For QK [BM, BN]: each 8x8 block gets one SG
    QK_BLOCKS_M = BM // 8
    QK_BLOCKS_N = BN // 8
    QK_BLOCKS = QK_BLOCKS_M * QK_BLOCKS_N
    QK_BPS = max(1, (QK_BLOCKS + NUM_SG - 1) // NUM_SG)

    # For PV [BM, d]: each 8x8 block gets assigned to SGs
    PV_BLOCKS_M = BM // 8
    PV_BLOCKS_N = d // 8
    PV_BLOCKS = PV_BLOCKS_M * PV_BLOCKS_N
    PV_BPS = max(1, (PV_BLOCKS + NUM_SG - 1) // NUM_SG)

    # Memory sizes (element counts)
    Q_SIZE = BM * d
    K_SIZE = BN * d
    QK_SIZE = BM * BN
    O_SIZE = BM * d

    # Per-thread load counts
    Q_LOADS = max(1, (Q_SIZE + THREADS - 1) // THREADS)
    K_LOADS = max(1, (K_SIZE + THREADS - 1) // THREADS)
    O_ELEMS = max(1, (O_SIZE + THREADS - 1) // THREADS)

    # For half path: QK matmul uses half inputs with float accumulators,
    # softmax in float, P stored as half for P@V, O accumulated in float.
    # sQK is float (for softmax), with a half alias for P storage.
    qk_mat_type = "half" if use_half else "float"  # simdgroup_matrix element type for Q,K,V loads
    pv_load_type = "half" if use_half else "float"  # type for loading P in P@V

    msl = f"""#include <metal_stdlib>
using namespace metal;

kernel void flash_attention_fwd(
    device const {io_type}* Q [[buffer(0)]],
    device const {io_type}* K [[buffer(1)]],
    device const {io_type}* V [[buffer(2)]],
    device {io_type}* O [[buffer(3)]],
    constant int& N       [[buffer(4)]],   // sequence length
    constant int& d_head  [[buffer(5)]],   // head dimension
    constant float& scale [[buffer(6)]],   // 1/sqrt(d)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]]
) {{
    const uint t = tid.x;
    const uint sg = t / 32u;
    const uint lane = t % 32u;

    // This threadgroup handles query rows [off_m, off_m + BM)
    int off_m = (int)tgid.x * {BM};

    // Shared memory
    threadgroup {smem_type} sQ[{Q_SIZE}];    // [{BM}][{d}]
    threadgroup {smem_type} sKV[{K_SIZE}];   // [{BN}][{d}] - reused for K and V
    threadgroup float sQK[{QK_SIZE}];        // [{BM}][{BN}] - float for softmax precision
    threadgroup float sO[{O_SIZE}];          // [{BM}][{d}] - float accumulator

    // Load Q tile into shared memory (persistent for entire kernel)
    for (uint _i = 0; _i < {Q_LOADS}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {Q_SIZE}u) {{
            uint row = idx / {d}u;
            uint col = idx % {d}u;
            sQ[idx] = Q[(off_m + (int)row) * d_head + (int)col];
        }}
    }}

    // Initialize output accumulator to zero
    for (uint _i = 0; _i < {O_ELEMS}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {O_SIZE}u) sO[idx] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-row running max and log-sum-exp (in registers)
    float m_i = -HUGE_VALF;  // running max
    float l_i = 0.0f;         // running sum of exp

    // simdgroup accumulators for QK [BM, BN] - always float for precision
    simdgroup_matrix<float, 8, 8> sg_QK[{QK_BPS}];

    // simdgroup accumulators for O [BM, d] - always float for precision
    simdgroup_matrix<float, 8, 8> sg_O[{PV_BPS}];
    // Initialize O accumulators from sO
    for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
        uint blk = sg * {PV_BPS}u + bi;
        if (blk < {PV_BLOCKS}u) {{
            uint br = blk / {PV_BLOCKS_N}u;
            uint bc = blk % {PV_BLOCKS_N}u;
            simdgroup_load(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
        }}
    }}

    // Main loop over K/V blocks
    for (int kv_start = 0; kv_start < N; kv_start += {BN}) {{

        // --- Load K tile ---
        for (uint _i = 0; _i < {K_LOADS}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {K_SIZE}u) {{
                uint row = idx / {d}u;
                uint col = idx % {d}u;
                sKV[idx] = K[(kv_start + (int)row) * d_head + (int)col];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Compute QK = Q @ K^T  [BM, BN] ---
        // Zero QK accumulators
        for (uint bi = 0; bi < {QK_BPS}u; bi++) {{
            sg_QK[bi] = simdgroup_matrix<float, 8, 8>(0.0f);
        }}

        // QK += Q_block @ K_block^T, iterating over d in steps of 8
        for (uint bi = 0; bi < {QK_BPS}u; bi++) {{
            uint blk = sg * {QK_BPS}u + bi;
            if (blk < {QK_BLOCKS}u) {{
                uint br = blk / {QK_BLOCKS_N}u;
                uint bc = blk % {QK_BLOCKS_N}u;
                for (uint kk = 0; kk < {d}u; kk += 8u) {{
                    simdgroup_matrix<{qk_mat_type}, 8, 8> sg_A, sg_B;
                    simdgroup_load(sg_A, &sQ[br * {8 * d}u + kk], {d}ul);
                    simdgroup_load(sg_B, &sKV[bc * {8 * d}u + kk], {d}ul, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sg_QK[bi], sg_A, sg_B, sg_QK[bi]);
                }}
            }}
        }}

        // Store QK to shared memory (float for softmax precision)
        for (uint bi = 0; bi < {QK_BPS}u; bi++) {{
            uint blk = sg * {QK_BPS}u + bi;
            if (blk < {QK_BLOCKS}u) {{
                uint br = blk / {QK_BLOCKS_N}u;
                uint bc = blk % {QK_BLOCKS_N}u;
                simdgroup_store(sg_QK[bi], &sQK[br * {8 * BN}u + bc * 8u], {BN}ul);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Online softmax (always in float) ---
        float qk_val = 0.0f;
        if (t < {QK_SIZE}u) {{
            qk_val = sQK[t] * scale;
        }}

        float row_max = simd_max(qk_val);
        float m_new = max(m_i, row_max);
        float alpha = exp(m_i - m_new);
        float p_val = exp(qk_val - m_new);
        float row_sum = simd_sum(p_val);
        l_i = l_i * alpha + row_sum;

        // Store P to shared memory
        if (t < {QK_SIZE}u) {{
            sQK[t] = p_val;
        }}
        m_i = m_new;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Rescale O accumulator by alpha ---
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u;
                uint bc = blk % {PV_BLOCKS_N}u;
                simdgroup_store(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float s_alpha[{BM}];
        if (lane == 0u && sg < {BM}u) {{
            s_alpha[sg] = alpha;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint _i = 0; _i < {O_ELEMS}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {O_SIZE}u) {{
                uint row = idx / {d}u;
                sO[idx] *= s_alpha[row];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reload O into simdgroup registers
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u;
                uint bc = blk % {PV_BLOCKS_N}u;
                simdgroup_load(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
            }}
        }}

        // --- Load V tile (reuse sKV) ---
        for (uint _i = 0; _i < {K_LOADS}u; _i++) {{
            uint idx = t + _i * {THREADS}u;
            if (idx < {K_SIZE}u) {{
                uint row = idx / {d}u;
                uint col = idx % {d}u;
                sKV[idx] = V[(kv_start + (int)row) * d_head + (int)col];
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Compute O += P @ V  [BM, d] ---
        // P is [BM, BN] in sQK (float), V is [BN, d] in sKV
        for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
            uint blk = sg * {PV_BPS}u + bi;
            if (blk < {PV_BLOCKS}u) {{
                uint br = blk / {PV_BLOCKS_N}u;
                uint bc = blk % {PV_BLOCKS_N}u;
                for (uint kk = 0; kk < {BN}u; kk += 8u) {{
                    // P is float in sQK, V is {smem_type} in sKV
                    // For half path: load P as float, V as half -> mixed precision accumulate
                    // For float path: both float
                    simdgroup_matrix<float, 8, 8> sg_P;
                    simdgroup_matrix<{pv_load_type}, 8, 8> sg_V;
                    simdgroup_load(sg_P, &sQK[br * {8 * BN}u + kk], {BN}ul);
                    simdgroup_load(sg_V, &sKV[kk * {d}u + bc * 8u], {d}ul);
                    simdgroup_multiply_accumulate(sg_O[bi], sg_P, sg_V, sg_O[bi]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }} // end K/V loop

    // --- Store final O from simdgroup registers to shared memory ---
    for (uint bi = 0; bi < {PV_BPS}u; bi++) {{
        uint blk = sg * {PV_BPS}u + bi;
        if (blk < {PV_BLOCKS}u) {{
            uint br = blk / {PV_BLOCKS_N}u;
            uint bc = blk % {PV_BLOCKS_N}u;
            simdgroup_store(sg_O[bi], &sO[br * {8 * d}u + bc * 8u], {d}ul);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Final normalization: O = O / l_i ---
    threadgroup float s_l[{BM}];
    if (lane == 0u && sg < {BM}u) {{
        s_l[sg] = l_i;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint _i = 0; _i < {O_ELEMS}u; _i++) {{
        uint idx = t + _i * {THREADS}u;
        if (idx < {O_SIZE}u) {{
            uint row = idx / {d}u;
            float inv_l = 1.0f / s_l[row];
            O[(off_m + (int)(idx / {d}u)) * d_head + (int)(idx % {d}u)] = ({io_type})(sO[idx] * inv_l);
        }}
    }}
}}
"""
    return msl


def bench_flash_attention(N, d, BM=32, BN=32, iters=20, use_half=False):
    """Benchmark Flash Attention 2 forward pass."""
    THREADS = 1024
    msl = generate_flash_attention_msl(BM, BN, d, THREADS, use_half=use_half)
    pipe = compile_kernel(msl, "flash_attention_fwd")

    grid = (N // BM, 1, 1)
    scale = 1.0 / math.sqrt(d)

    buf_dtype = 'e' if use_half else 'f'  # 'e' = IEEE 754 half

    import random
    random.seed(42)
    q_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    k_data = [random.gauss(0, 0.5) for _ in range(N * d)]
    v_data = [random.gauss(0, 0.5) for _ in range(N * d)]

    q_buf = make_buffer(q_data, buf_dtype)
    k_buf = make_buffer(k_data, buf_dtype)
    v_buf = make_buffer(v_data, buf_dtype)
    o_buf = make_zero_buffer(N * d, buf_dtype)
    n_buf = scalar_buf(N)
    d_buf = scalar_buf(d)
    s_buf = float_buf(scale)

    bufs = [q_buf, k_buf, v_buf, o_buf, n_buf, d_buf, s_buf]

    # Warmup
    for _ in range(3):
        o_buf = make_zero_buffer(N * d, buf_dtype)
        bufs[3] = o_buf
        dispatch(pipe, grid, THREADS, bufs)

    # Timed runs
    gpu_times = []
    for _ in range(iters):
        o_buf = make_zero_buffer(N * d, buf_dtype)
        bufs[3] = o_buf
        cb = dispatch(pipe, grid, THREADS, bufs)
        gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
        gpu_times.append(gpu_ms)

    avg = sum(gpu_times) / len(gpu_times)
    mn = min(gpu_times)
    # FLOPS for attention: 2 * N * N * d (for QK) + 2 * N * N * d (for PV) = 4 * N^2 * d
    flops = 4.0 * N * N * d
    gflops_avg = flops / (avg / 1000.0) / 1e9
    gflops_peak = flops / (mn / 1000.0) / 1e9

    # Correctness check: compute reference attention for first row
    o_vals = read_buffer(o_buf, N * d, buf_dtype)

    # Reference: softmax(Q[0,:] @ K^T * scale) @ V
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

    # Check first few elements (looser tolerance for fp16)
    err = max(abs(o_vals[k] - ref_o[k]) for k in range(d))
    tol = 0.05 if use_half else 0.01
    ok = err < tol

    return avg, mn, gflops_avg, gflops_peak, ok, err


def bench_mps_attention(N, d, num_heads=1, iters=20, use_half=False):
    """Benchmark PyTorch MPS scaled_dot_product_attention."""
    import torch
    torch.mps.synchronize()

    dtype = torch.float16 if use_half else torch.float32
    # Single head, batch=1
    q = torch.randn(1, num_heads, N, d, device='mps', dtype=dtype)
    k = torch.randn(1, num_heads, N, d, device='mps', dtype=dtype)
    v = torch.randn(1, num_heads, N, d, device='mps', dtype=dtype)

    scale = 1.0 / math.sqrt(d)

    # Warmup
    for _ in range(5):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.mps.synchronize()

    times = []
    for _ in range(iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    flops = 4.0 * N * N * d * num_heads
    return avg_ms, min_ms, flops / (avg_ms / 1000.0) / 1e9, flops / (min_ms / 1000.0) / 1e9


print(f"Metal Device: {device.name()}")
print()

def _pick_tile(d, use_half):
    """Pick BM based on threadgroup memory budget (32KB).

    Memory layout: sQ(smem BM*d) + sKV(smem BN*d) + sQK(float BM*BN) + sO(float BM*d)
    where smem is half (2B) when use_half, else float (4B).
    """
    BN = 32
    smem_bytes = 2 if use_half else 4
    for BM in [32, 16, 8]:
        total = BM * d * smem_bytes + BN * d * smem_bytes + BM * BN * 4 + BM * d * 4
        if total <= 32768:
            return BM
    return 8


configs = [
    (256, 64),
    (512, 64),
    (1024, 64),
    (2048, 64),
    (4096, 64),
    (512, 128),
    (1024, 128),
    (2048, 128),
    (4096, 128),
]

for use_half in [False, True]:
    dtype_str = "fp16" if use_half else "fp32"

    # Filter configs to those that fit in threadgroup memory
    valid_configs = [(N, d) for N, d in configs if _pick_tile(d, use_half) is not None]

    # --- MPS baseline ---
    print(f"=== MPS baseline (scaled_dot_product_attention, {dtype_str}) ===")
    print(f"{'N':>6s}  {'d':>4s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}")
    print("-" * 50)

    mps_results = {}
    for N, d in valid_configs:
        avg, mn, gf_avg, gf_peak = bench_mps_attention(N, d, use_half=use_half)
        mps_results[(N, d)] = gf_peak
        print(f"  {N:>4d}  {d:>4d}  {avg:>8.3f}  {mn:>8.3f}  {gf_peak:>9.1f}")

    # --- Flash Attention benchmark ---
    print()
    print(f"=== Flash Attention 2 Forward (single head, {dtype_str}) ===")
    print(f"{'N':>6s}  {'d':>4s}  {'Tile':>7s}  {'Avg (ms)':>8s}  {'Min (ms)':>8s}  {'GFLOP/s':>9s}  {'vs MPS':>7s}  {'Err':>8s}  {'OK':>3s}")
    print("-" * 80)

    for N, d in valid_configs:
        BM = _pick_tile(d, use_half)
        BN = 32
        tile_str = f"{BM}x{BN}"
        avg, mn, gf_avg, gf_peak, ok, err = bench_flash_attention(N, d, BM, BN, use_half=use_half)
        mps_peak = mps_results.get((N, d), 0)
        ratio = f"{gf_peak/mps_peak:.1f}x" if mps_peak > 0 else "N/A"
        print(f"  {N:>4d}  {d:>4d}  {tile_str:>7s}  {avg:>8.3f}  {mn:>8.3f}  {gf_peak:>9.1f}  {ratio:>7s}  {err:>8.5f}  {'Y' if ok else 'N'}")
    print()
