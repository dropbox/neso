#!/usr/bin/env python3
"""Benchmark: optimized matmul MSL kernels vs current codegen vs MPS.

Tests key optimizations:
1. Eliminate sC from threadgroup memory (store from simdgroup registers)
2. Register tiling: each simdgroup computes TM*TN 8x8 blocks with A/B reuse
3. Larger output tiles (128x128) enabled by freed shared memory
"""
import sys, os, struct, time
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


def gen_coop_load(dest, base_ptr, stride, cols, total, loads, threads, row_off, col_off, indent=2, vec4=False):
    """Generate cooperative tile loading code. vec4=True uses half4 vectorized loads."""
    pad = "    " * indent
    lines = []
    if vec4 and cols % 4 == 0:
        # Vectorized: each thread loads 4 half values at once
        vec_total = total // 4
        vec_cols = cols // 4
        vec_loads = max(1, (vec_total + threads - 1) // threads)
        if vec_loads > 1:
            lines.append(f"{pad}for (uint _li = 0; _li < {vec_loads}u; _li++) {{")
            lines.append(f"{pad}    uint _vflat = tid + _li * {threads}u;")
            lines.append(f"{pad}    if (_vflat < {vec_total}u) {{")
            lines.append(f"{pad}        uint _r = _vflat / {vec_cols}u, _vc = _vflat % {vec_cols}u;")
            lines.append(f"{pad}        *((threadgroup half4*)&{dest}[_r * {cols}u + _vc * 4u]) = *((device const half4*)&{base_ptr}[({row_off} + (int)_r) * {stride} + ({col_off} + (int)_vc * 4)]);")
            lines.append(f"{pad}    }}")
            lines.append(f"{pad}}}")
        else:
            lines.append(f"{pad}{{ uint _vflat = tid, _r = _vflat / {vec_cols}u, _vc = _vflat % {vec_cols}u;")
            lines.append(f"{pad}  if (_vflat < {vec_total}u)")
            lines.append(f"{pad}    *((threadgroup half4*)&{dest}[_r * {cols}u + _vc * 4u]) = *((device const half4*)&{base_ptr}[({row_off} + (int)_r) * {stride} + ({col_off} + (int)_vc * 4)]); }}")
        return "\n".join(lines)

    if loads > 1:
        lines.append(f"{pad}for (uint _li = 0; _li < {loads}u; _li++) {{")
        lines.append(f"{pad}    uint _flat = tid + _li * {threads}u;")
        lines.append(f"{pad}    if (_flat < {total}u) {{")
        lines.append(f"{pad}        uint _r = _flat / {cols}u, _c = _flat % {cols}u;")
        lines.append(f"{pad}        {dest}[_flat] = {base_ptr}[({row_off} + (int)_r) * {stride} + ({col_off} + (int)_c)];")
        lines.append(f"{pad}    }}")
        lines.append(f"{pad}}}")
    else:
        lines.append(f"{pad}{{ uint _flat = tid, _r = _flat / {cols}u, _c = _flat % {cols}u;")
        lines.append(f"{pad}  {dest}[_flat] = {base_ptr}[({row_off} + (int)_r) * {stride} + ({col_off} + (int)_c)]; }}")
    return "\n".join(lines)


def generate_optimized_msl(BM, BN, BK, TM, TN, THREADS=1024, double_buf=True,
                           vec4=False, swizzle=False):
    """Generate optimized matmul: no sC, register tiling, direct device store.

    Output is float32 (accumulator type). C buffer must be float.
    vec4: use half4 vectorized global→shared loads
    swizzle: reorder threadgroup blocks for better L2 locality
    """
    NUM_SG = THREADS // 32
    BLOCKS_M = BM // 8
    BLOCKS_N = BN // 8
    SUPER_M = BLOCKS_M // TM
    SUPER_N = BLOCKS_N // TN
    TOTAL_SUPER = SUPER_M * SUPER_N
    assert TOTAL_SUPER == NUM_SG, \
        f"Need {TOTAL_SUPER} super-blocks but have {NUM_SG} simdgroups. " \
        f"Adjust TM/TN so (BM/8/TM)*(BN/8/TN) == {NUM_SG}"

    A_ELEMS = BM * BK
    B_ELEMS = BK * BN
    A_LOADS = (A_ELEMS + THREADS - 1) // THREADS
    B_LOADS = (B_ELEMS + THREADS - 1) // THREADS
    K_BLOCKS = BK // 8

    shmem = (2 if double_buf else 1) * (A_ELEMS + B_ELEMS) * 2
    assert shmem <= 32768, f"Shared memory {shmem} > 32KB"

    # Build MSL
    s = f"""#include <metal_stdlib>
using namespace metal;

kernel void matmul_opt(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const int& K_param [[buffer(3)]],
    device const int& stride_am [[buffer(4)]],
    device const int& stride_bk [[buffer(5)]],
    device const int& stride_cm [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{{
"""
    if swizzle:
        # Swizzled block assignment: tiles in GROUP_SIZE x GROUP_SIZE groups
        # improves L2 cache reuse between adjacent threadgroups
        s += """    // Swizzled block assignment for L2 cache locality
    const int GROUP_SIZE = 8;
    int num_pid_n = (int)tgid.z;  // passed via grid.z
    int pid_linear = (int)tgid.x * num_pid_n + (int)tgid.y;
    int num_groups = (num_pid_n + GROUP_SIZE - 1) / GROUP_SIZE;
    int group_id = pid_linear / (GROUP_SIZE * num_pid_n);
    int within = pid_linear % (GROUP_SIZE * num_pid_n);
    int group_size_m = min(num_pid_n, GROUP_SIZE);
    int pid_m = group_id * GROUP_SIZE + within / (group_size_m);
    int pid_n = within % group_size_m;
"""
    else:
        s += """    int pid_m = (int)tgid.x;
    int pid_n = (int)tgid.y;
"""

    if double_buf:
        s += f"    threadgroup half sA0[{A_ELEMS}];\n"
        s += f"    threadgroup half sA1[{A_ELEMS}];\n"
        s += f"    threadgroup half sB0[{B_ELEMS}];\n"
        s += f"    threadgroup half sB1[{B_ELEMS}];\n"
        s += f"    threadgroup half* sA_cur = sA0;\n"
        s += f"    threadgroup half* sA_nxt = sA1;\n"
        s += f"    threadgroup half* sB_cur = sB0;\n"
        s += f"    threadgroup half* sB_nxt = sB1;\n"
    else:
        s += f"    threadgroup half sA_arr[{A_ELEMS}];\n"
        s += f"    threadgroup half sB_arr[{B_ELEMS}];\n"
        s += f"    threadgroup half* sA_cur = sA_arr;\n"
        s += f"    threadgroup half* sB_cur = sB_arr;\n"

    s += f"""
    // Map simdgroup to its TM*TN region in the output tile
    int sg_row = (int)sg_id / {SUPER_N};
    int sg_col = (int)sg_id % {SUPER_N};
    int base_br = sg_row * {TM};
    int base_bc = sg_col * {TN};

    // Register accumulators (NO shared memory needed for C!)
    simdgroup_matrix<float, 8, 8> acc[{TM}][{TN}];
    for (int i = 0; i < {TM}; i++)
        for (int j = 0; j < {TN}; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0);

"""

    if double_buf:
        # Prefetch first K-block
        s += gen_coop_load("sA_cur", "A", "stride_am", BK, A_ELEMS, A_LOADS, THREADS,
                          f"pid_m * {BM}", "0", vec4=vec4) + "\n"
        s += gen_coop_load("sB_cur", "B", "stride_bk", BN, B_ELEMS, B_LOADS, THREADS,
                          "0", f"pid_n * {BN}", vec4=vec4) + "\n"
        s += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
        s += f"    for (int kk = 0; kk < K_param; kk += {BK}) {{\n"
        # Prefetch next
        s += f"        if (kk + {BK} < K_param) {{\n"
        s += gen_coop_load("sA_nxt", "A", "stride_am", BK, A_ELEMS, A_LOADS, THREADS,
                          f"pid_m * {BM}", f"(kk + {BK})", indent=3, vec4=vec4) + "\n"
        s += gen_coop_load("sB_nxt", "B", "stride_bk", BN, B_ELEMS, B_LOADS, THREADS,
                          f"(kk + {BK})", f"pid_n * {BN}", indent=3, vec4=vec4) + "\n"
        s += "        }\n\n"
    else:
        s += f"    for (int kk = 0; kk < K_param; kk += {BK}) {{\n"
        s += gen_coop_load("sA_cur", "A", "stride_am", BK, A_ELEMS, A_LOADS, THREADS,
                          f"pid_m * {BM}", "kk", indent=2, vec4=vec4) + "\n"
        s += gen_coop_load("sB_cur", "B", "stride_bk", BN, B_ELEMS, B_LOADS, THREADS,
                          "kk", f"pid_n * {BN}", indent=2, vec4=vec4) + "\n"
        s += "        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"

    # Inner MMA with register tiling
    s += f"""        // MMA: load A/B once per K-block, reuse across TM*TN output blocks
        for (int kb = 0; kb < {K_BLOCKS}; kb++) {{
            simdgroup_matrix<half, 8, 8> sg_A[{TM}];
            simdgroup_matrix<half, 8, 8> sg_B[{TN}];
            for (int tm = 0; tm < {TM}; tm++)
                simdgroup_load(sg_A[tm], sA_cur + (base_br + tm) * {8 * BK} + kb * 8, {BK}ul);
            for (int tn = 0; tn < {TN}; tn++)
                simdgroup_load(sg_B[tn], sB_cur + kb * {8 * BN} + (base_bc + tn) * 8, {BN}ul);
            for (int tm = 0; tm < {TM}; tm++)
                for (int tn = 0; tn < {TN}; tn++)
                    simdgroup_multiply_accumulate(acc[tm][tn], sg_A[tm], sg_B[tn], acc[tm][tn]);
        }}
"""

    if double_buf:
        s += """        threadgroup_barrier(mem_flags::mem_threadgroup);
        { threadgroup half* _tmp;
          _tmp = sA_cur; sA_cur = sA_nxt; sA_nxt = _tmp;
          _tmp = sB_cur; sB_cur = sB_nxt; sB_nxt = _tmp; }
"""
    else:
        s += "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"

    s += "    }\n\n"

    # Store directly from registers to global memory
    s += f"""    // Store accumulators directly to device memory (no sC needed!)
    int c_row = pid_m * {BM} + base_br * 8;
    int c_col = pid_n * {BN} + base_bc * 8;
    for (int tm = 0; tm < {TM}; tm++)
        for (int tn = 0; tn < {TN}; tn++)
            simdgroup_store(acc[tm][tn],
                            C + (c_row + tm * 8) * stride_cm + (c_col + tn * 8),
                            (ulong)stride_cm);
}}
"""
    return s, "matmul_opt"


def generate_baseline_msl(BM, BN, BK, THREADS=1024):
    """Current codegen pattern: sC in shared memory, no register tiling."""
    NUM_SG = THREADS // 32
    BLOCKS_N = BN // 8
    TOTAL_BLOCKS = (BM // 8) * BLOCKS_N
    BLOCKS_PER_SG = max(1, (TOTAL_BLOCKS + NUM_SG - 1) // NUM_SG)

    A_ELEMS = BM * BK
    B_ELEMS = BK * BN
    C_ELEMS = BM * BN
    A_LOADS = (A_ELEMS + THREADS - 1) // THREADS
    B_LOADS = (B_ELEMS + THREADS - 1) // THREADS
    C_STORES = (C_ELEMS + THREADS - 1) // THREADS

    s = f"""#include <metal_stdlib>
using namespace metal;

kernel void matmul_base(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const int& K_param [[buffer(3)]],
    device const int& stride_am [[buffer(4)]],
    device const int& stride_bk [[buffer(5)]],
    device const int& stride_cm [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]])
{{
    int pid_m = (int)tgid.x;
    int pid_n = (int)tgid.y;

    threadgroup half sA[{A_ELEMS}];
    threadgroup half sB[{B_ELEMS}];
    threadgroup float sC[{C_ELEMS}];

    // Zero sC
    for (uint ci = 0; ci < {C_STORES}u; ci++) {{
        uint idx = tid + ci * {THREADS}u;
        if (idx < {C_ELEMS}u) sC[idx] = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Init accumulators from sC
    simdgroup_matrix<float, 8, 8> _acc[{BLOCKS_PER_SG}];
    for (uint bi = 0; bi < {BLOCKS_PER_SG}u; bi++) {{
        uint blk = sg_id * {BLOCKS_PER_SG}u + bi;
        if (blk < {TOTAL_BLOCKS}u) {{
            uint br = blk / {BLOCKS_N}u, bc = blk % {BLOCKS_N}u;
            simdgroup_load(_acc[bi], &sC[br * {8 * BN}u + bc * 8u], {BN}ul);
        }}
    }}

    for (int iv = 0; iv < K_param; iv += {BK}) {{
"""
    s += gen_coop_load("sA", "A", "stride_am", BK, A_ELEMS, A_LOADS, THREADS,
                      f"pid_m * {BM}", "iv", indent=2) + "\n"
    s += gen_coop_load("sB", "B", "stride_bk", BN, B_ELEMS, B_LOADS, THREADS,
                      "iv", f"pid_n * {BN}", indent=2) + "\n"
    s += f"""        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint bi = 0; bi < {BLOCKS_PER_SG}u; bi++) {{
            uint blk = sg_id * {BLOCKS_PER_SG}u + bi;
            if (blk < {TOTAL_BLOCKS}u) {{
                uint br = blk / {BLOCKS_N}u, bc = blk % {BLOCKS_N}u;
                for (uint k8 = 0; k8 < {BK}u; k8 += 8u) {{
                    simdgroup_matrix<half, 8, 8> sg_A, sg_B;
                    simdgroup_load(sg_A, &sA[br * {8 * BK}u + k8], {BK}ul);
                    simdgroup_load(sg_B, &sB[k8 * {BN}u + bc * 8u], {BN}ul);
                    simdgroup_multiply_accumulate(_acc[bi], sg_A, sg_B, _acc[bi]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Store accumulators to sC, then sC to global
    for (uint bi = 0; bi < {BLOCKS_PER_SG}u; bi++) {{
        uint blk = sg_id * {BLOCKS_PER_SG}u + bi;
        if (blk < {TOTAL_BLOCKS}u) {{
            uint br = blk / {BLOCKS_N}u, bc = blk % {BLOCKS_N}u;
            simdgroup_store(_acc[bi], &sC[br * {8 * BN}u + bc * 8u], {BN}ul);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ci = 0; ci < {C_STORES}u; ci++) {{
        uint idx = tid + ci * {THREADS}u;
        if (idx < {C_ELEMS}u) {{
            uint cr = idx / {BN}u, cc = idx % {BN}u;
            C[(pid_m * {BM} + (int)cr) * stride_cm + (pid_n * {BN} + (int)cc)] = sC[idx];
        }}
    }}
}}
"""
    return s, "matmul_base"


def bench(pipe, grid, threads, bufs, iters=20):
    for _ in range(3):
        dispatch(pipe, grid, threads, bufs)
    gpu_times = []
    for _ in range(iters):
        cb = dispatch(pipe, grid, threads, bufs)
        gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1000.0
        gpu_times.append(gpu_ms)
    gpu_times.sort()
    return gpu_times[len(gpu_times) // 2]  # median


def bench_mps(M, N, K, iters=20):
    import torch
    torch.mps.synchronize()
    a = torch.randn(M, K, device='mps', dtype=torch.float16)
    b = torch.randn(K, N, device='mps', dtype=torch.float16)
    for _ in range(5):
        c = a @ b
    torch.mps.synchronize()
    times = []
    for _ in range(iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_config(label, msl_gen_fn, M, N, K, BM, BN, mps_gf):
    """Compile, verify, benchmark a config."""
    try:
        msl, fn = msl_gen_fn()
        pipe = compile_kernel(msl, fn)
        grid = (M // BM, N // BN, 1)

        import random
        random.seed(42)
        a_data = [random.gauss(0, 0.3) for _ in range(M * K)]
        b_data = [random.gauss(0, 0.3) for _ in range(K * N)]
        a_buf = make_buffer(a_data, 'e')  # half
        b_buf = make_buffer(b_data, 'e')
        c_buf = make_zero_buffer(M * N, 'f')  # float32 output
        bufs = [a_buf, b_buf, c_buf, scalar_buf(K), scalar_buf(K),
                scalar_buf(N), scalar_buf(N)]

        dispatch(pipe, grid, 1024, bufs)
        c_vals = read_buffer(c_buf, M * N, 'f')
        expected = sum(a_data[k] * b_data[k * N] for k in range(K))
        err = abs(c_vals[0] - expected)
        ok = err < K * 5e-3

        mn = bench(pipe, grid, 1024, bufs)
        flops = 2.0 * M * N * K
        gflops = flops / (mn / 1000.0) / 1e9
        ratio = f"{gflops/mps_gf:.0%}" if mps_gf > 0 else "N/A"
        status = "OK" if ok else f"FAIL err={err:.4f}"
        print(f"  {label:<45s} {mn:>7.3f}ms  {gflops:>6.0f} GFLOP/s  {ratio:>5s} MPS  {status}")
        return gflops
    except Exception as e:
        print(f"  {label:<45s} ERROR: {e}")
        return 0


if __name__ == "__main__":
    print(f"Metal Device: {device.name()}")
    print()

    has_mps = False
    try:
        import torch
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except ImportError:
        pass

    sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    mps_results = {}
    if has_mps:
        print("=== MPS baseline (torch.matmul, fp16 in / fp32 out) ===")
        for M, N, K in sizes:
            mn = bench_mps(M, N, K)
            flops = 2.0 * M * N * K
            gf = flops / (mn / 1000.0) / 1e9
            mps_results[(M, N, K)] = gf
            print(f"  {M}x{N}x{K}: {mn:.3f}ms  {gf:.0f} GFLOP/s")
        print()

    # Configs to test
    for M, N, K in sizes:
        flops = 2.0 * M * N * K
        mps_gf = mps_results.get((M, N, K), 0)
        print(f"--- {M}x{N}x{K} (MPS: {mps_gf:.0f} GFLOP/s) ---")

        # Baseline: current codegen pattern (sC in shared memory)
        for BM, BN, BK in [(64, 64, 32)]:
            if M % BM or N % BN or K % BK:
                continue
            run_config(f"Baseline {BM}x{BN}x{BK} (sC in shmem)",
                      lambda: generate_baseline_msl(BM, BN, BK),
                      M, N, K, BM, BN, mps_gf)

        # Optimized: no sC, register tiling
        # (BM, BN, BK, TM, TN, double_buf, vec4, label_suffix)
        opt_cfgs = [
            # Best from round 1
            (128, 128, 32, 4, 2, False, False, ""),
            # Same but with vec4 loads
            (128, 128, 32, 4, 2, False, True, "+vec4"),
            # Double buffered 128x128x32 (exactly 32KB)
            (128, 128, 32, 4, 2, True, False, ""),
            (128, 128, 32, 4, 2, True, True, "+vec4"),
            # Larger tiles
            (256, 64, 32, 8, 1, False, False, ""),
            (256, 64, 32, 8, 1, False, True, "+vec4"),
            (64, 256, 32, 1, 8, False, False, ""),
            # Alternative TM/TN ratios
            (128, 128, 16, 2, 4, True, True, "+vec4"),
            (128, 64, 32, 4, 1, True, True, "+vec4"),
        ]
        for BM, BN, BK, TM, TN, db, v4, suffix in opt_cfgs:
            if M % BM or N % BN or K % BK:
                continue
            buf = "dbl" if db else "sgl"
            run_config(f"Opt {BM}x{BN}x{BK} TM={TM},TN={TN} ({buf}){suffix}",
                      lambda bm=BM, bn=BN, bk=BK, tm=TM, tn=TN, d=db, v=v4:
                          generate_optimized_msl(bm, bn, bk, tm, tn, double_buf=d, vec4=v),
                      M, N, K, BM, BN, mps_gf)

        print()
