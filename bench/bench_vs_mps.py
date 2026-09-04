#!/usr/bin/env python3
"""Benchmark: Neso backend vs PyTorch MPS.

Uses Metal GPU timestamps for Triton (kernel-only time, no copy overhead).
Uses wall-clock + synchronize for MPS (zero-copy).

NOTE: Current Neso driver copies data CPU<->GPU each launch.
      GPU timestamps isolate actual kernel execution time.
"""
import os
import sys
import io
import re
import time
import torch
import triton
import triton.language as tl
from contextlib import redirect_stdout

os.environ['NESO_PROFILE'] = '1'
torch.mps.synchronize()

# ============================================================
# Triton kernels
# ============================================================

@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask), mask=mask)

@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.sigmoid(tl.load(x_ptr + offs, mask=mask)), mask=mask)

@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    k = 0.7978845608028654
    inner = k * (x + 0.044715 * x * x * x)
    tl.store(out_ptr + offs, 0.5 * x * (1.0 + 2.0 * tl.sigmoid(2.0 * inner) - 1.0), mask=mask)

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-1e9)
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    tl.store(out_ptr + row * n_cols + offs, ex / tl.sum(ex, axis=0), mask=mask)

@triton.jit
def layernorm_kernel(x_ptr, g_ptr, b_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    xn = diff / tl.sqrt(tl.sum(diff * diff, axis=0) / N + eps)
    tl.store(out_ptr + row * N + offs,
             tl.load(g_ptr + offs, mask=mask, other=1.0) * xn +
             tl.load(b_ptr + offs, mask=mask, other=0.0), mask=mask)

@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + row * N + offs,
             x * tl.rsqrt(tl.sum(x * x, axis=0) / N + eps) * w, mask=mask)

@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(out_ptr + row * N + offs,
             tl.maximum(tl.load(x_ptr + row * N + offs, mask=mask, other=0.0) +
                        tl.load(bias_ptr + offs, mask=mask, other=0.0), 0.0), mask=mask)

@triton.jit
def silu_gate_kernel(x_ptr, g_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x) * tl.load(g_ptr + offs, mask=mask), mask=mask)

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    A = a_ptr + rm[:, None] * stride_am + rk[None, :]
    B = b_ptr + rk[:, None] * stride_bk + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(B, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        A += BK
        B += BK * stride_bk
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


# ============================================================
# Helpers
# ============================================================

def triton_gpu_us(fn):
    """Single call, capture GPU time from [profile] output."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    m = re.search(r'gpu=([\d.]+)ms', buf.getvalue())
    return float(m.group(1)) * 1000 if m else None

def triton_bench(fn, n=5):
    """Median of n GPU-time measurements."""
    times = []
    for _ in range(n):
        t = triton_gpu_us(fn)
        if t is not None:
            times.append(t)
    if not times:
        return 0
    times.sort()
    return times[len(times) // 2]

def mps_bench(fn, n=20):
    """Median wall-clock time for MPS ops."""
    for _ in range(3):
        fn()
    torch.mps.synchronize()
    times = []
    for _ in range(n):
        torch.mps.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1000)
    times.sort()
    return times[len(times) // 2]


# ============================================================
# Run benchmarks
# ============================================================

if __name__ == "__main__":
    results = []

    # --- Element-wise ---
    N = 65536  # 64K to keep copies fast
    a = torch.randn(N, device='mps'); b = torch.randn(N, device='mps')
    out = torch.zeros(N, device='mps')
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)

    print("Compiling kernels...", file=sys.stderr, flush=True)

    t = triton_bench(lambda: add_kernel[grid](a, b, out, N, BLOCK=256))
    m = mps_bench(lambda: torch.add(a, b, out=out))
    results.append(("Add (64K f32)", t, m, ""))

    t = triton_bench(lambda: sigmoid_kernel[grid](a, out, N, BLOCK=256))
    m = mps_bench(lambda: torch.sigmoid(a, out=out))
    results.append(("Sigmoid (64K)", t, m, ""))

    t = triton_bench(lambda: gelu_kernel[grid](a, out, N, BLOCK=256))
    m = mps_bench(lambda: torch.nn.functional.gelu(a, approximate='tanh'))
    results.append(("GELU (64K)", t, m, ""))

    results.append(None)  # separator

    # --- Row-wise ops ---
    M, NC = 256, 128
    x = torch.randn(M, NC, device='mps')
    gamma = torch.ones(NC, device='mps'); beta = torch.zeros(NC, device='mps')
    o = torch.zeros(M, NC, device='mps')

    t = triton_bench(lambda: softmax_kernel[(M,)](x, o, NC, BLOCK=128))
    m = mps_bench(lambda: torch.softmax(x, dim=1))
    results.append((f"Softmax ({M}x{NC})", t, m, ""))

    t = triton_bench(lambda: layernorm_kernel[(M,)](x, gamma, beta, o, NC, 1e-5, BLOCK=128))
    m = mps_bench(lambda: torch.nn.functional.layer_norm(x, [NC], gamma, beta))
    results.append((f"LayerNorm ({M}x{NC})", t, m, ""))

    w = torch.ones(NC, device='mps')
    t = triton_bench(lambda: rmsnorm_kernel[(M,)](x, w, o, NC, 1e-5, BLOCK=128))
    m = mps_bench(lambda: x * torch.rsqrt((x*x).mean(dim=1, keepdim=True) + 1e-5) * w)
    results.append((f"RMSNorm ({M}x{NC})", t, m, ""))

    results.append(None)

    # --- Fused ops ---
    bias = torch.randn(NC, device='mps')
    t = triton_bench(lambda: bias_relu_kernel[(M,)](x, bias, o, NC, BLOCK=128))
    m = mps_bench(lambda: torch.relu(x + bias))
    results.append((f"Bias+ReLU ({M}x{NC})", t, m, ""))

    g = torch.randn(N, device='mps')
    t = triton_bench(lambda: silu_gate_kernel[grid](a, g, out, N, BLOCK=256))
    m = mps_bench(lambda: torch.nn.functional.silu(a) * g)
    results.append(("SiLU*Gate (64K)", t, m, ""))

    results.append(None)

    # --- Matmul ---
    for sz, bm, bn, bk in [(32,32,32,32), (64,64,64,32), (128,64,64,32)]:
        A = torch.randn(sz, sz, device='mps'); B = torch.randn(sz, sz, device='mps')
        C = torch.zeros(sz, sz, device='mps')
        mg = (triton.cdiv(sz, bm), triton.cdiv(sz, bn))
        t = triton_bench(lambda: matmul_kernel[mg](A, B, C, sz, sz, sz, sz, sz, sz,
                                                      BM=bm, BN=bn, BK=bk))
        m = mps_bench(lambda: torch.mm(A, B, out=C))
        flops = 2 * sz * sz * sz
        tf_t = flops / t / 1e6 if t > 0 else 0
        tf_m = flops / m / 1e6 if m > 0 else 0
        results.append((f"Matmul {sz}x{sz}", t, m, f"  [{tf_t:.1f} vs {tf_m:.1f} TF/s]"))

    # --- Print ---
    print(file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print("  Neso vs PyTorch MPS — Kernel Execution Time", file=sys.stderr)
    print("  Triton: Metal GPU timestamps | MPS: wall-clock + sync", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(f"  {'Kernel':<25} {'Triton':>10} {'MPS':>10}  {'Triton vs MPS':>16}  Extra", file=sys.stderr)
    print(f"  {'-'*25} {'-'*10} {'-'*10}  {'-'*16}  -----", file=sys.stderr)

    for r in results:
        if r is None:
            print(file=sys.stderr)
            continue
        name, tt, tm, extra = r
        if tt == 0:
            print(f"  {name:<25}  {'N/A':>10} {tm:>10.1f}us", file=sys.stderr)
            continue
        ratio = tm / tt if tt > 0 else 0
        if ratio >= 1:
            sp = f"{ratio:5.2f}x faster"
        else:
            sp = f"{1/ratio:5.2f}x slower"
        def ft(v):
            return f"{v:.1f}us" if v < 1000 else f"{v/1000:.2f}ms"
        print(f"  {name:<25} {ft(tt):>10} {ft(tm):>10}  {sp:>16}{extra}", file=sys.stderr)

    print(file=sys.stderr)
    print("  Note: Triton=GPU-only time. MPS includes dispatch overhead.", file=sys.stderr)
    print("  Current driver adds CPU<->GPU copy per launch (not measured).", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
