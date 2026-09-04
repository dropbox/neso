#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 14.

Targets: remaining op coverage - fma, clamp, philox RNG, tl.full,
tl.zeros_like, constexpr arithmetic, multiple return values,
and edge cases around masking.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Fused multiply-add (tl.fma or manual a*b+c)
@triton.jit
def fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = a * b + c (fused)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    out = a * b + c
    tl.store(out_ptr + offs, out, mask=mask)


# 2. Clamp (tl.where chain for min/max bounding)
@triton.jit
def clamp_kernel(x_ptr, out_ptr, lo, hi, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.minimum(tl.maximum(x, lo), hi)
    tl.store(out_ptr + offs, out, mask=mask)


# 3. Constexpr arithmetic in kernel
@triton.jit
def constexpr_kernel(x_ptr, out_ptr, n,
                     BLOCK: tl.constexpr, SCALE: tl.constexpr, OFFSET: tl.constexpr):
    """Test constexpr values in arithmetic."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * SCALE + OFFSET
    tl.store(out_ptr + offs, out, mask=mask)


# 4. Multiple return values via atomics
@triton.jit
def minmax_atomic_kernel(x_ptr, min_ptr, max_ptr, n, BLOCK: tl.constexpr):
    """Find global min and max using atomics."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    block_max = tl.max(x, axis=0)
    block_min = tl.min(x, axis=0)
    # Use atomic for max (positive values only for IEEE-754 trick)
    tl.atomic_max(max_ptr, block_max)
    tl.atomic_min(min_ptr, block_min)


# 5. Tiled matrix addition (2D)
@triton.jit
def matrix_add_kernel(a_ptr, b_ptr, c_ptr, M, N,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    b = tl.load(b_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], a + b, mask=mask)


# 6. Weighted average (sum of products / sum of weights)
@triton.jit
def weighted_avg_kernel(val_ptr, weight_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(val_ptr + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    weighted_sum = tl.sum(v * w, axis=0)
    weight_sum = tl.sum(w, axis=0)
    result = weighted_sum / weight_sum
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, result)


# 7. Running variance (Welford's online algorithm)
@triton.jit
def welford_kernel(x_ptr, mean_ptr, var_ptr, n, BLOCK: tl.constexpr):
    """Compute mean and variance using Welford's algorithm."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Simple two-pass: mean then variance
    mean_val = tl.sum(x, axis=0) / n
    diff = x - mean_val
    var_val = tl.sum(diff * diff, axis=0) / n
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(mean_ptr, mean_val)
        tl.store(var_ptr, var_val)


# 8. Tensor comparison and count
@triton.jit
def count_nonzero_kernel(x_ptr, count_ptr, n, BLOCK: tl.constexpr):
    """Count nonzero elements."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    nonzero = (x != 0.0).to(tl.float32)
    count = tl.sum(nonzero, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(count_ptr, count)


# 9. Exclusive prefix sum (offset by 1)
@triton.jit
def exclusive_cumsum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Exclusive prefix sum: out[i] = sum(x[0:i])."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Inclusive cumsum then shift right
    inclusive = tl.cumsum(x, axis=0)
    # Exclusive: shift by subtracting current element
    exclusive = inclusive - x
    tl.store(out_ptr + offs, exclusive, mask=mask)


# 10. Two-pass norm: compute norm then normalize
@triton.jit
def l2_normalize_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """L2 normalize: out = x / ||x||_2."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    norm_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(norm_sq)
    out = x / norm
    tl.store(out_ptr + offs, out, mask=mask)


# 11. Masked scatter add (accumulate into specific positions)
@triton.jit
def scatter_weighted_kernel(src_ptr, idx_ptr, weight_ptr, out_ptr, n,
                            BLOCK: tl.constexpr):
    """out[idx[i]] += src[i] * weight[i] (scatter weighted add)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    src = tl.load(src_ptr + offs, mask=mask)
    idx = tl.load(idx_ptr + offs, mask=mask)
    w = tl.load(weight_ptr + offs, mask=mask)
    val = src * w
    tl.atomic_add(out_ptr + idx, val, mask=mask)


# 12. Batch softmax with temperature
@triton.jit
def softmax_temp_kernel(x_ptr, out_ptr, row_stride, n_cols, temp,
                        BLOCK_SIZE: tl.constexpr):
    """Softmax with temperature: softmax(x / temp)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    x = x / temp
    max_val = tl.max(x, axis=0)
    x = x - max_val
    ex = tl.exp(x)
    sum_ex = tl.sum(ex, axis=0)
    out = ex / sum_ex
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_fma():
    n = 2048
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    c = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fma_kernel[grid](a, b, c, out, n, BLOCK=256)
    ref = a * b + c
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_clamp():
    n = 2048
    x = torch.randn(n, device='mps') * 5
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    clamp_kernel[grid](x, out, -2.0, 2.0, n, BLOCK=256)
    ref = x.clamp(-2.0, 2.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_constexpr():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    constexpr_kernel[grid](x, out, n, BLOCK=256, SCALE=3, OFFSET=7)
    ref = x * 3 + 7
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_minmax_atomic():
    n = 4096
    x = torch.rand(n, device='mps') + 0.1  # positive for atomic trick
    min_out = torch.full((1,), float('inf'), device='mps')
    max_out = torch.zeros(1, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    minmax_atomic_kernel[grid](x, min_out, max_out, n, BLOCK=256)
    ref_min = x.min()
    ref_max = x.max()
    err = max(abs(min_out.item() - ref_min.item()),
              abs(max_out.item() - ref_max.item()))
    return err < 1e-5, err


def test_matrix_add():
    M, N = 64, 128
    a = torch.randn(M, N, device='mps')
    b = torch.randn(M, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    matrix_add_kernel[grid](a, b, c, M, N, BLOCK_M=32, BLOCK_N=32)
    ref = a + b
    err = (c - ref).abs().max().item()
    return err < 1e-5, err


def test_weighted_avg():
    n = 256
    vals = torch.randn(n, device='mps')
    weights = torch.rand(n, device='mps') + 0.1
    out = torch.zeros(1, device='mps')
    weighted_avg_kernel[(1,)](vals, weights, out, n, BLOCK=256)
    ref = (vals * weights).sum() / weights.sum()
    err = abs(out.item() - ref.item())
    return err < 1e-3, err


def test_welford():
    n = 256
    x = torch.randn(n, device='mps')
    mean_out = torch.zeros(1, device='mps')
    var_out = torch.zeros(1, device='mps')
    welford_kernel[(1,)](x, mean_out, var_out, n, BLOCK=256)
    ref_mean = x.mean()
    ref_var = x.var(correction=0)
    err = max(abs(mean_out.item() - ref_mean.item()),
              abs(var_out.item() - ref_var.item()))
    return err < 1e-3, err


def test_count_nonzero():
    n = 256
    x = torch.randn(n, device='mps')
    x[x.abs() < 0.5] = 0  # zero out some elements
    count = torch.zeros(1, device='mps')
    count_nonzero_kernel[(1,)](x, count, n, BLOCK=256)
    ref = (x != 0).sum().float()
    err = abs(count.item() - ref.item())
    return err < 1, err


def test_exclusive_cumsum():
    n = 128
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    exclusive_cumsum_kernel[(1,)](x, out, n, BLOCK=128)
    ref = x.cumsum(0) - x  # exclusive = inclusive - current
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_l2_normalize():
    n = 256
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    l2_normalize_kernel[(1,)](x, out, n, BLOCK=256)
    ref = x / x.norm()
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_scatter_weighted():
    n = 512
    n_bins = 32
    src = torch.randn(n, device='mps')
    idx = torch.randint(0, n_bins, (n,), device='mps', dtype=torch.int32)
    weights = torch.rand(n, device='mps')
    out = torch.zeros(n_bins, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    scatter_weighted_kernel[grid](src, idx, weights, out, n, BLOCK=256)
    # Reference
    ref = torch.zeros(n_bins, device='mps')
    for i in range(n):
        ref[idx[i].item()] += src[i].item() * weights[i].item()
    err = (out - ref).abs().max().item()
    return err < 1e-2, err  # atomics may have ordering issues


def test_softmax_temp():
    M, N = 16, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    temp = 2.0
    softmax_temp_kernel[(M,)](x, out, N, N, temp, BLOCK_SIZE=64)
    ref = torch.softmax(x / temp, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 14) ===\n")

    tests = [
        ("FMA (a*b+c)", test_fma),
        ("Clamp (min/max)", test_clamp),
        ("Constexpr Arithmetic", test_constexpr),
        ("Atomic Min+Max", test_minmax_atomic),
        ("2D Matrix Add", test_matrix_add),
        ("Weighted Average", test_weighted_avg),
        ("Welford (mean+var)", test_welford),
        ("Count Nonzero", test_count_nonzero),
        ("Exclusive CumSum", test_exclusive_cumsum),
        ("L2 Normalize", test_l2_normalize),
        ("Scatter Weighted Add", test_scatter_weighted),
        ("Softmax+Temperature", test_softmax_temp),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            ok, err = fn()
            status = "PASS" if ok else "FAIL"
            print(f"  {name}: {status} (max_err={err:.2e})")
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  {name}: ERROR ({e})")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
