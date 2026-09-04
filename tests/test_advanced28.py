#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 28.

Targets: final stress tests and edge cases:
- Large BLOCK_SIZE (1024) element-wise
- Multiple tl.reduce in one kernel (min, max, sum, count)
- Conditional store (only write if condition met)
- Complex index computation (2D -> 1D flattening)
- Strided load with non-unit stride
- Fused quantize (float -> int8 with scale/zero-point)
- Fused dequantize (int8 -> float with scale/zero-point)
- Multi-pass accumulator (loop with tile iter_arg)
- Row-wise variance (mean already known)
- Fused multiply-accumulate reduction
- Chain of 5 element-wise ops
- Very large reduction (BLOCK=1024)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Large BLOCK_SIZE (1024) element-wise
@triton.jit
def large_block_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Element-wise add with BLOCK=1024."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out = a + b
    tl.store(out_ptr + offs, out, mask=mask)


# 2. Multiple reductions in one kernel
@triton.jit
def multi_reduce_kernel(x_ptr, sum_ptr, min_ptr, max_ptr, count_ptr,
                          threshold, n, BLOCK: tl.constexpr):
    """Compute sum, min, max, and count(x > threshold) in one pass."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_min = tl.load(x_ptr + offs, mask=mask, other=float('inf'))
    x_max = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))

    total = tl.sum(x, axis=0)
    mn = tl.min(x_min, axis=0)
    mx = tl.max(x_max, axis=0)
    # Count elements above threshold
    above = tl.where(x > threshold, 1.0, 0.0)
    above_masked = tl.where(mask, above, 0.0)
    cnt = tl.sum(above_masked, axis=0)

    pid = tl.program_id(0)
    if pid == 0:
        tl.store(sum_ptr, total)
        tl.store(min_ptr, mn)
        tl.store(max_ptr, mx)
        tl.store(count_ptr, cnt)


# 3. Conditional store
@triton.jit
def cond_store_kernel(x_ptr, out_ptr, threshold, n, BLOCK: tl.constexpr):
    """Store only where x > threshold, leave out untouched otherwise."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Only store where x > threshold
    store_mask = mask & (x > threshold)
    tl.store(out_ptr + offs, x, mask=store_mask)


# 4. Complex index: row-major to column-major transpose via index math
@triton.jit
def transpose_idx_kernel(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    """Transpose via index arithmetic: out[j*M+i] = x[i*N+j]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = M * N
    mask = offs < total
    # offs represents flat index in output (column-major)
    # out[j*M+i] = x[i*N+j] where i = offs % M, j = offs // M
    j = offs // M
    i = offs % M
    src_idx = i * N + j
    x = tl.load(x_ptr + src_idx, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


# 5. Fused quantize: float -> int8 with scale/zero-point
@triton.jit
def quantize_kernel(x_ptr, out_ptr, scale, zero_point, n,
                      BLOCK: tl.constexpr):
    """Quantize: out = clamp(round(x / scale) + zero_point, 0, 255)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    q = x / scale + zero_point
    q = tl.minimum(tl.maximum(q + 0.5, 0.0), 255.0)  # round + clamp
    # Convert to int32 for storage (simulating int8)
    qi = q.to(tl.int32)
    tl.store(out_ptr + offs, qi, mask=mask)


# 6. Fused dequantize: int -> float with scale/zero-point
@triton.jit
def dequantize_kernel(x_ptr, out_ptr, scale, zero_point, n,
                        BLOCK: tl.constexpr):
    """Dequantize: out = (x - zero_point) * scale."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    out = (xf - zero_point) * scale
    tl.store(out_ptr + offs, out, mask=mask)


# 7. Row-wise variance with pre-computed mean
@triton.jit
def variance_kernel(x_ptr, mean_ptr, var_ptr, N, BLOCK_N: tl.constexpr):
    """var[row] = mean((x[row,:] - mean[row])^2)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row)
    diff = x - mean
    sq = diff * diff
    var = tl.sum(sq, axis=0) / N
    tl.store(var_ptr + row, var)


# 8. Chain of 5 element-wise ops
@triton.jit
def chain5_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = sin(abs(x * 2.0 + 1.0) - 0.5) * 3.0."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = x * 2.0 + 1.0
    y = tl.abs(y)
    y = y - 0.5
    y = tl.sin(y)
    out = y * 3.0
    tl.store(out_ptr + offs, out, mask=mask)


# 9. Very large reduction (BLOCK=1024)
@triton.jit
def large_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Sum with BLOCK=1024."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 10. Fused add + exp (softmax numerator pattern)
@triton.jit
def add_exp_kernel(x_ptr, bias_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    """out = exp(x + bias) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    out = tl.exp(x + b)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 11. Row-wise weighted sum (dot product between weights and row)
@triton.jit
def weighted_row_sum_kernel(x_ptr, w_ptr, out_ptr, N,
                              BLOCK_N: tl.constexpr):
    """out[row] = sum(x[row,:] * w[:])."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x * w, axis=0)
    tl.store(out_ptr + row, total)


# 12. Reciprocal sqrt (rsqrt) + multiply (normalization step)
@triton.jit
def rsqrt_mul_kernel(x_ptr, var_ptr, out_ptr, N, eps,
                       BLOCK_N: tl.constexpr):
    """out[row,:] = x[row,:] * rsqrt(var[row] + eps)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    var = tl.load(var_ptr + row)
    rstd = tl.rsqrt(var + eps)
    out = x * rstd
    tl.store(out_ptr + row * N + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_large_block():
    n = 4096
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    large_block_kernel[grid](a, b, out, n, BLOCK=1024)
    ref = a + b
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_multi_reduce():
    n = 256
    x = torch.randn(n, device='mps')
    sum_out = torch.zeros(1, device='mps')
    min_out = torch.zeros(1, device='mps')
    max_out = torch.zeros(1, device='mps')
    count_out = torch.zeros(1, device='mps')
    multi_reduce_kernel[(1,)](x, sum_out, min_out, max_out, count_out,
                                0.0, n, BLOCK=256)
    err = max(abs(sum_out.item() - x.sum().item()),
              abs(min_out.item() - x.min().item()),
              abs(max_out.item() - x.max().item()),
              abs(count_out.item() - (x > 0).sum().float().item()))
    return err < 0.5, err


def test_cond_store():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.full((n,), -999.0, device='mps')  # sentinel
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    cond_store_kernel[grid](x, out, 0.0, n, BLOCK=256)
    # Only positive values should be written
    ref = torch.where(x > 0, x, torch.tensor(-999.0, device='mps'))
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_transpose_idx():
    M, N = 16, 32
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(N * M, device='mps')
    grid = lambda meta: (triton.cdiv(M * N, meta['BLOCK']),)
    transpose_idx_kernel[grid](x.contiguous().view(-1), out, M, N, BLOCK=256)
    ref = x.T.contiguous().view(-1)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_quantize():
    n = 1024
    x = torch.randn(n, device='mps') * 2  # values in roughly [-6, 6]
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    scale = 0.05
    zp = 128.0
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    quantize_kernel[grid](x, out, scale, zp, n, BLOCK=256)
    ref = ((x / scale + zp + 0.5).clamp(0, 255)).int()
    err = (out - ref).abs().max().item()
    return err <= 1, float(err)  # rounding may cause ±1


def test_dequantize():
    n = 1024
    x = torch.randint(0, 255, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    scale = 0.05
    zp = 128.0
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    dequantize_kernel[grid](x, out, scale, zp, n, BLOCK=256)
    ref = (x.float() - zp) * scale
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_variance():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    mean = x.mean(dim=1)
    var_out = torch.zeros(M, device='mps')
    variance_kernel[(M,)](x, mean, var_out, N, BLOCK_N=128)
    ref = x.var(dim=1, correction=0)
    err = (var_out - ref).abs().max().item()
    return err < 1e-3, err


def test_chain5():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    chain5_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.sin(torch.abs(x * 2.0 + 1.0) - 0.5) * 3.0
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_large_sum():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    large_sum_kernel[(1,)](x, out, n, BLOCK=1024)
    ref = x.sum()
    err = abs(out.item() - ref.item())
    return err < 0.5, err


def test_add_exp():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps') * 0.5  # keep values moderate
    bias = torch.randn(N, device='mps') * 0.5
    out = torch.zeros(M, N, device='mps')
    add_exp_kernel[(M,)](x, bias, out, N, BLOCK=64)
    ref = torch.exp(x + bias)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_weighted_row_sum():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    w = torch.randn(N, device='mps')
    out = torch.zeros(M, device='mps')
    weighted_row_sum_kernel[(M,)](x, w, out, N, BLOCK_N=128)
    ref = (x * w).sum(dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-2, err


def test_rsqrt_mul():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    var = torch.rand(M, device='mps') + 0.01
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    rsqrt_mul_kernel[(M,)](x, var, out, N, eps, BLOCK_N=128)
    ref = x * torch.rsqrt(var + eps).unsqueeze(1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 28) ===\n")
    print("    (Stress tests, quantize/dequantize, large blocks)\n")

    tests = [
        ("Large Block (1024)", test_large_block),
        ("Multi-Reduce (4 stats)", test_multi_reduce),
        ("Conditional Store", test_cond_store),
        ("Transpose via Index", test_transpose_idx),
        ("Quantize (f32->i8)", test_quantize),
        ("Dequantize (i8->f32)", test_dequantize),
        ("Row Variance", test_variance),
        ("Chain of 5 Ops", test_chain5),
        ("Large Sum (1024)", test_large_sum),
        ("Add+Exp (per row)", test_add_exp),
        ("Weighted Row Sum", test_weighted_row_sum),
        ("Rsqrt+Mul (norm)", test_rsqrt_mul),
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
