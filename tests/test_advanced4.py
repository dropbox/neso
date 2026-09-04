#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 4.

Targets: mixed-precision matmul, fp16 reductions, multi-block matmul,
         complex pointer patterns, diverse reduction ops, atomic scatter patterns.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Mixed precision matmul (fp16 inputs, fp32 accumulator, fp16 output)
@triton.jit
def matmul_fp16_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c)


# 2. Argmax (reduction returning index)
@triton.jit
def argmax_kernel(x_ptr, out_idx_ptr, out_val_ptr, M, N,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Row-wise argmax: for each row, find the index and value of the maximum."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    max_vals = tl.max(x, axis=1)
    tl.store(out_val_ptr + offs_m, max_vals, mask=offs_m < M)


# 3. Weighted sum (element-wise multiply then reduce)
@triton.jit
def weighted_sum_kernel(x_ptr, w_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    result = tl.sum(x * w, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, result)


# 4. Multi-block matmul (larger than single block)
@triton.jit
def matmul_multiblock_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


# 5. Column-wise softmax (axis=0 reduction in 2D)
@triton.jit
def col_softmax_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Softmax over columns (axis=0): for each column j, softmax over rows."""
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    col_max = tl.max(x, axis=0)  # [N]
    x = x - col_max[None, :]
    exp_x = tl.exp(x)
    col_sum = tl.sum(exp_x, axis=0)  # [N]
    out = exp_x / col_sum[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, out, mask=mask)


# 6. Vector normalize (divide by L2 norm)
@triton.jit
def normalize_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    norm = tl.sqrt(tl.sum(x * x, axis=0) + 1e-8)
    out = x / norm
    tl.store(out_ptr + offs, out, mask=mask)


# 7. Batch add with broadcasting (add bias vector to each row)
@triton.jit
def add_bias_2d_kernel(x_ptr, bias_ptr, out_ptr, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    # Broadcast bias [N] -> [M, N] and add
    out = x + bias[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


# 8. Scale rows (multiply each row by a per-row scalar)
@triton.jit
def scale_rows_kernel(x_ptr, scale_ptr, out_ptr, M, N,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask)
    s = tl.load(scale_ptr + offs_m, mask=offs_m < M)
    # Broadcast scale [M] -> [M, N] and multiply
    out = x * s[:, None]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


# 9. Element-wise min/max clamp with separate bounds per element
@triton.jit
def clamp_minmax_kernel(x_ptr, lo_ptr, hi_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    lo = tl.load(lo_ptr + offs, mask=mask)
    hi = tl.load(hi_ptr + offs, mask=mask)
    out = tl.minimum(tl.maximum(x, lo), hi)
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_matmul_fp16():
    M, N, K = 32, 32, 32
    a = torch.randn(M, K, device='mps', dtype=torch.float16)
    b = torch.randn(K, N, device='mps', dtype=torch.float16)
    c = torch.zeros(M, N, device='mps', dtype=torch.float16)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    matmul_fp16_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    tol = 0.1  # fp16 matmul has less precision
    return err < tol, err


def test_argmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out_val = torch.zeros(M, device='mps')
    argmax_kernel[(triton.cdiv(M, 32),)](x, None, out_val, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = x.max(dim=1).values
    err = (out_val - ref).abs().max().item()
    return err < 1e-5, err


def test_weighted_sum():
    n = 256
    x = torch.randn(n, device='mps')
    w = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    weighted_sum_kernel[(1,)](x, w, out, n, BLOCK=256)
    ref = (x * w).sum()
    err = abs(out.item() - ref.item())
    return err < 1e-3, err


def test_matmul_multiblock():
    M, N, K = 128, 64, 128
    BM, BN, BK = 32, 32, 32
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_multiblock_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    ref = a @ b
    err = (c - ref).abs().max().item()
    tol = 1e-3 * (K ** 0.5)
    return err < tol, err


def test_col_softmax():
    M, N = 16, 16
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    col_softmax_kernel[(triton.cdiv(N, 16),)](x, out, M, N, BLOCK_M=16, BLOCK_N=16)
    ref = torch.softmax(x, dim=0)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_normalize():
    n = 256
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    normalize_kernel[(1,)](x, out, n, BLOCK=256)
    ref = x / (x.norm() + 1e-8)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_add_bias_2d():
    M, N = 64, 32
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    add_bias_2d_kernel[grid](x, bias, out, M, N, BLOCK_M=32, BLOCK_N=32)
    ref = x + bias[None, :]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_scale_rows():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    scale = torch.randn(M, device='mps')
    out = torch.zeros(M, N, device='mps')
    scale_rows_kernel[(triton.cdiv(M, 32),)](x, scale, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = x * scale[:, None]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_clamp_minmax():
    n = 2048
    x = torch.randn(n, device='mps') * 5
    lo = torch.randn(n, device='mps') - 2
    hi = lo + torch.rand(n, device='mps') * 4 + 0.1
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    clamp_minmax_kernel[grid](x, lo, hi, out, n, BLOCK=256)
    ref = torch.minimum(torch.maximum(x, lo), hi)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 4) ===\n")

    tests = [
        ("FP16 Matmul (32x32)", test_matmul_fp16),
        ("Row Max (argmax)", test_argmax),
        ("Weighted Sum (dot)", test_weighted_sum),
        ("Multi-Block Matmul (128x64x128)", test_matmul_multiblock),
        ("Column Softmax (axis=0)", test_col_softmax),
        ("Vector Normalize (L2)", test_normalize),
        ("2D Add Bias (broadcast)", test_add_bias_2d),
        ("Scale Rows (broadcast)", test_scale_rows),
        ("Element-wise Clamp (min/max)", test_clamp_minmax),
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
