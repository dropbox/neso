#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 11.

Targets: fp16 matmul, mixed-precision matmul (f16 input, f32 acc, f16 output),
tl.where on 2D tiles, multi-dim program_id with large grids,
pointer stride patterns, and quantized int8 matmul building blocks.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. fp16 matmul (pure fp16)
@triton.jit
def fp16_matmul_kernel(
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
        a = tl.load(a_ptrs).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c)


# 2. Fused scale + clamp on 2D tile
@triton.jit
def scale_clamp_2d_kernel(x_ptr, out_ptr, M, N, scale, lo, hi,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    y = x * scale
    y = tl.where(y < lo, lo, tl.where(y > hi, hi, y))
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


# 3. Row-wise argmax (find index of max per row)
@triton.jit
def row_argmax_kernel(x_ptr, out_ptr, M, N,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=-float('inf'))
    max_vals = tl.max(x, axis=1)  # [BLOCK_M]
    # Find argmax by comparing each element to the row max
    is_max = (x == max_vals[:, None])
    # Use the column index where is_max is true (take first match)
    col_indices = offs_n[None, :].to(tl.float32)
    # Where not max, use N (large value); take min to get first match
    idx = tl.where(is_max, col_indices, float(1e6))
    argmax = tl.min(idx, axis=1).to(tl.int32)
    tl.store(out_ptr + offs_m, argmax, mask=offs_m < M)


# 4. Dequantize int8 -> fp32 (building block for quantized matmul)
@triton.jit
def dequantize_kernel(x_int8_ptr, scale_ptr, zero_ptr, out_ptr, n,
                      BLOCK: tl.constexpr):
    """Dequantize: out = (x_int8 - zero) * scale."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_int8_ptr + offs, mask=mask).to(tl.float32)
    # Per-tensor scale and zero point
    scale = tl.load(scale_ptr)
    zero = tl.load(zero_ptr)
    out = (x - zero) * scale
    tl.store(out_ptr + offs, out, mask=mask)


# 5. Column-wise softmax (reduce along axis=0)
@triton.jit
def col_softmax_kernel(x_ptr, out_ptr, M, N,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Softmax along columns (axis=0) for each column."""
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=-float('inf'))
    # Column-wise max and subtract
    col_max = tl.max(x, axis=0)  # [BLOCK_N]
    x = x - col_max[None, :]
    # Exp and sum
    ex = tl.exp(x)
    col_sum = tl.sum(ex, axis=0)  # [BLOCK_N]
    out = ex / col_sum[None, :]
    # Mask out-of-bounds
    out = tl.where(mask, out, 0.0)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


# 6. Fused layer norm + scale + bias (transformer block)
@triton.jit
def layernorm_scale_bias_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr,
                                 row_stride, n_cols, eps,
                                 BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n_cols
    xn = xc / tl.sqrt(var + eps)
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    out = gamma * xn + beta
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 7. Gather + transform + scatter (index-based data shuffling)
@triton.jit
def gather_transform_scatter_kernel(src_ptr, idx_ptr, out_ptr, n,
                                     BLOCK: tl.constexpr):
    """Gather from src using indices, apply transform, write back."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    gathered = tl.load(src_ptr + idx, mask=mask)
    transformed = gathered * 2.0 + 1.0
    tl.store(out_ptr + offs, transformed, mask=mask)


# 8. 2D grid with non-square tiles
@triton.jit
def nonsquare_2d_kernel(x_ptr, out_ptr, M, N,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    out = x * 2.0 + 1.0
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_fp16_matmul():
    M, N, K = 32, 32, 32
    a = torch.randn(M, K, device='mps', dtype=torch.float16)
    b = torch.randn(K, N, device='mps', dtype=torch.float16)
    c = torch.zeros(M, N, device='mps', dtype=torch.float16)
    grid = (1, 1)
    fp16_matmul_kernel[grid](a, b, c, M, N, K,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    ref = (a.float() @ b.float()).half()
    err = (c.float() - ref.float()).abs().max().item()
    return err < 0.5, err  # fp16 has limited precision


def test_scale_clamp_2d():
    M, N = 64, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    scale_clamp_2d_kernel[grid](x, out, M, N, 2.0, -1.0, 1.0,
                                 BLOCK_M=32, BLOCK_N=32)
    ref = (x * 2.0).clamp(-1.0, 1.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_row_argmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps', dtype=torch.int32)
    row_argmax_kernel[(1,)](x, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = x.argmax(dim=1).int()
    err = (out - ref).abs().max().item()
    return err == 0, err


def test_dequantize():
    n = 1024
    x_int8 = torch.randint(-128, 127, (n,), device='mps', dtype=torch.int32)
    scale = torch.tensor([0.05], device='mps')
    zero = torch.tensor([3.0], device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    dequantize_kernel[grid](x_int8, scale, zero, out, n, BLOCK=256)
    ref = (x_int8.float() - 3.0) * 0.05
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_col_softmax():
    M, N = 16, 32
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    col_softmax_kernel[(1,)](x, out, M, N, BLOCK_M=16, BLOCK_N=32)
    ref = torch.softmax(x, dim=0)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_layernorm_scale_bias():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    gamma = torch.randn(N, device='mps')
    beta = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    layernorm_scale_bias_kernel[(M,)](x, gamma, beta, out, N, N, eps, BLOCK_SIZE=128)
    # Reference
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, correction=0)
    ref = gamma * (x - mean) / (var + eps).sqrt() + beta
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_gather_transform_scatter():
    n = 512
    src = torch.randn(n, device='mps')
    idx = torch.randint(0, n, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    gather_transform_scatter_kernel[grid](src, idx, out, n, BLOCK=256)
    ref = src[idx.long()] * 2.0 + 1.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_nonsquare_2d():
    M, N = 48, 96
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 32))
    nonsquare_2d_kernel[grid](x, out, M, N, BLOCK_M=16, BLOCK_N=32)
    ref = x * 2.0 + 1.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 11) ===\n")

    tests = [
        ("FP16 Matmul (32x32x32)", test_fp16_matmul),
        ("2D Scale+Clamp (tl.where 2D)", test_scale_clamp_2d),
        ("Row Argmax", test_row_argmax),
        ("Dequantize (int8->fp32)", test_dequantize),
        ("Column Softmax (axis=0)", test_col_softmax),
        ("LayerNorm+Scale+Bias", test_layernorm_scale_bias),
        ("Gather+Transform+Scatter", test_gather_transform_scatter),
        ("Non-Square 2D (16x32 tiles)", test_nonsquare_2d),
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
