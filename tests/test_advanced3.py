#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 3.

Targets: scf.for with non-dot body, nested loops, mixed 2D ops,
         advanced pointer arithmetic, dynamic indexing.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Prefix sum via scf.for (non-dot loop body)
@triton.jit
def block_prefix_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Naive parallel prefix sum within a block."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # tl.cumsum does the prefix sum
    y = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, y, mask=mask)


# 2. Absolute value (unary op via tl.where)
@triton.jit
def abs_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x >= 0, x, -x), mask=mask)


# 3. Reciprocal sqrt (1/sqrt(x)) with guard
@triton.jit
def rsqrt_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + offs, 1.0 / tl.sqrt(x), mask=mask)


# 4. Exponential moving average (two loads, fused multiply-add)
@triton.jit
def ema_kernel(x_ptr, prev_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):
    """out = alpha * x + (1 - alpha) * prev"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    prev = tl.load(prev_ptr + offs, mask=mask)
    out = alpha * x + (1.0 - alpha) * prev
    tl.store(out_ptr + offs, out, mask=mask)


# 5. Leaky ReLU (compound tl.where with negative slope)
@triton.jit
def leaky_relu_kernel(x_ptr, out_ptr, n, negative_slope, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.where(x >= 0, x, negative_slope * x)
    tl.store(out_ptr + offs, out, mask=mask)


# 6. Batch vector dot product (2D load, reduction per row)
@triton.jit
def batch_dot_kernel(x_ptr, y_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """For each row i in [0,M), compute out[i] = sum(x[i,:] * y[i,:])"""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    y = tl.load(y_ptrs, mask=mask, other=0.0)
    dots = tl.sum(x * y, axis=1)  # reduce over N dimension
    tl.store(out_ptr + offs_m, dots, mask=offs_m < M)


# 7. Softplus (log(1 + exp(x)) with numerical stability)
@triton.jit
def softplus_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Numerically stable: for large x, softplus(x) ≈ x
    out = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tl.store(out_ptr + offs, out, mask=mask)


# 8. L2 norm of rows (2D load, square, sum, sqrt)
@triton.jit
def l2_norm_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    sq_sum = tl.sum(x * x, axis=1)
    norms = tl.sqrt(sq_sum)
    tl.store(out_ptr + offs_m, norms, mask=offs_m < M)


# 9. Sigmoid activation
@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.sigmoid(x)
    tl.store(out_ptr + offs, out, mask=mask)


# 10. Multi-head attention score computation (batch matmul slice)
@triton.jit
def attention_score_kernel(
    q_ptr, k_ptr, out_ptr,
    seq_len, d_model,
    stride_q_seq, stride_k_seq,
    stride_out_seq,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Compute QK^T / sqrt(d) for a single attention head."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Q[m, :] @ K[n, :]^T → accumulate over d
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d in range(0, d_model, BLOCK_D):
        q = tl.load(q_ptr + offs_m[:, None] * stride_q_seq + (offs_d[None, :] + d))
        k = tl.load(k_ptr + offs_n[:, None] * stride_k_seq + (offs_d[None, :] + d))
        acc += tl.dot(q, tl.trans(k))

    acc = acc * scale
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_seq + offs_n[None, :]
    tl.store(out_ptrs, acc)


# 11. Vector distance (squared L2 between two vectors)
@triton.jit
def sq_distance_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    diff = x - y
    sq_dist = tl.sum(diff * diff, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, sq_dist)


# 12. Tanh activation
@triton.jit
def tanh_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    e2x = tl.exp(2.0 * x)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_prefix_sum():
    n = 256
    x = torch.ones(n, device='mps')
    out = torch.zeros(n, device='mps')
    block_prefix_sum_kernel[(1,)](x, out, n, BLOCK=256)
    ref = torch.arange(1, n + 1, device='mps', dtype=torch.float32)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_abs():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    abs_kernel[grid](x, out, n, BLOCK=256)
    ref = x.abs()
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_rsqrt():
    n = 2048
    x = torch.rand(n, device='mps') + 0.01  # positive values
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    rsqrt_kernel[grid](x, out, n, BLOCK=256)
    ref = 1.0 / torch.sqrt(x)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_ema():
    n = 2048
    x = torch.randn(n, device='mps')
    prev = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    alpha = 0.1
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    ema_kernel[grid](x, prev, out, alpha, n, BLOCK=256)
    ref = alpha * x + (1.0 - alpha) * prev
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_leaky_relu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    slope = 0.01
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    leaky_relu_kernel[grid](x, out, n, slope, BLOCK=256)
    ref = torch.where(x >= 0, x, slope * x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_batch_dot():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    y = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    batch_dot_kernel[(triton.cdiv(M, 32),)](x, y, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = (x * y).sum(dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_softplus():
    n = 2048
    x = torch.randn(n, device='mps') * 5
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    softplus_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.nn.functional.softplus(x)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_l2_norm():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    l2_norm_kernel[(triton.cdiv(M, 32),)](x, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = torch.norm(x, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_sigmoid():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    sigmoid_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.sigmoid(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_attention_score():
    seq_len = 32
    d_model = 32
    BM, BN, BD = 32, 32, 32
    q = torch.randn(seq_len, d_model, device='mps')
    k = torch.randn(seq_len, d_model, device='mps')
    out = torch.zeros(seq_len, seq_len, device='mps')
    scale = 1.0 / (d_model ** 0.5)
    grid = (triton.cdiv(seq_len, BM), triton.cdiv(seq_len, BN))
    attention_score_kernel[grid](
        q, k, out, seq_len, d_model,
        q.stride(0), k.stride(0), out.stride(0),
        scale,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BD,
    )
    ref = (q @ k.T) * scale
    err = (out - ref).abs().max().item()
    tol = 1e-3 * (d_model ** 0.5)
    return err < tol, err


def test_sq_distance():
    n = 256
    x = torch.randn(n, device='mps')
    y = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    sq_distance_kernel[(1,)](x, y, out, n, BLOCK=256)
    ref = ((x - y) ** 2).sum()
    err = abs(out.item() - ref.item())
    return err < 1e-2, err


def test_tanh():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    tanh_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.tanh(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 3) ===\n")

    tests = [
        ("Prefix Sum (cumsum)", test_prefix_sum),
        ("Absolute Value", test_abs),
        ("Reciprocal Sqrt", test_rsqrt),
        ("EMA (fused mul-add)", test_ema),
        ("Leaky ReLU", test_leaky_relu),
        ("Batch Dot Product", test_batch_dot),
        ("Softplus (log1+exp)", test_softplus),
        ("L2 Norm (rows)", test_l2_norm),
        ("Sigmoid", test_sigmoid),
        ("Attention Score (QK^T)", test_attention_score),
        ("Squared Distance", test_sq_distance),
        ("Tanh", test_tanh),
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
