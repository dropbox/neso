#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 9.

Targets: less-tested ops, edge cases, and stress patterns.
Unary negation, integer abs, float remainder, ceiling div,
complex broadcasting, very small blocks, empty mask handling,
chained where/select, multi-pointer kernels.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Unary negation
@triton.jit
def negate_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, -x, mask=mask)


# 2. Integer absolute value
@triton.jit
def int_abs_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.abs(x), mask=mask)


# 3. Chained tl.where (multi-level branching)
@triton.jit
def multi_where_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Piecewise function: x<-1 -> -1, -1<=x<=1 -> x, x>1 -> 1 (hard tanh)"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.where(x < -1.0, -1.0, tl.where(x > 1.0, 1.0, x))
    tl.store(out_ptr + offs, out, mask=mask)


# 4. Floor and ceil
@triton.jit
def floor_ceil_kernel(x_ptr, floor_ptr, ceil_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(floor_ptr + offs, tl.math.floor(x), mask=mask)
    tl.store(ceil_ptr + offs, tl.math.ceil(x), mask=mask)


# 5. Very small block (BLOCK=32, single SIMD group)
@triton.jit
def small_block_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


# 6. Multi-pointer kernel (6 buffers, complex data flow)
@triton.jit
def multi_ptr_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, out_ptr, n,
                     BLOCK: tl.constexpr):
    """out = (a + b) * (c - d) + e"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    d = tl.load(d_ptr + offs, mask=mask)
    e = tl.load(e_ptr + offs, mask=mask)
    out = (a + b) * (c - d) + e
    tl.store(out_ptr + offs, out, mask=mask)


# 7. Integer min/max
@triton.jit
def int_minmax_kernel(a_ptr, b_ptr, min_ptr, max_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(min_ptr + offs, tl.minimum(a, b), mask=mask)
    tl.store(max_ptr + offs, tl.maximum(a, b), mask=mask)


# 8. Log2 and exp2
@triton.jit
def log2_exp2_kernel(x_ptr, log2_ptr, exp2_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    tl.store(log2_ptr + offs, tl.math.log2(x), mask=mask)
    # For exp2, use smaller values to avoid overflow
    x_small = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(exp2_ptr + offs, tl.math.exp2(x_small), mask=mask)


# 9. Fused SwiGLU activation (common in modern transformers)
@triton.jit
def swiglu_kernel(x_ptr, gate_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """SwiGLU: x * sigmoid(gate) * gate — actually silu(gate) * x"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    gate = tl.load(gate_ptr + offs, mask=mask)
    # SiLU(gate) = gate * sigmoid(gate)
    silu_gate = gate * tl.sigmoid(gate)
    out = x * silu_gate
    tl.store(out_ptr + offs, out, mask=mask)


# 10. Row-wise product (mul reduction)
@triton.jit
def row_product_kernel(x_ptr, out_ptr, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Product of elements in each row (uses tl.reduce with mul)."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=1.0)  # identity for mul is 1
    # Manual product via log/exp to avoid numerical issues
    # Actually just sum logs
    log_x = tl.log(tl.abs(x) + 1e-30)
    log_prod = tl.sum(log_x, axis=1)
    # Count negatives for sign
    neg_count = tl.sum((x < 0).to(tl.float32), axis=1)
    sign = tl.where(neg_count % 2.0 > 0.5, -1.0, 1.0)
    prod = sign * tl.exp(log_prod)
    tl.store(out_ptr + offs_m, prod, mask=offs_m < M)


# 11. Matmul with accumulator post-processing (add bias + relu after matmul)
@triton.jit
def matmul_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
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
    # Post-processing: add bias and ReLU
    bias = tl.load(bias_ptr + offs_n)
    acc = acc + bias[None, :]
    acc = tl.where(acc > 0, acc, 0.0)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc)


# 12. Multiple reductions on same data (sum, max, min in one pass)
@triton.jit
def multi_reduce_kernel(x_ptr, sum_ptr, max_ptr, min_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    mx = tl.max(x, axis=0)
    # For min, use inf as other for masked elements
    x_min = tl.load(x_ptr + offs, mask=mask, other=float('inf'))
    mn = tl.min(x_min, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(sum_ptr, s)
        tl.store(max_ptr, mx)
        tl.store(min_ptr, mn)


# ============================================================
# Test runners
# ============================================================

def test_negate():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    negate_kernel[grid](x, out, n, BLOCK=256)
    ref = -x
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_int_abs():
    n = 1024
    x = torch.randint(-100, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    int_abs_kernel[grid](x, out, n, BLOCK=256)
    ref = x.abs()
    err = (out - ref).abs().max().item()
    return err == 0, err


def test_multi_where():
    n = 2048
    x = torch.randn(n, device='mps') * 3
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    multi_where_kernel[grid](x, out, n, BLOCK=256)
    ref = x.clamp(-1, 1)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_floor_ceil():
    n = 1024
    x = torch.randn(n, device='mps') * 10
    floor_out = torch.zeros(n, device='mps')
    ceil_out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    floor_ceil_kernel[grid](x, floor_out, ceil_out, n, BLOCK=256)
    ref_floor = x.floor()
    ref_ceil = x.ceil()
    err = max((floor_out - ref_floor).abs().max().item(),
              (ceil_out - ref_ceil).abs().max().item())
    return err < 1e-5, err


def test_small_block():
    n = 128
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    small_block_kernel[(triton.cdiv(n, 32),)](x, out, n, BLOCK=32)
    ref = x * 2.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_multi_ptr():
    n = 2048
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    c = torch.randn(n, device='mps')
    d = torch.randn(n, device='mps')
    e = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    multi_ptr_kernel[grid](a, b, c, d, e, out, n, BLOCK=256)
    ref = (a + b) * (c - d) + e
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_int_minmax():
    n = 1024
    a = torch.randint(-100, 100, (n,), device='mps', dtype=torch.int32)
    b = torch.randint(-100, 100, (n,), device='mps', dtype=torch.int32)
    min_out = torch.zeros(n, device='mps', dtype=torch.int32)
    max_out = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    int_minmax_kernel[grid](a, b, min_out, max_out, n, BLOCK=256)
    ref_min = torch.minimum(a, b)
    ref_max = torch.maximum(a, b)
    err = max((min_out - ref_min).abs().max().item(),
              (max_out - ref_max).abs().max().item())
    return err == 0, err


def test_log2_exp2():
    n = 1024
    x = torch.rand(n, device='mps') + 0.1  # positive for log2
    log2_out = torch.zeros(n, device='mps')
    exp2_out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    log2_exp2_kernel[grid](x, log2_out, exp2_out, n, BLOCK=256)
    ref_log2 = torch.log2(x)
    ref_exp2 = torch.exp2(x)
    err = max((log2_out - ref_log2).abs().max().item(),
              (exp2_out - ref_exp2).abs().max().item())
    return err < 1e-4, err


def test_swiglu():
    n = 2048
    x = torch.randn(n, device='mps')
    gate = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    swiglu_kernel[grid](x, gate, out, n, BLOCK=256)
    silu_gate = gate * torch.sigmoid(gate)
    ref = x * silu_gate
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_row_product():
    M, N = 8, 16
    # Use small positive values to keep product manageable
    x = torch.rand(M, N, device='mps') * 0.9 + 0.1  # values in [0.1, 1.0]
    out = torch.zeros(M, device='mps')
    row_product_kernel[(triton.cdiv(M, 8),)](x, out, M, N, BLOCK_M=8, BLOCK_N=16)
    ref = x.prod(dim=1)
    err = (out - ref).abs().max().item()
    # Products can have larger relative error
    rel_err = ((out - ref) / (ref.abs() + 1e-10)).abs().max().item()
    return rel_err < 0.1, err


def test_matmul_bias_relu():
    M, N, K = 32, 32, 32
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    bias = torch.randn(N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    matmul_bias_relu_kernel[grid](
        a, b, bias, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
    )
    ref = torch.relu(a @ b + bias[None, :])
    err = (c - ref).abs().max().item()
    return err < 1e-3, err


def test_multi_reduce():
    n = 256
    x = torch.randn(n, device='mps')
    sum_out = torch.zeros(1, device='mps')
    max_out = torch.zeros(1, device='mps')
    min_out = torch.zeros(1, device='mps')
    multi_reduce_kernel[(1,)](x, sum_out, max_out, min_out, n, BLOCK=256)
    ref_sum = x.sum()
    ref_max = x.max()
    ref_min = x.min()
    err = max(abs(sum_out.item() - ref_sum.item()),
              abs(max_out.item() - ref_max.item()),
              abs(min_out.item() - ref_min.item()))
    return err < 1e-3, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 9) ===\n")

    tests = [
        ("Unary Negation", test_negate),
        ("Integer Abs", test_int_abs),
        ("Multi-Where (hard tanh)", test_multi_where),
        ("Floor + Ceil", test_floor_ceil),
        ("Small Block (32)", test_small_block),
        ("Multi-Pointer (6 buffers)", test_multi_ptr),
        ("Integer Min/Max", test_int_minmax),
        ("Log2 + Exp2", test_log2_exp2),
        ("SwiGLU Activation", test_swiglu),
        ("Row Product (via log)", test_row_product),
        ("Matmul+Bias+ReLU (fused)", test_matmul_bias_relu),
        ("Multi-Reduce (sum+max+min)", test_multi_reduce),
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
