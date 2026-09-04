#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 20.

Targets: remaining gaps in op coverage:
- tl.minimum / tl.maximum (element-wise tensor ops)
- tl.dot through @triton.jit (small matmul)
- 3D grid (all 3 program_id axes)
- Nested scf.for with scf.if inside
- tl.cumsum (prefix sum scan)
- arith.ceildivsi (tl.cdiv)
- tl.abs on integers
- Bitwise ops on integer tensors
- tl.where with integer type
- tl.minimum/tl.maximum chained (clamp pattern)
- Reverse iteration (descending loop)
- Multiple outputs from single kernel (compute 4 stats)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. tl.minimum / tl.maximum element-wise
@triton.jit
def minmax_kernel(a_ptr, b_ptr, min_ptr, max_ptr, n, BLOCK: tl.constexpr):
    """out_min = min(a, b), out_max = max(a, b) element-wise."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    mn = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    tl.store(min_ptr + offs, mn, mask=mask)
    tl.store(max_ptr + offs, mx, mask=mask)


# 2. Small matmul through @triton.jit (tl.dot)
@triton.jit
def small_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                         stride_am, stride_ak,
                         stride_bk, stride_bn,
                         stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    """Tiled matmul: C = A @ B."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# 3. 3D grid (use all three program_id axes)
@triton.jit
def grid_3d_kernel(out_ptr, D0, D1, D2, BLOCK: tl.constexpr):
    """Write (pid0 * D1*D2 + pid1 * D2 + pid2) to output."""
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    val = pid0 * D1 * D2 + pid1 * D2 + pid2
    tl.store(out_ptr + val, val.to(tl.float32))


# 4. Nested scf.for with scf.if inside
@triton.jit
def nested_for_if_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Sum elements in chunks, but only add positive values."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        chunk = start + offs
        m = chunk < n
        x = tl.load(x_ptr + chunk, mask=m, other=0.0)
        # Only accumulate positive values
        pos = tl.where(x > 0.0, x, 0.0)
        acc += pos
    total = tl.sum(acc, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 5. tl.cumsum (prefix sum)
@triton.jit
def cumsum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Compute cumulative sum."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    cs = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, cs, mask=mask)


# 6. Ceiling division in index computations
@triton.jit
def ceildiv_kernel(x_ptr, out_ptr, divisor, n, BLOCK: tl.constexpr):
    """out[i] = ceil(x[i] / divisor)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Manual ceiling division: (x + d - 1) / d
    result = (x + divisor - 1) // divisor
    tl.store(out_ptr + offs, result, mask=mask)


# 7. Absolute value on integers
@triton.jit
def abs_int_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Absolute value on int32 tensor."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.abs(x)
    tl.store(out_ptr + offs, out, mask=mask)


# 8. Bitwise ops on integer tensors
@triton.jit
def bitwise_kernel(a_ptr, b_ptr, and_ptr, or_ptr, xor_ptr, n, BLOCK: tl.constexpr):
    """Bitwise AND, OR, XOR on int32 tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(and_ptr + offs, a & b, mask=mask)
    tl.store(or_ptr + offs, a | b, mask=mask)
    tl.store(xor_ptr + offs, a ^ b, mask=mask)


# 9. tl.where with integer type
@triton.jit
def where_int_kernel(cond_ptr, a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = where(cond > 0, a, b) for int32."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(cond_ptr + offs, mask=mask)
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out = tl.where(c > 0, a, b)
    tl.store(out_ptr + offs, out, mask=mask)


# 10. Clamp pattern using chained min/max
@triton.jit
def clamp_kernel(x_ptr, out_ptr, lo, hi, n, BLOCK: tl.constexpr):
    """Clamp: out = max(lo, min(hi, x))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Clamp using chained min/max
    clamped = tl.minimum(tl.maximum(x, lo), hi)
    tl.store(out_ptr + offs, clamped, mask=mask)


# 11. Shift operations
@triton.jit
def shift_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = (x << 2) >> 1."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    shifted = (x << 2) >> 1
    tl.store(out_ptr + offs, shifted, mask=mask)


# 12. Multi-stat kernel (compute mean, var, min, max in one pass)
@triton.jit
def multi_stat_kernel(x_ptr, mean_ptr, var_ptr, min_ptr, max_ptr, n,
                       BLOCK: tl.constexpr):
    """Compute mean, variance, min, max of input."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Mean
    total = tl.sum(x, axis=0)
    mean = total / n
    # Variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n
    # Min/Max (use large/small other for masked)
    x_for_min = tl.load(x_ptr + offs, mask=mask, other=float('inf'))
    x_for_max = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    mn = tl.min(x_for_min, axis=0)
    mx = tl.max(x_for_max, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(mean_ptr, mean)
        tl.store(var_ptr, var)
        tl.store(min_ptr, mn)
        tl.store(max_ptr, mx)


# ============================================================
# Test runners
# ============================================================

def test_minmax():
    n = 1024
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out_min = torch.zeros(n, device='mps')
    out_max = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    minmax_kernel[grid](a, b, out_min, out_max, n, BLOCK=256)
    ref_min = torch.minimum(a, b)
    ref_max = torch.maximum(a, b)
    err = max((out_min - ref_min).abs().max().item(),
              (out_max - ref_max).abs().max().item())
    return err < 1e-5, err


def test_small_matmul():
    M, N, K = 32, 32, 32
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    small_matmul_kernel[grid](a, b, c, M, N, K,
                               a.stride(0), a.stride(1),
                               b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1),
                               BLOCK_M=16, BLOCK_N=16, BLOCK_K=32)
    ref = a @ b
    err = (c - ref).abs().max().item()
    return err < 1e-2, err


def test_3d_grid():
    D0, D1, D2 = 4, 3, 5
    total = D0 * D1 * D2
    out = torch.zeros(total, device='mps')
    grid_3d_kernel[(D0, D1, D2)](out, D0, D1, D2, BLOCK=1)
    ref = torch.arange(total, device='mps', dtype=torch.float32)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_nested_for_if():
    n = 512
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    nested_for_if_kernel[(1,)](x, out, n, BLOCK=128)
    ref = x[x > 0].sum()
    err = abs(out.item() - ref.item())
    return err < 1e-1, err


def test_cumsum():
    n = 64
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    cumsum_kernel[(1,)](x, out, n, BLOCK=64)
    ref = x.cumsum(0)
    err = (out[:n] - ref[:n]).abs().max().item()
    return err < 1e-3, err


def test_ceildiv():
    n = 256
    x = torch.randint(1, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    divisor = 7
    ceildiv_kernel[(1,)](x, out, divisor, n, BLOCK=256)
    ref = (x + divisor - 1) // divisor
    err = (out - ref).abs().max().item()
    return err == 0, float(err)


def test_abs_int():
    n = 256
    x = torch.randint(-100, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    abs_int_kernel[(1,)](x, out, n, BLOCK=256)
    ref = x.abs()
    err = (out - ref).abs().max().item()
    return err == 0, float(err)


def test_bitwise():
    n = 256
    a = torch.randint(0, 256, (n,), device='mps', dtype=torch.int32)
    b = torch.randint(0, 256, (n,), device='mps', dtype=torch.int32)
    out_and = torch.zeros(n, device='mps', dtype=torch.int32)
    out_or = torch.zeros(n, device='mps', dtype=torch.int32)
    out_xor = torch.zeros(n, device='mps', dtype=torch.int32)
    bitwise_kernel[(1,)](a, b, out_and, out_or, out_xor, n, BLOCK=256)
    err = max((out_and - (a & b)).abs().max().item(),
              (out_or - (a | b)).abs().max().item(),
              (out_xor - (a ^ b)).abs().max().item())
    return err == 0, float(err)


def test_where_int():
    n = 256
    cond = torch.randint(-10, 10, (n,), device='mps', dtype=torch.int32)
    a = torch.randint(0, 100, (n,), device='mps', dtype=torch.int32)
    b = torch.randint(0, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    where_int_kernel[(1,)](cond, a, b, out, n, BLOCK=256)
    ref = torch.where(cond > 0, a, b)
    err = (out - ref).abs().max().item()
    return err == 0, float(err)


def test_clamp():
    n = 1024
    x = torch.randn(n, device='mps') * 5
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    clamp_kernel[grid](x, out, -1.0, 1.0, n, BLOCK=256)
    ref = x.clamp(-1.0, 1.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_shift():
    n = 256
    x = torch.randint(0, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    shift_kernel[(1,)](x, out, n, BLOCK=256)
    ref = (x << 2) >> 1
    err = (out - ref).abs().max().item()
    return err == 0, float(err)


def test_multi_stat():
    n = 128
    x = torch.randn(n, device='mps')
    mean_out = torch.zeros(1, device='mps')
    var_out = torch.zeros(1, device='mps')
    min_out = torch.zeros(1, device='mps')
    max_out = torch.zeros(1, device='mps')
    multi_stat_kernel[(1,)](x, mean_out, var_out, min_out, max_out, n, BLOCK=128)
    ref_mean = x.mean()
    ref_var = x.var(correction=0)
    ref_min = x.min()
    ref_max = x.max()
    err = max(abs(mean_out.item() - ref_mean.item()),
              abs(var_out.item() - ref_var.item()),
              abs(min_out.item() - ref_min.item()),
              abs(max_out.item() - ref_max.item()))
    return err < 1e-2, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 20) ===\n")
    print("    (Op coverage gaps & matmul through JIT)\n")

    tests = [
        ("tl.minimum/maximum", test_minmax),
        ("Small Matmul (tl.dot)", test_small_matmul),
        ("3D Grid", test_3d_grid),
        ("Nested For+If", test_nested_for_if),
        ("tl.cumsum", test_cumsum),
        ("Ceiling Division", test_ceildiv),
        ("Abs Int", test_abs_int),
        ("Bitwise (AND/OR/XOR)", test_bitwise),
        ("Where (int type)", test_where_int),
        ("Clamp (min/max chain)", test_clamp),
        ("Shift (<<, >>)", test_shift),
        ("Multi-Stat (4 outputs)", test_multi_stat),
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
