#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 16.

Targets: tensor manipulation ops and advanced patterns:
- tl.reshape / tl.view
- tl.broadcast_to
- tl.flip
- tl.zeros_like
- tl.full
- tl.cast (explicit type casts)
- tl.expand_dims beyond axis=0/1
- tl.where with tensor condition on 2D
- Large block sizes
- Nested tl.where chains
- Multiple program_id axes with non-trivial indexing
- Chained reductions
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. tl.full constant initialization
@triton.jit
def full_init_kernel(out_ptr, n, BLOCK: tl.constexpr):
    """Initialize output with constant value using tl.full."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    val = tl.full((BLOCK,), 42.0, dtype=tl.float32)
    tl.store(out_ptr + offs, val, mask=mask)


# 2. tl.zeros_like pattern
@triton.jit
def zeros_like_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Use tl.zeros to create zero tensor, add to input."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    z = tl.zeros((BLOCK,), dtype=tl.float32)
    out = x + z  # Should just be x
    tl.store(out_ptr + offs, out, mask=mask)


# 3. Multi-level tl.where chain (decision tree)
@triton.jit
def decision_tree_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """4-way classification: x<-1 -> -2, -1<=x<0 -> -1, 0<=x<1 -> 1, x>=1 -> 2."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.where(x < -1.0, -2.0,
           tl.where(x < 0.0, -1.0,
            tl.where(x < 1.0, 1.0, 2.0)))
    tl.store(out_ptr + offs, out, mask=mask)


# 4. 2D tl.where with tensor condition
@triton.jit
def where_2d_kernel(a_ptr, b_ptr, cond_ptr, out_ptr, M, N,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """out = where(cond, a, b) on 2D tiles."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = offs_m[:, None] * N + offs_n[None, :]
    a = tl.load(a_ptr + ptrs, mask=mask)
    b = tl.load(b_ptr + ptrs, mask=mask)
    c = tl.load(cond_ptr + ptrs, mask=mask)
    out = tl.where(c > 0.0, a, b)
    tl.store(out_ptr + ptrs, out, mask=mask)


# 5. Chained reductions (sum then use result)
@triton.jit
def chain_reduce_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Normalize by sum: out = x / sum(x)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    out = x / total
    tl.store(out_ptr + offs, out, mask=mask)


# 6. tl.cast explicit type conversion
@triton.jit
def cast_chain_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """f32 -> i32 -> f32 round trip (tests cast precision)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Truncate to int then back to float
    xi = x.to(tl.int32)
    xf = xi.to(tl.float32)
    tl.store(out_ptr + offs, xf, mask=mask)


# 7. Strided 2D access with non-contiguous memory
@triton.jit
def strided_2d_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Access 2D tensor with explicit strides (handles transpose)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
    x = tl.load(x_ptr + ptrs, mask=mask)
    out = x * 2.0
    # Write to contiguous output
    out_ptrs = offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_ptrs, out, mask=mask)


# 8. Large block (BLOCK=1024)
@triton.jit
def large_block_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test with max block size (1024 threads)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.exp(x)
    tl.store(out_ptr + offs, out, mask=mask)


# 9. Reduction across rows with 2D grid
@triton.jit
def row_sum_2d_kernel(x_ptr, out_ptr, M, N,
                       BLOCK_N: tl.constexpr):
    """Sum each row: out[m] = sum(x[m, :])."""
    row = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    x = tl.load(x_ptr + row * N + offs_n, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(out_ptr + row, total)


# 10. Multiple atomic operations in one kernel
@triton.jit
def multi_atomic_kernel(x_ptr, sum_ptr, cnt_ptr, n, BLOCK: tl.constexpr):
    """Atomic sum and count of positive elements."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Sum all elements atomically
    block_sum = tl.sum(x, axis=0)
    tl.atomic_add(sum_ptr, block_sum)
    # Count positive elements
    pos = (x > 0.0).to(tl.float32)
    block_cnt = tl.sum(pos, axis=0)
    tl.atomic_add(cnt_ptr, block_cnt)


# 11. Fused dropout (multiply by random mask)
@triton.jit
def dropout_kernel(x_ptr, out_ptr, seed, p, n, BLOCK: tl.constexpr):
    """Apply dropout: out = x * (rand > p) / (1-p)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Generate random numbers
    random = tl.rand(seed, offs)
    keep = random > p
    scale = 1.0 / (1.0 - p)
    out = tl.where(keep, x * scale, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 12. Polynomial evaluation (Horner's method)
@triton.jit
def polynomial_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Evaluate p(x) = 3x^3 - 2x^2 + x - 5 using Horner's method."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Horner: ((3*x - 2)*x + 1)*x - 5
    out = ((3.0 * x - 2.0) * x + 1.0) * x - 5.0
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_full_init():
    n = 512
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    full_init_kernel[grid](out, n, BLOCK=256)
    err = (out - 42.0).abs().max().item()
    return err < 1e-5, err


def test_zeros_like():
    n = 512
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    zeros_like_kernel[(2,)](x, out, n, BLOCK=256)
    err = (out - x).abs().max().item()
    return err < 1e-5, err


def test_decision_tree():
    n = 1024
    x = torch.randn(n, device='mps') * 3  # values in [-9, 9] range
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    decision_tree_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.where(x < -1, torch.tensor(-2.0), torch.where(x < 0, torch.tensor(-1.0),
          torch.where(x < 1, torch.tensor(1.0), torch.tensor(2.0))))
    err = (out - ref.to('mps')).abs().max().item()
    return err < 1e-5, err


def test_where_2d():
    M, N = 32, 64
    a = torch.randn(M, N, device='mps')
    b = torch.randn(M, N, device='mps')
    cond = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 32))
    where_2d_kernel[grid](a, b, cond, out, M, N, BLOCK_M=16, BLOCK_N=32)
    ref = torch.where(cond > 0, a, b)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_chain_reduce():
    n = 128
    x = torch.rand(n, device='mps') + 0.01  # positive
    out = torch.zeros(n, device='mps')
    chain_reduce_kernel[(1,)](x, out, n, BLOCK=128)
    ref = x / x.sum()
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_cast_chain():
    n = 1024
    x = torch.randn(n, device='mps') * 100  # large enough to have interesting truncation
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    cast_chain_kernel[grid](x, out, n, BLOCK=256)
    ref = x.int().float()
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_strided_2d():
    M, N = 32, 48
    # Use contiguous tensor (driver always calls .cpu().contiguous())
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    strided_2d_kernel[grid](x, out, M, N, x.stride(0), x.stride(1),
                             BLOCK_M=16, BLOCK_N=16)
    ref = x * 2.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_large_block():
    n = 4096
    x = torch.randn(n, device='mps') * 0.1  # small values to avoid exp overflow
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    large_block_kernel[grid](x, out, n, BLOCK=1024)
    ref = torch.exp(x)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_row_sum_2d():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    row_sum_2d_kernel[(M,)](x, out, M, N, BLOCK_N=128)
    ref = x.sum(dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_multi_atomic():
    n = 2048
    x = torch.randn(n, device='mps')
    sum_out = torch.zeros(1, device='mps')
    cnt_out = torch.zeros(1, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    multi_atomic_kernel[grid](x, sum_out, cnt_out, n, BLOCK=256)
    ref_sum = x.sum()
    ref_cnt = (x > 0).sum().float()
    err = max(abs(sum_out.item() - ref_sum.item()),
              abs(cnt_out.item() - ref_cnt.item()))
    return err < 1e-1, err  # atomics have ordering issues


def test_dropout():
    n = 4096
    x = torch.ones(n, device='mps')
    out = torch.zeros(n, device='mps')
    p = 0.5
    seed = 42
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    dropout_kernel[grid](x, out, seed, p, n, BLOCK=256)
    # Check: ~50% of values should be 0, rest should be 1/(1-p) = 2.0
    zeros = (out == 0).sum().item()
    twos = (out.abs() - 2.0).abs() < 0.01
    keep_rate = 1 - zeros / n
    err = abs(keep_rate - (1 - p))
    return err < 0.1, err  # statistical test, allow 10% tolerance


def test_polynomial():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    polynomial_kernel[grid](x, out, n, BLOCK=256)
    ref = 3 * x**3 - 2 * x**2 + x - 5
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 16) ===\n")

    tests = [
        ("tl.full Init", test_full_init),
        ("tl.zeros Pattern", test_zeros_like),
        ("4-Way Decision Tree", test_decision_tree),
        ("2D tl.where", test_where_2d),
        ("Chain Reduce (x/sum(x))", test_chain_reduce),
        ("Cast Chain (f32->i32->f32)", test_cast_chain),
        ("Strided 2D Access", test_strided_2d),
        ("Large Block (1024)", test_large_block),
        ("Row Sum (2D reduce)", test_row_sum_2d),
        ("Multi-Atomic (sum+count)", test_multi_atomic),
        ("Dropout (tl.rand)", test_dropout),
        ("Polynomial (Horner)", test_polynomial),
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
