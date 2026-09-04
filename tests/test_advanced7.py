#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 7.

Targets: nested loops, scf.if inside scf.for, multi-accumulator loops,
         tl.where with tensor conditions, complex pointer arithmetic,
         fp16 reductions, int64 types, very large grids.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Nested for loops (not matmul - general purpose)
@triton.jit
def nested_loop_sum_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    """Sum all elements by iterating over rows then columns."""
    acc = 0.0
    for row in range(M):
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
        row_sum = tl.sum(x, axis=0)
        acc += row_sum
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, acc)


# 2. scf.if inside scf.for (conditional in loop body)
@triton.jit
def conditional_accumulate_kernel(x_ptr, out_ptr, n, threshold, BLOCK: tl.constexpr):
    """Sum only positive elements chunk by chunk with conditional logic."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        chunk_offs = start + offs
        mask = chunk_offs < n
        x = tl.load(x_ptr + chunk_offs, mask=mask, other=0.0)
        # Conditional: only accumulate positive elements
        pos_mask = x > threshold
        acc += tl.where(pos_mask, x, 0.0)
    total = tl.sum(acc, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 3. Multi-accumulator loop: track both sum and count
@triton.jit
def mean_streaming_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Compute mean by streaming through data with running sum + count."""
    offs = tl.arange(0, BLOCK)
    running_sum = 0.0
    for start in range(0, n, BLOCK):
        chunk_offs = start + offs
        mask = chunk_offs < n
        x = tl.load(x_ptr + chunk_offs, mask=mask, other=0.0)
        chunk_sum = tl.sum(x, axis=0)
        running_sum += chunk_sum
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, running_sum / n)


# 4. tl.where with complex conditions (AND/OR of comparisons)
@triton.jit
def band_pass_kernel(x_ptr, out_ptr, lo, hi, n, BLOCK: tl.constexpr):
    """Zero out values outside [lo, hi] range."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    in_band = (x >= lo) & (x <= hi)
    out = tl.where(in_band, x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 5. Power function (x^n via repeated multiply — tests complex expression chains)
@triton.jit
def power3_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Compute x^3."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * x * x
    tl.store(out_ptr + offs, out, mask=mask)


# 6. Reduction min (not just max — test min reduce)
@triton.jit
def row_min_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=float('inf'))
    row_mins = tl.min(x, axis=1)
    tl.store(out_ptr + offs_m, row_mins, mask=offs_m < M)


# 7. Multi-output reduction: sum AND max in one kernel
@triton.jit
def sum_and_max_kernel(x_ptr, sum_ptr, max_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)
    m = tl.max(x, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(sum_ptr, s)
        tl.store(max_ptr, m)


# 8. Negative indexing pattern (load from end of array)
@triton.jit
def reverse_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Reverse an array."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # Read from reversed position
    rev_offs = n - 1 - offs
    x = tl.load(x_ptr + rev_offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


# 9. Exponential sum (logsumexp building block)
@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Numerically stable logsumexp."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    sum_exp = tl.sum(exp_x, axis=0)
    result = x_max + tl.log(sum_exp)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, result)


# 10. Very large grid (many program IDs)
@triton.jit
def large_grid_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + 1.0, mask=mask)


# 11. Double reduction: reduce 2D to scalar
@triton.jit
def global_sum_2d_kernel(x_ptr, out_ptr, M, N,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Sum all elements of a 2D matrix."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    # Reduce over both dimensions
    row_sums = tl.sum(x, axis=1)  # [M]
    total = tl.sum(row_sums, axis=0)  # scalar
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 12. Fused residual + layer norm pattern
@triton.jit
def residual_add_kernel(x_ptr, residual_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = x + residual (fused pattern common in transformers)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.load(residual_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + r, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_nested_loop_sum():
    M, N = 8, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(1, device='mps')
    nested_loop_sum_kernel[(1,)](x, out, M, N, BLOCK_N=64)
    ref = x.sum()
    err = abs(out.item() - ref.item())
    return err < 1e-2, err


def test_conditional_accumulate():
    n = 512
    x = torch.randn(n, device='mps')
    threshold = 0.0
    out = torch.zeros(1, device='mps')
    conditional_accumulate_kernel[(1,)](x, out, n, threshold, BLOCK=128)
    ref = x[x > threshold].sum()
    err = abs(out.item() - ref.item())
    return err < 1e-2, err


def test_mean_streaming():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    mean_streaming_kernel[(1,)](x, out, n, BLOCK=128)
    ref = x.mean()
    err = abs(out.item() - ref.item())
    return err < 1e-3, err


def test_band_pass():
    n = 2048
    x = torch.randn(n, device='mps') * 3
    lo, hi = -1.0, 1.0
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    band_pass_kernel[grid](x, out, lo, hi, n, BLOCK=256)
    ref = torch.where((x >= lo) & (x <= hi), x, torch.zeros_like(x))
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_power3():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    power3_kernel[grid](x, out, n, BLOCK=256)
    ref = x ** 3
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_row_min():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.full((M,), float('inf'), device='mps')
    row_min_kernel[(triton.cdiv(M, 32),)](x, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = x.min(dim=1).values
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_sum_and_max():
    n = 256
    x = torch.randn(n, device='mps')
    sum_out = torch.zeros(1, device='mps')
    max_out = torch.zeros(1, device='mps')
    sum_and_max_kernel[(1,)](x, sum_out, max_out, n, BLOCK=256)
    ref_sum = x.sum()
    ref_max = x.max()
    err_sum = abs(sum_out.item() - ref_sum.item())
    err_max = abs(max_out.item() - ref_max.item())
    err = max(err_sum, err_max)
    return err < 1e-3, err


def test_reverse():
    n = 512
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    reverse_kernel[grid](x, out, n, BLOCK=256)
    ref = x.flip(0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_logsumexp():
    n = 256
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    logsumexp_kernel[(1,)](x, out, n, BLOCK=256)
    ref = torch.logsumexp(x, dim=0)
    err = abs(out.item() - ref.item())
    return err < 1e-3, err


def test_large_grid():
    n = 65536
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    large_grid_kernel[grid](x, out, n, BLOCK=256)
    ref = x + 1.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_global_sum_2d():
    M, N = 16, 32
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(1, device='mps')
    global_sum_2d_kernel[(1,)](x, out, M, N, BLOCK_M=16, BLOCK_N=32)
    ref = x.sum()
    err = abs(out.item() - ref.item())
    return err < 1e-2, err


def test_residual_add():
    n = 4096
    x = torch.randn(n, device='mps')
    r = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    residual_add_kernel[grid](x, r, out, n, BLOCK=256)
    ref = x + r
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 7) ===\n")

    tests = [
        ("Nested Loop Sum (rows*cols)", test_nested_loop_sum),
        ("Conditional Accumulate", test_conditional_accumulate),
        ("Mean Streaming (chunked)", test_mean_streaming),
        ("Band Pass (AND cond)", test_band_pass),
        ("Power x^3", test_power3),
        ("Row Min (2D reduce)", test_row_min),
        ("Sum+Max (dual reduce)", test_sum_and_max),
        ("Reverse Array", test_reverse),
        ("LogSumExp", test_logsumexp),
        ("Large Grid (65K elems)", test_large_grid),
        ("Global Sum 2D (double reduce)", test_global_sum_2d),
        ("Residual Add", test_residual_add),
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
