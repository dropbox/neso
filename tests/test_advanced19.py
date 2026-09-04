#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 19.

Targets: edge cases and stress tests for the codegen:
- Single-element tensors
- BLOCK > n (over-masked)
- Multiple scf.for loops in one kernel
- Nested scf.if inside scf.for
- tl.where with both branches being tensors (no scalar)
- Reductions on very small tensors (n=1, n=2)
- Cast between int and float mid-computation
- Large number of function arguments (>10)
- Multiple tl.store to same location (write-after-write)
- Complex boolean logic (AND/OR/NOT chains)
- Accumulator with different init values
- Mixing tl.load from different dtypes in same kernel
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Single element operations
@triton.jit
def single_element_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    """Test with n=1 (single element)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < 1
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.exp(x)
    tl.store(out_ptr + offs, out, mask=mask)


# 2. Over-masked (BLOCK >> n)
@triton.jit
def over_masked_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test where BLOCK (256) >> n (7)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # Reduction should only sum the valid elements
    total = tl.sum(x, axis=0)
    # Store total to first element
    if tl.program_id(0) == 0:
        tl.store(out_ptr, total)


# 3. Multiple sequential for loops
@triton.jit
def multi_loop_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Two independent for loops operating on the same data."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n

    # First pass: sum
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        chunk = start + offs
        m = chunk < n
        x = tl.load(x_ptr + chunk, mask=m, other=0.0)
        acc1 += x
    sum_val = tl.sum(acc1, axis=0)

    # Second pass: sum of squares
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        chunk = start + offs
        m = chunk < n
        x = tl.load(x_ptr + chunk, mask=m, other=0.0)
        acc2 += x * x
    sum_sq = tl.sum(acc2, axis=0)

    # Store both results
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, sum_val)
        tl.store(out_ptr + 1, sum_sq)


# 4. Complex boolean logic
@triton.jit
def bool_logic_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test complex boolean: out = (a > 0 AND b > 0) OR (c < 0 AND a < -1)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    cond1 = (a > 0.0) & (b > 0.0)
    cond2 = (c < 0.0) & (a < -1.0)
    cond = cond1 | cond2
    out = tl.where(cond, 1.0, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 5. Write-after-write (multiple stores to same location)
@triton.jit
def waw_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Write initial value, then overwrite with computed value."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # First write: zeros
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)
    # Second write: actual values
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


# 6. Many arguments (>10 params)
@triton.jit
def many_args_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr,
                      out_ptr, alpha, beta, gamma, n,
                      BLOCK: tl.constexpr):
    """Kernel with 11 non-constexpr args."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    d = tl.load(d_ptr + offs, mask=mask)
    e = tl.load(e_ptr + offs, mask=mask)
    f = tl.load(f_ptr + offs, mask=mask)
    out = alpha * (a + b) + beta * (c + d) + gamma * (e + f)
    tl.store(out_ptr + offs, out, mask=mask)


# 7. Mixed int/float loads in same kernel
@triton.jit
def mixed_dtype_kernel(float_ptr, int_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Load float and int data, combine them."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    fval = tl.load(float_ptr + offs, mask=mask)
    ival = tl.load(int_ptr + offs, mask=mask)
    # Int to float conversion + arithmetic
    out = fval + ival.to(tl.float32) * 0.1
    tl.store(out_ptr + offs, out, mask=mask)


# 8. Reduction on n=1 and n=2
@triton.jit
def tiny_reduce_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Sum over a very small number of elements."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 9. tl.where with both branches being loaded tensors
@triton.jit
def where_tensor_branches_kernel(cond_ptr, a_ptr, b_ptr, out_ptr, n,
                                   BLOCK: tl.constexpr):
    """out = where(cond > 0, a, b) where both a and b are tensors."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(cond_ptr + offs, mask=mask)
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out = tl.where(c > 0.0, a, b)
    tl.store(out_ptr + offs, out, mask=mask)


# 10. Non-zero accumulator init
@triton.jit
def biased_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Sum with a non-zero initial accumulator (bias=100)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0) + 100.0  # bias
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 11. Index arithmetic with modulo and divide
@triton.jit
def index_math_kernel(x_ptr, out_ptr, H, W, BLOCK: tl.constexpr):
    """Convert linear index to 2D and back: out[h*W+w] = x[h*W+w] * (h+w)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n = H * W
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    h = offs // W
    w = offs % W
    scale = (h + w).to(tl.float32)
    out = x * scale
    tl.store(out_ptr + offs, out, mask=mask)


# 12. Interleaved reads from two arrays (zip pattern)
@triton.jit
def zip_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Interleave: out[2*i] = a[i], out[2*i+1] = b[i]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs * 2, a, mask=mask)
    tl.store(out_ptr + offs * 2 + 1, b, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_single_element():
    x = torch.tensor([2.0], device='mps')
    out = torch.zeros(1, device='mps')
    single_element_kernel[(1,)](x, out, BLOCK=256)
    ref = torch.exp(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_over_masked():
    n = 7
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    over_masked_kernel[(1,)](x, out, n, BLOCK=256)
    ref = x.sum()
    err = abs(out.item() - ref.item())
    return err < 1e-4, err


def test_multi_loop():
    n = 512
    x = torch.randn(n, device='mps')
    out = torch.zeros(2, device='mps')
    multi_loop_kernel[(1,)](x, out, n, BLOCK=128)
    ref_sum = x.sum()
    ref_sumsq = (x * x).sum()
    err = max(abs(out[0].item() - ref_sum.item()),
              abs(out[1].item() - ref_sumsq.item()))
    return err < 1e-1, err


def test_bool_logic():
    n = 1024
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    c = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    bool_logic_kernel[grid](a, b, c, out, n, BLOCK=256)
    ref = ((a > 0) & (b > 0) | (c < 0) & (a < -1)).float()
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_waw():
    n = 256
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    waw_kernel[(1,)](x, out, n, BLOCK=256)
    ref = x * 2.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_many_args():
    n = 256
    a, b, c, d, e, f = [torch.randn(n, device='mps') for _ in range(6)]
    out = torch.zeros(n, device='mps')
    alpha, beta, gamma = 2.0, 3.0, 0.5
    many_args_kernel[(1,)](a, b, c, d, e, f, out, alpha, beta, gamma, n, BLOCK=256)
    ref = alpha * (a + b) + beta * (c + d) + gamma * (e + f)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_mixed_dtype():
    n = 256
    fval = torch.randn(n, device='mps')
    ival = torch.randint(-100, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    mixed_dtype_kernel[(1,)](fval, ival, out, n, BLOCK=256)
    ref = fval + ival.float() * 0.1
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_tiny_reduce():
    # n=1
    x1 = torch.tensor([42.0], device='mps')
    out1 = torch.zeros(1, device='mps')
    tiny_reduce_kernel[(1,)](x1, out1, 1, BLOCK=256)
    err1 = abs(out1.item() - 42.0)

    # n=2
    x2 = torch.tensor([3.0, 7.0], device='mps')
    out2 = torch.zeros(1, device='mps')
    tiny_reduce_kernel[(1,)](x2, out2, 2, BLOCK=256)
    err2 = abs(out2.item() - 10.0)

    err = max(err1, err2)
    return err < 1e-5, err


def test_where_tensor_branches():
    n = 1024
    cond = torch.randn(n, device='mps')
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    where_tensor_branches_kernel[grid](cond, a, b, out, n, BLOCK=256)
    ref = torch.where(cond > 0, a, b)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_biased_sum():
    n = 128
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    biased_sum_kernel[(1,)](x, out, n, BLOCK=128)
    ref = x.sum() + 100.0
    err = abs(out.item() - ref.item())
    return err < 1e-3, err


def test_index_math():
    H, W = 16, 32
    n = H * W
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    index_math_kernel[grid](x, out, H, W, BLOCK=256)
    indices = torch.arange(n, device='mps')
    h = indices // W
    w = indices % W
    ref = x * (h + w).float()
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_zip():
    n = 128
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(n * 2, device='mps')
    zip_kernel[(1,)](a, b, out, n, BLOCK=128)
    ref = torch.zeros(n * 2, device='mps')
    ref[0::2] = a
    ref[1::2] = b
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 19) ===\n")
    print("    (Edge cases & stress tests)\n")

    tests = [
        ("Single Element (n=1)", test_single_element),
        ("Over-Masked (BLOCK>>n)", test_over_masked),
        ("Multi-Loop (2 passes)", test_multi_loop),
        ("Boolean Logic (AND/OR)", test_bool_logic),
        ("Write-After-Write", test_waw),
        ("Many Args (11 params)", test_many_args),
        ("Mixed Dtype (float+int)", test_mixed_dtype),
        ("Tiny Reduce (n=1,2)", test_tiny_reduce),
        ("Where(tensor,tensor)", test_where_tensor_branches),
        ("Biased Sum (+100)", test_biased_sum),
        ("Index Math (div/mod)", test_index_math),
        ("Zip Interleave", test_zip),
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
