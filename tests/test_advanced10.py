#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 10.

Targets: atomic operations, mixed-precision matmul, tl.trans,
histogram, prefix sum, strided access patterns, fp16 element-wise,
cumulative max, index manipulation, and tl.dot without scf.for.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Atomic add (histogram pattern)
@triton.jit
def atomic_add_kernel(x_ptr, hist_ptr, n, BLOCK: tl.constexpr):
    """Increment histogram bins using atomic add."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Bin index is the value itself (assuming small ints)
    bin_idx = x.to(tl.int32)
    # For each element, atomically add 1 to histogram bin
    tl.atomic_add(hist_ptr + bin_idx, 1, mask=mask)


# 2. fp16 element-wise kernel
@triton.jit
def fp16_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# 3. Mixed precision: fp16 input, fp32 accumulation, fp16 output
@triton.jit
def mixed_precision_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Compute (x * y) in fp32 then store as fp16."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    result = x * y
    tl.store(out_ptr + offs, result.to(tl.float16), mask=mask)


# 4. Strided load/store (non-contiguous access)
@triton.jit
def strided_access_kernel(x_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    """Load every stride-th element and write contiguously."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * stride, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


# 5. 2D tiled transpose
@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Transpose a matrix tile by tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    # Write transposed: swap row and col
    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :], tl.trans(x), mask=out_mask)


# 6. Prefix sum (cumulative sum via sequential scan)
@triton.jit
def prefix_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Simple prefix sum within a block using tl.cumsum."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    cumsum = tl.cumsum(x, axis=0)
    tl.store(out_ptr + offs, cumsum, mask=mask)


# 7. Single tt.dot (no scf.for loop)
@triton.jit
def single_dot_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak,
                      stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Matrix multiply with single dot (K <= BLOCK_K)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    c = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c)


# 8. Interleave two arrays
@triton.jit
def interleave_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Interleave: out[2i]=a[i], out[2i+1]=b[i]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs * 2, a, mask=mask)
    tl.store(out_ptr + offs * 2 + 1, b, mask=mask)


# 9. Conditional store (only write positive elements)
@triton.jit
def conditional_store_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Store only positive values, leave others at 0."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    pos_mask = (x > 0.0) & mask
    tl.store(out_ptr + offs, x, mask=pos_mask)


# 10. Chain of type casts: int32 -> float32 -> compute -> int32
@triton.jit
def cast_chain_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Convert int to float, compute sqrt, convert back."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x_int = tl.load(x_ptr + offs, mask=mask)
    x_float = x_int.to(tl.float32)
    result = tl.sqrt(x_float)
    tl.store(out_ptr + offs, result.to(tl.int32), mask=mask)


# 11. Dot product (reduce via sum of element-wise product)
@triton.jit
def dot_product_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Compute dot product a . b."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    dot = tl.sum(a * b, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, dot)


# 12. Chained arithmetic (long expression chain)
@triton.jit
def long_chain_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test long chain of arithmetic: ((x+1)*2 - 3) / 4 + x^2."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    result = ((x + 1.0) * 2.0 - 3.0) / 4.0 + x * x
    tl.store(out_ptr + offs, result, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_atomic_add():
    n = 1024
    num_bins = 8
    x = torch.randint(0, num_bins, (n,), device='mps', dtype=torch.int32)
    hist = torch.zeros(num_bins, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    atomic_add_kernel[grid](x, hist, n, BLOCK=256)
    ref = torch.zeros(num_bins, device='mps', dtype=torch.int32)
    for i in range(num_bins):
        ref[i] = (x == i).sum()
    err = (hist - ref).abs().max().item()
    return err == 0, err


def test_fp16_add():
    n = 2048
    x = torch.randn(n, device='mps', dtype=torch.float16)
    y = torch.randn(n, device='mps', dtype=torch.float16)
    out = torch.zeros(n, device='mps', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fp16_add_kernel[grid](x, y, out, n, BLOCK=256)
    ref = x + y
    err = (out.float() - ref.float()).abs().max().item()
    return err < 1e-3, err


def test_mixed_precision():
    n = 2048
    x = torch.randn(n, device='mps', dtype=torch.float16)
    y = torch.randn(n, device='mps', dtype=torch.float16)
    out = torch.zeros(n, device='mps', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    mixed_precision_kernel[grid](x, y, out, n, BLOCK=256)
    ref = (x.float() * y.float()).half()
    err = (out.float() - ref.float()).abs().max().item()
    return err < 1e-2, err


def test_strided_access():
    n = 256
    stride = 4
    x = torch.randn(n * stride, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    strided_access_kernel[grid](x, out, n, stride, BLOCK=256)
    ref = x[::stride]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_transpose():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(N, M, device='mps')
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    transpose_kernel[grid](x, out, M, N, BLOCK_M=32, BLOCK_N=32)
    ref = x.T
    # Only check the first 32 columns since BLOCK_N=32 and N=64 needs 2 blocks
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_prefix_sum():
    n = 128
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    prefix_sum_kernel[(1,)](x, out, n, BLOCK=128)
    ref = x.cumsum(dim=0)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_single_dot():
    M, N, K = 32, 32, 32
    a = torch.randn(M, K, device='mps')
    b = torch.randn(K, N, device='mps')
    c = torch.zeros(M, N, device='mps')
    grid = (1, 1)
    single_dot_kernel[grid](a, b, c, M, N, K,
                            a.stride(0), a.stride(1),
                            b.stride(0), b.stride(1),
                            c.stride(0), c.stride(1),
                            BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
    ref = a @ b
    err = (c - ref).abs().max().item()
    return err < 1e-3, err


def test_interleave():
    n = 512
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(2 * n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    interleave_kernel[grid](a, b, out, n, BLOCK=256)
    ref = torch.zeros(2 * n, device='mps')
    ref[0::2] = a
    ref[1::2] = b
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_conditional_store():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    conditional_store_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.where(x > 0, x, torch.zeros_like(x))
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_cast_chain():
    n = 256
    x = torch.randint(1, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    cast_chain_kernel[grid](x, out, n, BLOCK=256)
    ref = x.float().sqrt().int()
    err = (out - ref).abs().max().item()
    return err == 0, err


def test_dot_product():
    n = 256
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    dot_product_kernel[(1,)](a, b, out, n, BLOCK=256)
    ref = (a * b).sum()
    err = abs(out.item() - ref.item())
    return err < 1e-2, err


def test_long_chain():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    long_chain_kernel[grid](x, out, n, BLOCK=256)
    ref = ((x + 1.0) * 2.0 - 3.0) / 4.0 + x * x
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 10) ===\n")

    tests = [
        ("Atomic Add (histogram)", test_atomic_add),
        ("FP16 Add", test_fp16_add),
        ("Mixed Precision (f16*f16->f32->f16)", test_mixed_precision),
        ("Strided Access (stride=4)", test_strided_access),
        ("2D Transpose", test_transpose),
        ("Prefix Sum (cumsum)", test_prefix_sum),
        ("Single tt.dot (no loop)", test_single_dot),
        ("Interleave Arrays", test_interleave),
        ("Conditional Store", test_conditional_store),
        ("Cast Chain (i32->f32->sqrt->i32)", test_cast_chain),
        ("Dot Product (reduce)", test_dot_product),
        ("Long Arithmetic Chain", test_long_chain),
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
