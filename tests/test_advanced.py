#!/usr/bin/env python3
"""Test advanced Triton kernel patterns to find remaining IR coverage gaps."""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions – patterns targeting untested IR paths
# ============================================================

# 1. expand_dims: 1D -> 2D broadcasting pattern (outer product)
@triton.jit
def outer_product_kernel(x_ptr, y_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(x_ptr + offs_m, mask=offs_m < M)
    y = tl.load(y_ptr + offs_n, mask=offs_n < N)
    # expand_dims + broadcast -> 2D outer product
    out = x[:, None] * y[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out, mask=mask)


# 2. tl.where on 2D tensors
@triton.jit
def where_2d_kernel(x_ptr, y_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    y_ptrs = y_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask)
    y = tl.load(y_ptrs, mask=mask)
    out = tl.where(x > y, x, y)  # equivalent to max(x,y)
    tl.store(ptrs, out, mask=mask)


# 3. Multiple reductions in one kernel (mean + variance pattern)
@triton.jit
def mean_var_kernel(x_ptr, mean_ptr, var_ptr, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    # Only thread 0 stores (single-element output)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(mean_ptr, mean)
        tl.store(var_ptr, var)


# 4. Chained operations with mixed int/float
@triton.jit
def mixed_dtype_kernel(x_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Load float data, use int indices for gather-like access."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)  # int32 indices
    # Use indices to load from x (gather pattern)
    x = tl.load(x_ptr + idx, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


# 5. Multiple scf.if conditions (nested if/else through tl.where chains)
@triton.jit
def piecewise_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Piecewise function: f(x) = x^2 if x>0, -x if x<=-1, x+1 otherwise."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Nested where
    out = tl.where(x > 0, x * x, tl.where(x <= -1.0, -x, x + 1.0))
    tl.store(out_ptr + offs, out, mask=mask)


# 6. Type cast: float -> int -> float round trip
@triton.jit
def cast_roundtrip_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Cast to int32 (truncates) then back to float
    xi = x.to(tl.int32)
    xf = xi.to(tl.float32)
    tl.store(out_ptr + offs, xf, mask=mask)


# 7. fp16 element-wise operations
@triton.jit
def fp16_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# 8. Large block size (1024 = Metal's max threads per threadgroup)
@triton.jit
def large_block_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * 2.0, mask=mask)


# 9. Multiple outputs from single kernel
@triton.jit
def multi_output_kernel(x_ptr, out_sin_ptr, out_cos_ptr, out_neg_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_sin_ptr + offs, tl.sin(x), mask=mask)
    tl.store(out_cos_ptr + offs, tl.cos(x), mask=mask)
    tl.store(out_neg_ptr + offs, -x, mask=mask)


# 10. Reduction with axis on 2D tile (row-wise and col-wise)
@triton.jit
def row_sum_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Sum each row of a 2D matrix."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    row_sums = tl.sum(x, axis=1)  # reduce columns -> 1D
    tl.store(out_ptr + offs_m, row_sums, mask=offs_m < M)


# 11. Chained dot products (two matmuls in one kernel)
@triton.jit
def chained_dot_kernel(
    a_ptr, b_ptr, c_ptr, out_ptr,
    M, N, K, L,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cn, stride_cl,
    stride_om, stride_ol,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr,
):
    """Compute out = (A @ B) @ C, where A:[M,K], B:[K,N], C:[N,L]."""
    pid_m = tl.program_id(0)
    pid_l = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # First: AB = A @ B (accumulate over K)
    ab = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak))
        b = tl.load(b_ptr + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn))
        ab += tl.dot(a, b)
    # Second: out = AB @ C (accumulate over N)
    acc = tl.zeros((BLOCK_M, BLOCK_L), dtype=tl.float32)
    for n in range(0, N, BLOCK_N):
        # Need to reload AB columns... Actually this pattern requires storing AB.
        # Simpler: use element-wise accumulation for the second matmul
        c = tl.load(c_ptr + ((offs_n[:, None] + n) * stride_cn + offs_l[None, :] * stride_cl))
        # Extract columns of AB for this block of N
        ab_block = ab  # For BLOCK_N==N case
        acc += tl.dot(ab_block, c)
    out_ptrs = out_ptr + offs_m[:, None] * stride_ol + offs_l[None, :] * stride_ol
    tl.store(out_ptrs, acc)


# 12. tl.math.pow via repeated multiply (power function)
@triton.jit
def pow_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # x^3 via multiply chain
    tl.store(out_ptr + offs, x * x * x, mask=mask)


# 13. Complex index arithmetic (strided access with modular arithmetic)
@triton.jit
def strided_modular_kernel(x_ptr, out_ptr, n, stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # Modular index pattern (e.g., wrapping access)
    idx = (offs * stride) % n
    x = tl.load(x_ptr + idx, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)


# 14. tl.trans on 2D tensor
@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask)
    # Transpose and store
    xt = tl.trans(x)
    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(out_ptrs, xt, mask=out_mask)


# ============================================================
# Test runners
# ============================================================

def test_outer_product():
    M, N = 32, 32
    x = torch.randn(M, device='mps')
    y = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    outer_product_kernel[(1, 1)](x, y, out, M, N, BLOCK_M=32, BLOCK_N=32)
    ref = x[:, None] * y[None, :]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_where_2d():
    M, N = 32, 32
    x = torch.randn(M, N, device='mps')
    y = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    where_2d_kernel[(1, 1)](x, y, out, M, N, BLOCK_M=32, BLOCK_N=32)
    ref = torch.where(x > y, x, y)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_mean_var():
    N = 256
    x = torch.randn(N, device='mps')
    mean_out = torch.zeros(1, device='mps')
    var_out = torch.zeros(1, device='mps')
    mean_var_kernel[(1,)](x, mean_out, var_out, N, BLOCK=256)
    ref_mean = x.mean().item()
    ref_var = x.var(correction=0).item()
    mean_err = abs(mean_out.item() - ref_mean)
    var_err = abs(var_out.item() - ref_var)
    err = max(mean_err, var_err)
    return err < 1e-4, err


def test_gather():
    n = 1024
    x = torch.randn(n, device='mps')
    idx = torch.randint(0, n, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    mixed_dtype_kernel[grid](x, idx, out, n, BLOCK=256)
    ref = x[idx.long()]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_piecewise():
    n = 2048
    x = torch.randn(n, device='mps') * 3  # wider range to test all branches
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    piecewise_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.where(x > 0, x * x, torch.where(x <= -1.0, -x, x + 1.0))
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_cast_roundtrip():
    n = 2048
    x = torch.randn(n, device='mps') * 100  # large values to test truncation
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    cast_roundtrip_kernel[grid](x, out, n, BLOCK=256)
    ref = x.to(torch.int32).to(torch.float32)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_fp16_add():
    n = 2048
    x = torch.randn(n, device='mps', dtype=torch.float16)
    y = torch.randn(n, device='mps', dtype=torch.float16)
    out = torch.zeros(n, device='mps', dtype=torch.float16)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fp16_add_kernel[grid](x, y, out, n, BLOCK=256)
    ref = x + y
    err = (out - ref).abs().max().item()
    return err < 1e-3, err  # fp16 has lower precision


def test_large_block():
    n = 4096
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    large_block_kernel[grid](x, out, n, BLOCK=1024)
    ref = x * 2.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_multi_output():
    n = 2048
    x = torch.randn(n, device='mps')
    out_sin = torch.zeros(n, device='mps')
    out_cos = torch.zeros(n, device='mps')
    out_neg = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    multi_output_kernel[grid](x, out_sin, out_cos, out_neg, n, BLOCK=256)
    err = max(
        (out_sin - torch.sin(x)).abs().max().item(),
        (out_cos - torch.cos(x)).abs().max().item(),
        (out_neg - (-x)).abs().max().item(),
    )
    return err < 1e-5, err


def test_row_sum():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    row_sum_kernel[(triton.cdiv(M, 32),)](x, out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref = x.sum(dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err  # reduction accumulation tolerance


def test_pow3():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    pow_kernel[grid](x, out, n, BLOCK=256)
    ref = x ** 3
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_strided_modular():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    stride = 7
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    strided_modular_kernel[grid](x, out, n, stride, BLOCK=256)
    ref_idx = (torch.arange(n, device='mps') * stride) % n
    ref = x[ref_idx.long()]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_transpose():
    M, N = 32, 32
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(N, M, device='mps')
    transpose_kernel[(1, 1)](x, out, M, N, BLOCK_M=32, BLOCK_N=32)
    ref = x.T
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests ===\n")

    tests = [
        ("Outer Product (expand_dims)", test_outer_product),
        ("Where 2D", test_where_2d),
        ("Mean+Var (multi-reduce)", test_mean_var),
        ("Gather (int indices)", test_gather),
        ("Piecewise (nested where)", test_piecewise),
        ("Cast Roundtrip (f32->i32->f32)", test_cast_roundtrip),
        ("FP16 Add", test_fp16_add),
        ("Large Block (1024)", test_large_block),
        ("Multi-Output (sin/cos/neg)", test_multi_output),
        ("Row Sum (2D reduce axis=1)", test_row_sum),
        ("Pow3 (x*x*x)", test_pow3),
        ("Strided Modular Access", test_strided_modular),
        ("Transpose (tl.trans)", test_transpose),
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
