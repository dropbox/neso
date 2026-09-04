#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 5.

Targets: integer bitwise ops, type casting, layer norm (mean+var),
         3D grid, in-place update, large blocks, comparison ops,
         modular arithmetic, cross-iteration accumulator, multiple stores.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Integer bitwise operations (and, or, xor, shift)
@triton.jit
def bitwise_kernel(a_ptr, b_ptr, out_and_ptr, out_or_ptr, out_xor_ptr,
                   out_shl_ptr, out_shr_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_and_ptr + offs, a & b, mask=mask)
    tl.store(out_or_ptr + offs, a | b, mask=mask)
    tl.store(out_xor_ptr + offs, a ^ b, mask=mask)
    tl.store(out_shl_ptr + offs, a << 2, mask=mask)
    tl.store(out_shr_ptr + offs, a >> 1, mask=mask)


# 2. Type casting: int32 -> float32 and float32 -> int32
@triton.jit
def cast_i2f_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x.to(tl.float32)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def cast_f2i_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x.to(tl.int32)
    tl.store(out_ptr + offs, out, mask=mask)


# 3. Layer norm: computes mean + variance + normalization in one kernel
@triton.jit
def layernorm_kernel(x_ptr, out_ptr, weight_ptr, bias_ptr,
                     row_stride, n_cols, eps,
                     BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    # Mean
    mean = tl.sum(x, axis=0) / n_cols
    # Variance
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = xc * rstd
    # Affine transform
    w = tl.load(weight_ptr + offs, mask=mask)
    b = tl.load(bias_ptr + offs, mask=mask)
    out = xn * w + b
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 4. 3D grid kernel (uses program_id(0), (1), (2))
@triton.jit
def add_3d_kernel(x_ptr, y_ptr, out_ptr, D0, D1, D2,
                  BLOCK: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    idx = pid0 * D1 * D2 + pid1 * D2 + pid2
    offs = idx * BLOCK + tl.arange(0, BLOCK)
    n = D0 * D1 * D2 * BLOCK
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# 5. In-place update (load from buffer, modify, store back)
@triton.jit
def inplace_scale_kernel(x_ptr, scale, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(x_ptr + offs, x * scale, mask=mask)


# 6. Large block (1024 elements per block)
@triton.jit
def large_block_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


# 7. Comparison ops: generate boolean mask then use it
@triton.jit
def threshold_count_kernel(x_ptr, out_ptr, threshold, n, BLOCK: tl.constexpr):
    """Count elements above threshold (reduction of boolean mask)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    above = (x > threshold).to(tl.float32)
    count = tl.sum(above, axis=0)
    if pid == 0:
        tl.store(out_ptr, count)


# 8. Modular arithmetic (integer mod, div)
@triton.jit
def mod_div_kernel(x_ptr, out_mod_ptr, out_div_ptr, divisor, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_mod_ptr + offs, x % divisor, mask=mask)
    tl.store(out_div_ptr + offs, x // divisor, mask=mask)


# 9. Chained element-wise: polynomial evaluation (a*x^3 + b*x^2 + c*x + d)
@triton.jit
def polynomial_kernel(x_ptr, out_ptr, a, b, c, d, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Horner's method: ((a*x + b)*x + c)*x + d
    out = ((a * x + b) * x + c) * x + d
    tl.store(out_ptr + offs, out, mask=mask)


# 10. Multiple output tensors from one kernel
@triton.jit
def sincos_kernel(x_ptr, sin_ptr, cos_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(sin_ptr + offs, tl.sin(x), mask=mask)
    tl.store(cos_ptr + offs, tl.cos(x), mask=mask)


# 11. Row-wise mean + variance (two reductions in one kernel)
@triton.jit
def mean_var_kernel(x_ptr, mean_ptr, var_ptr, M, N,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    row_mean = tl.sum(x, axis=1) / N
    xc = x - row_mean[:, None]
    row_var = tl.sum(xc * xc, axis=1) / N
    tl.store(mean_ptr + offs_m, row_mean, mask=offs_m < M)
    tl.store(var_ptr + offs_m, row_var, mask=offs_m < M)


# 12. Fused multiply-add chain (tests FMA optimization)
@triton.jit
def fma_chain_kernel(a_ptr, b_ptr, c_ptr, d_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    c = tl.load(c_ptr + offs, mask=mask)
    d = tl.load(d_ptr + offs, mask=mask)
    # a*b + c*d
    out = a * b + c * d
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_bitwise():
    n = 1024
    a = torch.randint(0, 256, (n,), device='mps', dtype=torch.int32)
    b = torch.randint(0, 256, (n,), device='mps', dtype=torch.int32)
    out_and = torch.zeros(n, device='mps', dtype=torch.int32)
    out_or = torch.zeros(n, device='mps', dtype=torch.int32)
    out_xor = torch.zeros(n, device='mps', dtype=torch.int32)
    out_shl = torch.zeros(n, device='mps', dtype=torch.int32)
    out_shr = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    bitwise_kernel[grid](a, b, out_and, out_or, out_xor, out_shl, out_shr, n, BLOCK=256)
    errs = []
    errs.append((out_and - (a & b)).abs().max().item())
    errs.append((out_or - (a | b)).abs().max().item())
    errs.append((out_xor - (a ^ b)).abs().max().item())
    errs.append((out_shl - (a << 2)).abs().max().item())
    errs.append((out_shr - (a >> 1)).abs().max().item())
    err = max(errs)
    return err == 0, err


def test_cast_int_to_float():
    n = 1024
    x = torch.randint(-100, 100, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    cast_i2f_kernel[grid](x, out, n, BLOCK=256)
    ref = x.float()
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_cast_float_to_int():
    n = 1024
    x = torch.randint(-100, 100, (n,), device='mps').float()
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    cast_f2i_kernel[grid](x, out, n, BLOCK=256)
    ref = x.int()
    err = (out - ref).abs().max().item()
    return err == 0, err


def test_layernorm():
    M, N = 16, 128
    x = torch.randn(M, N, device='mps')
    w = torch.randn(N, device='mps')
    b = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    layernorm_kernel[(M,)](x, out, w, b, N, N, 1e-5, BLOCK_SIZE=128)
    # Reference
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, correction=0)
    ref = (x - mean) / torch.sqrt(var + 1e-5) * w + b
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_3d_grid():
    D0, D1, D2 = 4, 3, 2
    BLOCK = 32
    total = D0 * D1 * D2 * BLOCK
    x = torch.randn(total, device='mps')
    y = torch.randn(total, device='mps')
    out = torch.zeros(total, device='mps')
    add_3d_kernel[(D0, D1, D2)](x, y, out, D0, D1, D2, BLOCK=BLOCK)
    ref = x + y
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_inplace_scale():
    n = 2048
    x = torch.randn(n, device='mps')
    ref = x * 2.5
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    inplace_scale_kernel[grid](x, 2.5, n, BLOCK=256)
    err = (x - ref).abs().max().item()
    return err < 1e-5, err


def test_large_block():
    n = 4096
    x = torch.randn(n, device='mps')
    y = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    large_block_add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    ref = x + y
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_threshold_count():
    n = 256
    x = torch.randn(n, device='mps')
    threshold = 0.0
    out = torch.zeros(1, device='mps')
    threshold_count_kernel[(1,)](x, out, threshold, n, BLOCK=256)
    ref = (x > threshold).float().sum()
    err = abs(out.item() - ref.item())
    return err < 1.0, err  # allow small rounding


def test_mod_div():
    n = 1024
    x = torch.randint(1, 1000, (n,), device='mps', dtype=torch.int32)
    divisor = 7
    out_mod = torch.zeros(n, device='mps', dtype=torch.int32)
    out_div = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    mod_div_kernel[grid](x, out_mod, out_div, divisor, n, BLOCK=256)
    ref_mod = x % divisor
    ref_div = x // divisor
    err_mod = (out_mod - ref_mod).abs().max().item()
    err_div = (out_div - ref_div).abs().max().item()
    err = max(err_mod, err_div)
    return err == 0, err


def test_polynomial():
    n = 2048
    x = torch.randn(n, device='mps')
    a, b, c, d = 0.5, -1.2, 3.0, -0.7
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    polynomial_kernel[grid](x, out, a, b, c, d, n, BLOCK=256)
    ref = ((a * x + b) * x + c) * x + d
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_sincos():
    n = 2048
    x = torch.randn(n, device='mps')
    sin_out = torch.zeros(n, device='mps')
    cos_out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    sincos_kernel[grid](x, sin_out, cos_out, n, BLOCK=256)
    ref_sin = torch.sin(x)
    ref_cos = torch.cos(x)
    err_sin = (sin_out - ref_sin).abs().max().item()
    err_cos = (cos_out - ref_cos).abs().max().item()
    err = max(err_sin, err_cos)
    return err < 1e-4, err


def test_mean_var():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    mean_out = torch.zeros(M, device='mps')
    var_out = torch.zeros(M, device='mps')
    mean_var_kernel[(triton.cdiv(M, 32),)](x, mean_out, var_out, M, N, BLOCK_M=32, BLOCK_N=64)
    ref_mean = x.mean(dim=1)
    ref_var = x.var(dim=1, correction=0)
    err_mean = (mean_out - ref_mean).abs().max().item()
    err_var = (var_out - ref_var).abs().max().item()
    err = max(err_mean, err_var)
    return err < 1e-3, err


def test_fma_chain():
    n = 2048
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    c = torch.randn(n, device='mps')
    d = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fma_chain_kernel[grid](a, b, c, d, out, n, BLOCK=256)
    ref = a * b + c * d
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 5) ===\n")

    tests = [
        ("Bitwise Ops (and/or/xor/shl/shr)", test_bitwise),
        ("Cast Int->Float", test_cast_int_to_float),
        ("Cast Float->Int", test_cast_float_to_int),
        ("Layer Norm (mean+var+affine)", test_layernorm),
        ("3D Grid (program_id 0,1,2)", test_3d_grid),
        ("In-Place Scale", test_inplace_scale),
        ("Large Block (1024)", test_large_block),
        ("Threshold Count (bool reduce)", test_threshold_count),
        ("Mod/Div (integer)", test_mod_div),
        ("Polynomial (Horner chain)", test_polynomial),
        ("Sin+Cos (dual output)", test_sincos),
        ("Mean+Variance (dual reduce)", test_mean_var),
        ("FMA Chain (a*b+c*d)", test_fma_chain),
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
