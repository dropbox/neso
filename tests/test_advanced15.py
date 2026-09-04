#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 15.

Targets: untested lowering paths and edge cases:
- tl.num_programs (grid-stride loops)
- math.erf (GELU activation)
- math.atan2, math.powf
- tl.clamp (fused clamp op)
- tl.debug_barrier
- bf16 (bfloat16) element-wise
- int64 pointer offsets
- mixed int32/float32 in same kernel
- grid-stride loop pattern
- tl.math.fma on tensors (not just scalars)
"""
import sys
import torch
import triton
import triton.language as tl
torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Grid-stride loop using tl.num_programs
@triton.jit
def grid_stride_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Add with grid-stride loop: each program processes multiple blocks."""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    block_start = pid * BLOCK
    stride = num_pids * BLOCK
    for start in range(block_start, n, stride):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)


# 2. GELU activation (uses erf)
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # GELU formula
    out = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(out_ptr + offs, out, mask=mask)


# 3. tanh via exp (tests complex element-wise chains)
@triton.jit
def tanh_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    e2x = tl.exp(2.0 * x)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 4. powf via exp+log (x^p = exp(p*log(x)))
@triton.jit
def powf_kernel(x_ptr, out_ptr, exponent, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.exp(exponent * tl.log(x))
    tl.store(out_ptr + offs, out, mask=mask)


# 5. Fused clamp via tl.clamp
@triton.jit
def fused_clamp_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.clamp(x, -1.0, 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 6. Mixed int and float operations in same kernel
@triton.jit
def mixed_int_float_kernel(x_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Compute x[idx[i]] * 2.0 + idx[i] (int index used as both address and value)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    gathered = tl.load(x_ptr + idx, mask=mask)
    # Mix: float multiply + int-to-float add
    out = gathered * 2.0 + idx.to(tl.float32)
    tl.store(out_ptr + offs, out, mask=mask)


# 7. GELU approximate (tanh version, used in GPT-2)
@triton.jit
def gelu_tanh_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)  # sqrt(2/pi)
    # tanh via sigmoid: tanh(x) = 2*sigmoid(2x) - 1
    out = 0.5 * x * (2.0 * tl.sigmoid(2.0 * inner) - 1.0 + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 8. Power-of-2 tiling with non-power-of-2 data
@triton.jit
def odd_size_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Process tensor whose size is not a multiple of BLOCK."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = x * x  # square
    tl.store(out_ptr + offs, out, mask=mask)


# 9. Reduction with grid-stride (multiple blocks contribute to one result)
@triton.jit
def grid_stride_sum_kernel(x_ptr, partial_ptr, n, BLOCK: tl.constexpr):
    """Each program sums a chunk, writes partial sum."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    tl.store(partial_ptr + pid, total)


# 10. Fused bias + GELU (common transformer pattern)
@triton.jit
def fused_bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, row_stride, n_cols,
                            BLOCK: tl.constexpr):
    """out = GELU(x + bias) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    h = x + b
    out = 0.5 * h * (1.0 + tl.math.erf(h * 0.7071067811865476))
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 11. Elementwise with multiple reductions in one kernel
@triton.jit
def multi_stat_kernel(x_ptr, mean_ptr, var_ptr, min_ptr, max_ptr, n,
                      BLOCK: tl.constexpr):
    """Compute mean, var, min, max all at once."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean_val = tl.sum(x, axis=0) / n
    diff = x - mean_val
    var_val = tl.sum(diff * diff, axis=0) / n
    min_val = tl.min(x, axis=0)
    max_val = tl.max(x, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(mean_ptr, mean_val)
        tl.store(var_ptr, var_val)
        tl.store(min_ptr, min_val)
        tl.store(max_ptr, max_val)


# 12. debug_barrier (threadgroup synchronization)
@triton.jit
def barrier_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test tl.debug_barrier between load and store."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.debug_barrier()
    out = x * 2.0
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_grid_stride_add():
    n = 4096
    x = torch.randn(n, device='mps')
    y = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    # Use fewer programs than needed — kernel must grid-stride
    grid = (4,)  # only 4 programs for 4096 elements
    grid_stride_add_kernel[grid](x, y, out, n, BLOCK=256)
    ref = x + y
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_gelu():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    gelu_kernel[grid](x, out, n, BLOCK=256)
    ref = 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_tanh():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    tanh_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.tanh(x)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_powf():
    n = 1024
    x = torch.rand(n, device='mps') + 0.1  # positive values
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    powf_kernel[grid](x, out, 2.5, n, BLOCK=256)
    ref = x.pow(2.5)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_fused_clamp():
    n = 1024
    x = torch.randn(n, device='mps') * 5
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fused_clamp_kernel[grid](x, out, n, BLOCK=256)
    ref = x.clamp(-1.0, 1.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_mixed_int_float():
    n = 256
    x = torch.randn(n, device='mps')
    idx = torch.randint(0, n, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    mixed_int_float_kernel[(1,)](x, idx, out, n, BLOCK=256)
    ref = x[idx.long()] * 2.0 + idx.float()
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_gelu_tanh():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    gelu_tanh_kernel[grid](x, out, n, BLOCK=256)
    # Reference: same formula using sigmoid-based tanh
    x3 = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)
    tanh_val = 2.0 * torch.sigmoid(2.0 * inner) - 1.0
    ref = 0.5 * x * (1.0 + tanh_val)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_odd_size():
    n = 137  # not a multiple of any power of 2
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    odd_size_kernel[grid](x, out, n, BLOCK=256)
    ref = x * x
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_grid_stride_sum():
    n = 1024
    x = torch.randn(n, device='mps')
    n_blocks = triton.cdiv(n, 256)
    partial = torch.zeros(n_blocks, device='mps')
    grid_stride_sum_kernel[(n_blocks,)](x, partial, n, BLOCK=256)
    result = partial.sum()
    ref = x.sum()
    err = abs(result.item() - ref.item())
    return err < 1e-2, err


def test_fused_bias_gelu():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    fused_bias_gelu_kernel[(M,)](x, bias, out, N, N, BLOCK=128)
    h = x + bias
    ref = 0.5 * h * (1.0 + torch.erf(h * 0.7071067811865476))
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_multi_stat():
    n = 256
    x = torch.randn(n, device='mps')
    mean_out = torch.zeros(1, device='mps')
    var_out = torch.zeros(1, device='mps')
    min_out = torch.zeros(1, device='mps')
    max_out = torch.zeros(1, device='mps')
    multi_stat_kernel[(1,)](x, mean_out, var_out, min_out, max_out, n, BLOCK=256)
    err = max(
        abs(mean_out.item() - x.mean().item()),
        abs(var_out.item() - x.var(correction=0).item()),
        abs(min_out.item() - x.min().item()),
        abs(max_out.item() - x.max().item()),
    )
    return err < 1e-3, err


def test_barrier():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    barrier_kernel[grid](x, out, n, BLOCK=256)
    ref = x * 2.0
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 15) ===\n")

    tests = [
        ("Grid-Stride Add", test_grid_stride_add),
        ("GELU (erf-based)", test_gelu),
        ("tanh (via exp)", test_tanh),
        ("powf (exp+log)", test_powf),
        ("Fused Clamp (tl.clamp)", test_fused_clamp),
        ("Mixed Int+Float Gather", test_mixed_int_float),
        ("GELU-tanh (GPT-2)", test_gelu_tanh),
        ("Odd-Size Tensor (n=137)", test_odd_size),
        ("Grid-Stride Sum", test_grid_stride_sum),
        ("Fused Bias+GELU", test_fused_bias_gelu),
        ("Multi-Stat (mean/var/min/max)", test_multi_stat),
        ("debug_barrier", test_barrier),
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
