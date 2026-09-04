#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 13.

Targets: 3D grids, very large tensors, nested control flow,
multi-head attention through codegen, integer tile operations,
and edge cases in pointer arithmetic.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. 3D grid kernel (batched operation)
@triton.jit
def batched_add_3d_kernel(x_ptr, y_ptr, out_ptr, B, M, N,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """out[b, m, n] = x[b, m, n] + y[b, m, n] with 3D grid."""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base = pid_b * M * N
    ptrs = base + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask)
    y = tl.load(y_ptr + ptrs, mask=mask)
    tl.store(out_ptr + ptrs, x + y, mask=mask)


# 2. Large tensor (1M+ elements)
@triton.jit
def large_relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x > 0, x, 0.0), mask=mask)


# 3. Nested scf.if inside scf.for (piecewise loop)
@triton.jit
def piecewise_accumulate_kernel(x_ptr, out_ptr, n, threshold,
                                 BLOCK: tl.constexpr):
    """Accumulate with different weights based on sign."""
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        chunk_offs = start + offs
        mask = chunk_offs < n
        x = tl.load(x_ptr + chunk_offs, mask=mask, other=0.0)
        # Piecewise: positive gets weight 2, negative gets weight 0.5
        pos_mask = x > 0.0
        weighted = tl.where(pos_mask, x * 2.0, x * 0.5)
        acc += weighted
    total = tl.sum(acc, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 4. Batch matrix-vector multiply (batched matmul building block)
@triton.jit
def batch_matvec_kernel(A_ptr, x_ptr, out_ptr, B, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """out[b, m] = sum_n(A[b, m, n] * x[b, n])."""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    # Load A block [BLOCK_M, BLOCK_N]
    a_ptrs = A_ptr + pid_b * M * N + offs_m[:, None] * N + offs_n[None, :]
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    # Load x block [BLOCK_N]
    x = tl.load(x_ptr + pid_b * N + offs_n, mask=mask_n, other=0.0)
    # Element-wise multiply and reduce
    prod = a * x[None, :]
    result = tl.sum(prod, axis=1)  # [BLOCK_M]
    tl.store(out_ptr + pid_b * M + offs_m, result, mask=mask_m)


# 5. Integer tile division and modulo
@triton.jit
def divmod_kernel(x_ptr, div_ptr, mod_ptr, divisor, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(div_ptr + offs, x // divisor, mask=mask)
    tl.store(mod_ptr + offs, x % divisor, mask=mask)


# 6. Multiple sequential reductions (mean, std, skewness)
@triton.jit
def statistics_kernel(x_ptr, mean_ptr, std_ptr, n, BLOCK: tl.constexpr):
    """Compute mean and std in one kernel."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n
    std = tl.sqrt(var)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(mean_ptr, mean)
        tl.store(std_ptr, std)


# 7. Multi-head attention score computation
@triton.jit
def attention_score_kernel(q_ptr, k_ptr, score_ptr, seq_len, head_dim, scale,
                           BLOCK_SEQ: tl.constexpr, BLOCK_D: tl.constexpr):
    """Compute Q @ K^T for one head (one row of Q against all K rows)."""
    pid_q = tl.program_id(0)  # which query position
    pid_k = tl.program_id(1)  # which key position
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim
    # Load q[pid_q, :] and k[pid_k, :]
    q = tl.load(q_ptr + pid_q * head_dim + offs_d, mask=d_mask, other=0.0)
    k = tl.load(k_ptr + pid_k * head_dim + offs_d, mask=d_mask, other=0.0)
    # Dot product
    dot = tl.sum(q * k, axis=0) * scale
    pid = tl.program_id(0)
    tl.store(score_ptr + pid_q * seq_len + pid_k, dot)


# 8. Fused element-wise with multiple outputs
@triton.jit
def multi_output_kernel(x_ptr, sin_ptr, cos_ptr, exp_ptr, n,
                        BLOCK: tl.constexpr):
    """Compute sin(x), cos(x), exp(x) in one kernel."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(sin_ptr + offs, tl.sin(x), mask=mask)
    tl.store(cos_ptr + offs, tl.cos(x), mask=mask)
    tl.store(exp_ptr + offs, tl.exp(x), mask=mask)


# 9. Copy with padding (extend tensor boundaries)
@triton.jit
def padded_copy_kernel(x_ptr, out_ptr, n_in, n_out, pad_val,
                       BLOCK: tl.constexpr):
    """Copy with right-padding: out[:n_in] = x, out[n_in:n_out] = pad_val."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    in_range = offs < n_in
    x = tl.load(x_ptr + offs, mask=in_range, other=pad_val)
    tl.store(out_ptr + offs, x, mask=mask)


# 10. Multi-pass reduction (for data larger than one block)
@triton.jit
def multi_pass_max_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Find global max across the entire array."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    block_max = tl.max(x, axis=0)
    tl.atomic_max(out_ptr, block_max)


# ============================================================
# Test runners
# ============================================================

def test_batched_add_3d():
    B, M, N = 4, 16, 32
    x = torch.randn(B, M, N, device='mps')
    y = torch.randn(B, M, N, device='mps')
    out = torch.zeros(B, M, N, device='mps')
    grid = (B, triton.cdiv(M, 16), triton.cdiv(N, 32))
    batched_add_3d_kernel[grid](x.reshape(-1), y.reshape(-1), out.reshape(-1),
                                 B, M, N, BLOCK_M=16, BLOCK_N=32)
    ref = x + y
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_large_relu():
    n = 2_000_000
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    large_relu_kernel[grid](x, out, n, BLOCK=1024)
    ref = torch.relu(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_piecewise_accumulate():
    n = 512
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    threshold = 0.0
    piecewise_accumulate_kernel[(1,)](x, out, n, threshold, BLOCK=128)
    # Reference: positive * 2 + negative * 0.5
    ref = torch.where(x > 0, x * 2, x * 0.5).sum()
    err = abs(out.item() - ref.item())
    return err < 1e-2, err


def test_batch_matvec():
    B, M, N = 4, 16, 32
    A = torch.randn(B, M, N, device='mps')
    x = torch.randn(B, N, device='mps')
    out = torch.zeros(B, M, device='mps')
    grid = (B, triton.cdiv(M, 16))
    batch_matvec_kernel[grid](A.reshape(-1), x.reshape(-1), out.reshape(-1),
                               B, M, N, BLOCK_M=16, BLOCK_N=32)
    ref = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_divmod():
    n = 1024
    x = torch.randint(0, 1000, (n,), device='mps', dtype=torch.int32)
    divisor = 7
    div_out = torch.zeros(n, device='mps', dtype=torch.int32)
    mod_out = torch.zeros(n, device='mps', dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    divmod_kernel[grid](x, div_out, mod_out, divisor, n, BLOCK=256)
    ref_div = x // divisor
    ref_mod = x % divisor
    err = max((div_out - ref_div).abs().max().item(),
              (mod_out - ref_mod).abs().max().item())
    return err == 0, err


def test_statistics():
    n = 256
    x = torch.randn(n, device='mps')
    mean_out = torch.zeros(1, device='mps')
    std_out = torch.zeros(1, device='mps')
    statistics_kernel[(1,)](x, mean_out, std_out, n, BLOCK=256)
    ref_mean = x.mean()
    ref_std = x.std(correction=0)
    err = max(abs(mean_out.item() - ref_mean.item()),
              abs(std_out.item() - ref_std.item()))
    return err < 1e-3, err


def test_attention_score():
    seq_len = 16
    head_dim = 32
    scale = 1.0 / (head_dim ** 0.5)
    q = torch.randn(seq_len, head_dim, device='mps')
    k = torch.randn(seq_len, head_dim, device='mps')
    score = torch.zeros(seq_len, seq_len, device='mps')
    grid = (seq_len, seq_len)
    attention_score_kernel[grid](q, k, score, seq_len, head_dim, scale,
                                  BLOCK_SEQ=1, BLOCK_D=32)
    ref = (q @ k.T) * scale
    err = (score - ref).abs().max().item()
    return err < 1e-4, err


def test_multi_output():
    n = 1024
    x = torch.randn(n, device='mps') * 0.5  # keep small to avoid exp overflow
    sin_out = torch.zeros(n, device='mps')
    cos_out = torch.zeros(n, device='mps')
    exp_out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    multi_output_kernel[grid](x, sin_out, cos_out, exp_out, n, BLOCK=256)
    err = max((sin_out - x.sin()).abs().max().item(),
              (cos_out - x.cos()).abs().max().item(),
              (exp_out - x.exp()).abs().max().item())
    return err < 1e-5, err


def test_padded_copy():
    n_in = 200
    n_out = 256
    pad_val = -999.0
    x = torch.randn(n_in, device='mps')
    out = torch.zeros(n_out, device='mps')
    padded_copy_kernel[(1,)](x, out, n_in, n_out, pad_val, BLOCK=256)
    ref = torch.full((n_out,), pad_val, device='mps')
    ref[:n_in] = x
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_multi_pass_max():
    n = 4096
    # Use positive values only: IEEE-754 float atomic max only works reliably
    # for same-sign values (the uint reinterpretation trick)
    x = torch.rand(n, device='mps') + 0.1  # all positive
    out = torch.zeros(1, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    multi_pass_max_kernel[grid](x, out, n, BLOCK=256)
    ref = x.max()
    err = abs(out.item() - ref.item())
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 13) ===\n")

    tests = [
        ("3D Grid (batched add)", test_batched_add_3d),
        ("Large Tensor (2M ReLU)", test_large_relu),
        ("Piecewise Accumulate", test_piecewise_accumulate),
        ("Batch Matvec", test_batch_matvec),
        ("Integer DivMod", test_divmod),
        ("Statistics (mean+std)", test_statistics),
        ("Attention Score (Q@K^T)", test_attention_score),
        ("Multi-Output (sin/cos/exp)", test_multi_output),
        ("Padded Copy", test_padded_copy),
        ("Multi-Pass Max (atomic)", test_multi_pass_max),
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
