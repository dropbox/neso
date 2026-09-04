#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 24.

Targets: remaining gaps and stress tests:
- Larger 1D reduction (BLOCK=512, multi-SIMD)
- Chained reductions (mean then variance in same kernel)
- Nested scf.for (2 levels of loops)
- tl.where with float NaN handling
- Integer division and modulo
- tl.load with eviction_policy (cache hints)
- Large element-wise (n=16384, multi-block)
- Fused GELU activation (tanh approximation)
- Fused bias + GELU (transformer FFN pattern)
- Cosine similarity (dot / norms)
- Token embedding lookup (gather pattern)
- Triangular mask (causal attention mask generation)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Larger 1D reduction (BLOCK=512)
@triton.jit
def large_reduce_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Sum reduction with BLOCK=512 (multiple SIMD groups)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    total = tl.sum(x, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, total)


# 2. Chained reductions (mean then variance)
@triton.jit
def mean_var_kernel(x_ptr, mean_ptr, var_ptr, n, BLOCK: tl.constexpr):
    """Compute mean and variance in a single kernel."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(mean_ptr, mean)
        tl.store(var_ptr, var)


# 3. Nested scf.for (double loop)
@triton.jit
def nested_loop_kernel(x_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    """For each row, sum the row. out[i] = sum(x[i, :])."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        idx = start + offs
        mask = idx < N
        x = tl.load(x_ptr + row * N + idx, mask=mask, other=0.0)
        acc += x
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total)


# 4. tl.where with comparison (select negative values)
@triton.jit
def select_positive_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Replace negative values with 0 (manual ReLU via tl.where)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 5. Integer division and modulo
@triton.jit
def divmod_kernel(x_ptr, div_ptr, mod_ptr, divisor, n, BLOCK: tl.constexpr):
    """Compute x // divisor and x % divisor."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    d = x // divisor
    m = x % divisor
    tl.store(div_ptr + offs, d, mask=mask)
    tl.store(mod_ptr + offs, m, mask=mask)


# 6. Large element-wise (stress test multi-block)
@triton.jit
def large_ewise_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = a * b + a - b (large n, many blocks)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out = a * b + a - b
    tl.store(out_ptr + offs, out, mask=mask)


# 7. Fused GELU (tanh approximation)
@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # Constants
    k = 0.7978845608028654  # sqrt(2/pi)
    inner = k * (x + 0.044715 * x * x * x)
    # tanh(x) = 2*sigmoid(2x) - 1
    out = 0.5 * x * (1.0 + 2.0 * tl.sigmoid(2.0 * inner) - 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


# 8. Fused bias + GELU (FFN pattern)
@triton.jit
def bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    """out = GELU(x + bias) for each row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    xb = x + b
    k = 0.7978845608028654
    inner = k * (xb + 0.044715 * xb * xb * xb)
    out = 0.5 * xb * (1.0 + 2.0 * tl.sigmoid(2.0 * inner) - 1.0)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 9. Cosine similarity
@triton.jit
def cosine_sim_kernel(a_ptr, b_ptr, out_ptr, N, eps, BLOCK_N: tl.constexpr):
    """out[row] = dot(a[row], b[row]) / (||a[row]|| * ||b[row]|| + eps)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    a = tl.load(a_ptr + row * N + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + row * N + offs, mask=mask, other=0.0)
    dot = tl.sum(a * b, axis=0)
    norm_a = tl.sqrt(tl.sum(a * a, axis=0))
    norm_b = tl.sqrt(tl.sum(b * b, axis=0))
    sim = dot / (norm_a * norm_b + eps)
    tl.store(out_ptr + row, sim)


# 10. Token embedding lookup (gather)
@triton.jit
def embedding_kernel(indices_ptr, embed_ptr, out_ptr, vocab_size, dim,
                      BLOCK_D: tl.constexpr):
    """out[i, :] = embed[indices[i], :]."""
    token = tl.program_id(0)
    idx = tl.load(indices_ptr + token)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim
    emb = tl.load(embed_ptr + idx * dim + d_offs, mask=d_mask, other=0.0)
    tl.store(out_ptr + token * dim + d_offs, emb, mask=d_mask)


# 11. Triangular (causal) mask generation
@triton.jit
def causal_mask_kernel(out_ptr, N, BLOCK: tl.constexpr):
    """Generate lower-triangular mask: out[i,j] = 1.0 if j <= i else 0.0."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.where(offs <= row, 1.0, 0.0)
    tl.store(out_ptr + row * N + offs, vals, mask=mask)


# 12. Fused residual + dropout + scale (inference mode, p=0)
@triton.jit
def residual_scale_kernel(x_ptr, residual_ptr, out_ptr, scale, n,
                            BLOCK: tl.constexpr):
    """out = (x + residual) * scale."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.load(residual_ptr + offs, mask=mask)
    out = (x + r) * scale
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_large_reduce():
    n = 512
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    large_reduce_kernel[(1,)](x, out, n, BLOCK=512)
    ref = x.sum()
    err = abs(out.item() - ref.item())
    return err < 0.1, err


def test_mean_var():
    n = 256
    x = torch.randn(n, device='mps')
    mean_out = torch.zeros(1, device='mps')
    var_out = torch.zeros(1, device='mps')
    mean_var_kernel[(1,)](x, mean_out, var_out, n, BLOCK=256)
    ref_mean = x.mean().item()
    ref_var = x.var(correction=0).item()
    err = max(abs(mean_out.item() - ref_mean),
              abs(var_out.item() - ref_var))
    return err < 1e-3, err


def test_nested_loop():
    M, N = 32, 256
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    nested_loop_kernel[(M,)](x, out, M, N, BLOCK_N=64)
    ref = x.sum(dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-2, err


def test_select_positive():
    n = 1024
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    select_positive_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.relu(x)
    err = (out - ref).abs().max().item()
    return err < 1e-6, err


def test_divmod():
    n = 256
    x = torch.randint(1, 1000, (n,), device='mps', dtype=torch.int32)
    div_out = torch.zeros(n, device='mps', dtype=torch.int32)
    mod_out = torch.zeros(n, device='mps', dtype=torch.int32)
    divmod_kernel[(1,)](x, div_out, mod_out, 7, n, BLOCK=256)
    ref_div = x // 7
    ref_mod = x % 7
    err = max((div_out - ref_div).abs().max().item(),
              (mod_out - ref_mod).abs().max().item())
    return err == 0, float(err)


def test_large_ewise():
    n = 16384
    a = torch.randn(n, device='mps')
    b = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    large_ewise_kernel[grid](a, b, out, n, BLOCK=256)
    ref = a * b + a - b
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_gelu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    gelu_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.nn.functional.gelu(x, approximate='tanh')
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_bias_gelu():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    bias_gelu_kernel[(M,)](x, bias, out, N, BLOCK=128)
    xb = x + bias
    ref = torch.nn.functional.gelu(xb, approximate='tanh')
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_cosine_sim():
    M, N = 64, 128
    a = torch.randn(M, N, device='mps')
    b = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    cosine_sim_kernel[(M,)](a, b, out, N, 1e-8, BLOCK_N=128)
    ref = torch.nn.functional.cosine_similarity(a, b, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_embedding():
    seq_len = 64
    vocab_size = 1000
    dim = 128
    embed = torch.randn(vocab_size, dim, device='mps')
    indices = torch.randint(0, vocab_size, (seq_len,), device='mps', dtype=torch.int32)
    out = torch.zeros(seq_len, dim, device='mps')
    embedding_kernel[(seq_len,)](indices, embed, out, vocab_size, dim, BLOCK_D=128)
    ref = embed[indices.long()]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_causal_mask():
    N = 64
    out = torch.zeros(N, N, device='mps')
    causal_mask_kernel[(N,)](out, N, BLOCK=64)
    ref = torch.tril(torch.ones(N, N, device='mps'))
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_residual_scale():
    n = 2048
    x = torch.randn(n, device='mps')
    residual = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    residual_scale_kernel[grid](x, residual, out, 0.125, n, BLOCK=256)
    ref = (x + residual) * 0.125
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 24) ===\n")
    print("    (Stress tests, GELU, embeddings, causal mask)\n")

    tests = [
        ("Large Reduce (512)", test_large_reduce),
        ("Mean+Var (chained)", test_mean_var),
        ("Nested Loop (row sum)", test_nested_loop),
        ("Select Positive (where)", test_select_positive),
        ("IntDiv+Mod", test_divmod),
        ("Large Ewise (16384)", test_large_ewise),
        ("GELU (tanh approx)", test_gelu),
        ("Bias+GELU (FFN)", test_bias_gelu),
        ("Cosine Similarity", test_cosine_sim),
        ("Embedding Lookup", test_embedding),
        ("Causal Mask (tril)", test_causal_mask),
        ("Residual+Scale", test_residual_scale),
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
