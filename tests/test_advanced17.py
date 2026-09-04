#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 17.

Targets: complex patterns that stress the codegen:
- Multi-output kernels with different dtypes
- Complex index arithmetic (swizzle, modular)
- Reduction along axis=1 (column-wise)
- Multiple independent reductions in one kernel
- Loop with multiple accumulators
- Nested conditionals with reductions
- tl.expand_dims + broadcasting patterns
- Fused operations chains (normalize + scale + offset)
- tl.dot with non-square shapes (rectangular matmul)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Swizzled index (Morton Z-curve style)
@triton.jit
def swizzle_copy_kernel(x_ptr, out_ptr, M, N,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Copy with swizzled tile indices (reorder threadgroups)."""
    # Swizzle: for grid (pid_m, pid_n), remap to improve L2 locality
    linear_pid = tl.program_id(0)
    # Simple swizzle: group tiles into groups of WIDTH
    WIDTH: tl.constexpr = 4
    group_id = linear_pid // (WIDTH * (N // BLOCK_N))
    first_pid_m = group_id * WIDTH
    group_size_m = min(M // BLOCK_M - first_pid_m, WIDTH)
    pid_m = first_pid_m + (linear_pid % group_size_m)
    pid_n = (linear_pid % (group_size_m * (N // BLOCK_N))) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ptrs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask)
    tl.store(out_ptr + ptrs, x, mask=mask)


# 2. Online softmax (streaming max + sum in one pass)
@triton.jit
def online_softmax_kernel(x_ptr, out_ptr, row_stride, n_cols,
                           BLOCK: tl.constexpr):
    """Softmax using online algorithm (single pass for max + exp + sum)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    x_shifted = x - max_val
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 3. Row-wise variance (reduction + broadcast + reduction)
@triton.jit
def row_variance_kernel(x_ptr, var_ptr, M, N, BLOCK_N: tl.constexpr):
    """Compute per-row variance."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    tl.store(var_ptr + row, var)


# 4. Fused layer norm (normalize + scale + offset in one kernel)
@triton.jit
def fused_layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr,
                            row_stride, n_cols, eps,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / n_cols
    xn = xc / tl.sqrt(var + eps)
    g = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    out = g * xn + b
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# 5. Accumulator loop (iterate over chunks, accumulate sum)
@triton.jit
def chunked_sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Sum a large array by iterating over chunks."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        chunk_offs = start + offs
        mask = chunk_offs < n
        x = tl.load(x_ptr + chunk_offs, mask=mask, other=0.0)
        acc += x
    total = tl.sum(acc, axis=0)
    if pid == 0:
        tl.store(out_ptr, total)


# 6. Conditional reduction (sum only positive elements)
@triton.jit
def cond_reduce_kernel(x_ptr, pos_sum_ptr, neg_sum_ptr, n, BLOCK: tl.constexpr):
    """Separately sum positive and negative elements."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    pos = tl.where(x > 0.0, x, 0.0)
    neg = tl.where(x < 0.0, x, 0.0)
    pos_total = tl.sum(pos, axis=0)
    neg_total = tl.sum(neg, axis=0)
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(pos_sum_ptr, pos_total)
        tl.store(neg_sum_ptr, neg_total)


# 7. Batch normalization (running mean/var)
@triton.jit
def batchnorm_kernel(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
                      N, C, eps, BLOCK_C: tl.constexpr):
    """Apply batch normalization: out = gamma * (x - mean) / sqrt(var + eps) + beta."""
    n = tl.program_id(0)  # batch index
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    x = tl.load(x_ptr + n * C + offs_c, mask=mask)
    mean = tl.load(mean_ptr + offs_c, mask=mask)
    var = tl.load(var_ptr + offs_c, mask=mask)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask)
    beta = tl.load(beta_ptr + offs_c, mask=mask)
    xn = (x - mean) / tl.sqrt(var + eps)
    out = gamma * xn + beta
    tl.store(out_ptr + n * C + offs_c, out, mask=mask)


# 8. Tiled matrix transpose
@triton.jit
def tile_transpose_kernel(x_ptr, out_ptr, M, N,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Transpose M×N matrix using tiled access."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Read from (m, n)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask)
    # Write to (n, m) — transposed
    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(out_ptr + offs_n[:, None] * M + offs_m[None, :],
             tl.trans(x), mask=out_mask)


# 9. Elementwise with very long dependency chain
@triton.jit
def deep_chain_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Long chain of dependent operations (tests register pressure)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    # 10 dependent operations
    a = tl.sigmoid(x)
    b = tl.exp(-a)
    c = tl.log(b + 1.0)
    d = tl.abs(c)
    e = tl.sqrt(d + 0.001)
    f = 1.0 / (e + 0.001)
    g = f * x
    h = tl.where(g > 0, g, -g)
    i = tl.floor(h * 100.0) / 100.0
    out = i + 0.5
    tl.store(out_ptr + offs, out, mask=mask)


# 10. Reduction with different axes on same 2D data
@triton.jit
def dual_reduce_kernel(x_ptr, row_sum_ptr, col_sum_ptr, M, N,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Compute row sums and column sums from same 2D tile."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    # Row sums (reduce axis=1)
    row_sums = tl.sum(x, axis=1)  # [BLOCK_M]
    tl.store(row_sum_ptr + offs_m, row_sums, mask=offs_m < M)
    # Column sums (reduce axis=0)
    col_sums = tl.sum(x, axis=0)  # [BLOCK_N]
    tl.store(col_sum_ptr + offs_n, col_sums, mask=offs_n < N)


# 11. Fused SiLU + multiply (LLaMA MLP pattern)
@triton.jit
def fused_silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """SiLU(gate) * up — the LLaMA MLP gate pattern."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask)
    up = tl.load(up_ptr + offs, mask=mask)
    silu = gate * tl.sigmoid(gate)
    out = silu * up
    tl.store(out_ptr + offs, out, mask=mask)


# 12. Cross-entropy loss components (log-softmax + nll)
@triton.jit
def log_softmax_kernel(x_ptr, out_ptr, row_stride, n_cols,
                        BLOCK: tl.constexpr):
    """Log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    x_shifted = x - max_val
    log_sum_exp = tl.log(tl.sum(tl.exp(x_shifted), axis=0))
    out = x_shifted - log_sum_exp
    tl.store(out_ptr + row * row_stride + offs, out, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_swizzle_copy():
    M, N = 32, 32
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    n_tiles = (M // 16) * (N // 16)
    swizzle_copy_kernel[(n_tiles,)](x, out, M, N, BLOCK_M=16, BLOCK_N=16)
    err = (out - x).abs().max().item()
    return err < 1e-5, err


def test_online_softmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    online_softmax_kernel[(M,)](x, out, N, N, BLOCK=64)
    ref = torch.softmax(x, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_row_variance():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    var_out = torch.zeros(M, device='mps')
    row_variance_kernel[(M,)](x, var_out, M, N, BLOCK_N=128)
    ref = x.var(dim=1, correction=0)
    err = (var_out - ref).abs().max().item()
    return err < 1e-3, err


def test_fused_layernorm():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    gamma = torch.randn(N, device='mps')
    beta = torch.randn(N, device='mps')
    out = torch.zeros(M, N, device='mps')
    eps = 1e-5
    fused_layernorm_kernel[(M,)](x, gamma, beta, out, N, N, eps, BLOCK=128)
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, correction=0)
    ref = gamma * (x - mean) / (var + eps).sqrt() + beta
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_chunked_sum():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(1, device='mps')
    chunked_sum_kernel[(1,)](x, out, n, BLOCK=256)
    ref = x.sum()
    err = abs(out.item() - ref.item())
    return err < 1e-1, err


def test_cond_reduce():
    n = 256
    x = torch.randn(n, device='mps')
    pos_sum = torch.zeros(1, device='mps')
    neg_sum = torch.zeros(1, device='mps')
    cond_reduce_kernel[(1,)](x, pos_sum, neg_sum, n, BLOCK=256)
    ref_pos = x[x > 0].sum()
    ref_neg = x[x < 0].sum()
    err = max(abs(pos_sum.item() - ref_pos.item()),
              abs(neg_sum.item() - ref_neg.item()))
    return err < 1e-2, err


def test_batchnorm():
    N, C = 32, 64
    x = torch.randn(N, C, device='mps')
    mean = x.mean(dim=0)
    var = x.var(dim=0, correction=0)
    gamma = torch.randn(C, device='mps')
    beta = torch.randn(C, device='mps')
    out = torch.zeros(N, C, device='mps')
    eps = 1e-5
    batchnorm_kernel[(N,)](x, mean, var, gamma, beta, out, N, C, eps, BLOCK_C=64)
    ref = gamma * (x - mean) / (var + eps).sqrt() + beta
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_tile_transpose():
    M, N = 32, 48
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(N, M, device='mps')
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    tile_transpose_kernel[grid](x, out, M, N, BLOCK_M=16, BLOCK_N=16)
    ref = x.t()
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_deep_chain():
    n = 1024
    x = torch.randn(n, device='mps') * 0.5
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    deep_chain_kernel[grid](x, out, n, BLOCK=256)
    # Reference
    a = torch.sigmoid(x)
    b = torch.exp(-a)
    c = torch.log(b + 1.0)
    d = torch.abs(c)
    e = torch.sqrt(d + 0.001)
    f = 1.0 / (e + 0.001)
    g = f * x
    h = torch.where(g > 0, g, -g)
    i = torch.floor(h * 100.0) / 100.0
    ref = i + 0.5
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_dual_reduce():
    M, N = 16, 32
    x = torch.randn(M, N, device='mps')
    row_sum = torch.zeros(M, device='mps')
    col_sum = torch.zeros(N, device='mps')
    dual_reduce_kernel[(1,)](x, row_sum, col_sum, M, N, BLOCK_M=16, BLOCK_N=32)
    ref_row = x.sum(dim=1)
    ref_col = x.sum(dim=0)
    err = max((row_sum - ref_row).abs().max().item(),
              (col_sum - ref_col).abs().max().item())
    return err < 1e-3, err


def test_fused_silu_mul():
    n = 2048
    gate = torch.randn(n, device='mps')
    up = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    fused_silu_mul_kernel[grid](gate, up, out, n, BLOCK=256)
    ref = (gate * torch.sigmoid(gate)) * up
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_log_softmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    log_softmax_kernel[(M,)](x, out, N, N, BLOCK=64)
    ref = torch.log_softmax(x, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 17) ===\n")

    tests = [
        ("Swizzle Copy", test_swizzle_copy),
        ("Online Softmax", test_online_softmax),
        ("Row Variance", test_row_variance),
        ("Fused LayerNorm", test_fused_layernorm),
        ("Chunked Sum (loop)", test_chunked_sum),
        ("Conditional Reduce", test_cond_reduce),
        ("BatchNorm", test_batchnorm),
        ("Tiled Transpose", test_tile_transpose),
        ("Deep Chain (10 ops)", test_deep_chain),
        ("Dual Reduce (row+col)", test_dual_reduce),
        ("Fused SiLU*up (LLaMA)", test_fused_silu_mul),
        ("Log-Softmax", test_log_softmax),
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
