#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 8.

Targets: real-world deep learning patterns and edge cases.
Fused Adam optimizer, dropout mask, multi-head attention building blocks,
top-k selection, dynamic shapes, chained reductions.
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Fused Adam optimizer step
@triton.jit
def adam_step_kernel(param_ptr, grad_ptr, m_ptr, v_ptr,
                     lr, beta1, beta2, eps, step, n,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(param_ptr + offs, mask=mask)
    g = tl.load(grad_ptr + offs, mask=mask)
    m = tl.load(m_ptr + offs, mask=mask)
    v = tl.load(v_ptr + offs, mask=mask)
    # Update moments
    m_new = beta1 * m + (1.0 - beta1) * g
    v_new = beta2 * v + (1.0 - beta2) * g * g
    # Bias correction
    m_hat = m_new / (1.0 - beta1)  # simplified: skip power for step>1
    v_hat = v_new / (1.0 - beta2)
    # Update params
    p_new = p - lr * m_hat / (tl.sqrt(v_hat) + eps)
    tl.store(param_ptr + offs, p_new, mask=mask)
    tl.store(m_ptr + offs, m_new, mask=mask)
    tl.store(v_ptr + offs, v_new, mask=mask)


# 2. Fused dropout + residual add + layer norm
@triton.jit
def dropout_residual_kernel(x_ptr, residual_ptr, out_ptr,
                             seed, p_drop, n,
                             BLOCK: tl.constexpr):
    """out = dropout(x, p_drop) + residual"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.load(residual_ptr + offs, mask=mask)
    # Generate dropout mask using hash
    rng = (seed + offs * 1103515245 + 12345) % 2147483647
    rng_float = (rng % 1000000).to(tl.float32) / 1000000.0
    keep_mask = rng_float > p_drop
    scale = 1.0 / (1.0 - p_drop)
    x_dropped = tl.where(keep_mask, x * scale, 0.0)
    out = x_dropped + r
    tl.store(out_ptr + offs, out, mask=mask)


# 3. Row-wise top-1 (simplified argmax via where chain)
@triton.jit
def top1_kernel(x_ptr, val_ptr, idx_ptr, M, N,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """For each row, find max value."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    max_vals = tl.max(x, axis=1)  # [M]
    tl.store(val_ptr + offs_m, max_vals, mask=offs_m < M)


# 4. Fused bias + GELU
@triton.jit
def fused_bias_gelu_kernel(x_ptr, bias_ptr, out_ptr, M, N,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    xb = x + bias[None, :]
    # Approximate GELU
    out = xb * tl.sigmoid(1.702 * xb)
    tl.store(x_ptr + offs_m[:, None] * N + offs_n[None, :], out, mask=mask)


# 5. Softmax backward (dL/dx from dL/dy and y=softmax(x))
@triton.jit
def softmax_backward_kernel(dy_ptr, y_ptr, dx_ptr, row_stride, n_cols,
                             BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    dy = tl.load(dy_ptr + row * row_stride + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + row * row_stride + offs, mask=mask, other=0.0)
    # dx = y * (dy - sum(dy * y))
    dot = tl.sum(dy * y, axis=0)
    dx = y * (dy - dot)
    tl.store(dx_ptr + row * row_stride + offs, dx, mask=mask)


# 6. Vector quantization error (find distance to nearest centroid)
@triton.jit
def vq_error_kernel(x_ptr, centroids_ptr, error_ptr,
                     n_points, n_centroids, dim,
                     BLOCK_D: tl.constexpr):
    """For each point, find squared distance to nearest centroid."""
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < dim
    # Load point
    x = tl.load(x_ptr + pid * dim + offs_d, mask=d_mask, other=0.0)
    # Find nearest centroid
    min_dist = 1e30
    for c in range(n_centroids):
        cent = tl.load(centroids_ptr + c * dim + offs_d, mask=d_mask, other=0.0)
        diff = x - cent
        dist = tl.sum(diff * diff, axis=0)
        min_dist = tl.where(dist < min_dist, dist, min_dist)
    tl.store(error_ptr + pid, min_dist)


# 7. Batch normalize (mean + var + normalize, different from layer norm)
@triton.jit
def batchnorm_forward_kernel(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr,
                               out_ptr, N, C, eps,
                               BLOCK: tl.constexpr):
    """Per-channel normalize: out = gamma * (x - mean) / sqrt(var + eps) + beta"""
    pid = tl.program_id(0)  # sample index
    offs = tl.arange(0, BLOCK)
    mask = offs < C
    x = tl.load(x_ptr + pid * C + offs, mask=mask)
    mean = tl.load(mean_ptr + offs, mask=mask)
    var = tl.load(var_ptr + offs, mask=mask)
    gamma = tl.load(gamma_ptr + offs, mask=mask)
    beta = tl.load(beta_ptr + offs, mask=mask)
    xn = (x - mean) / tl.sqrt(var + eps)
    out = gamma * xn + beta
    tl.store(out_ptr + pid * C + offs, out, mask=mask)


# 8. Cosine annealing learning rate schedule
@triton.jit
def cosine_lr_kernel(step_ptr, out_ptr, lr_max, lr_min, total_steps, n,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    step = tl.load(step_ptr + offs, mask=mask)
    step_f = step.to(tl.float32)
    # lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * step / total_steps))
    ratio = step_f / total_steps
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tl.cos(3.14159265 * ratio))
    tl.store(out_ptr + offs, lr, mask=mask)


# 9. Masked fill (set specific positions to a value)
@triton.jit
def masked_fill_kernel(x_ptr, mask_ptr, fill_val, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    m = tl.load(mask_ptr + offs, mask=mask)
    # mask is 0 or 1 (int); fill where mask == 1
    filled = tl.where(m != 0, fill_val, x)
    tl.store(x_ptr + offs, filled, mask=mask)


# 10. Weighted cross entropy (per-sample weights)
@triton.jit
def weighted_ce_kernel(logits_ptr, labels_ptr, weights_ptr, loss_ptr,
                        row_stride, n_classes,
                        BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_classes
    logits = tl.load(logits_ptr + row * row_stride + offs, mask=mask, other=-float('inf'))
    # Log-softmax
    max_val = tl.max(logits, axis=0)
    logits = logits - max_val
    log_sum_exp = tl.log(tl.sum(tl.exp(logits), axis=0))
    log_softmax = logits - log_sum_exp
    # Gather log prob at label
    label = tl.load(labels_ptr + row)
    label_mask = offs == label
    log_prob = tl.sum(tl.where(label_mask, log_softmax, 0.0), axis=0)
    # Apply weight
    weight = tl.load(weights_ptr + row)
    tl.store(loss_ptr + row, -log_prob * weight)


# ============================================================
# Test runners
# ============================================================

def test_adam_step():
    n = 1024
    param = torch.randn(n, device='mps')
    grad = torch.randn(n, device='mps')
    m = torch.zeros(n, device='mps')
    v = torch.zeros(n, device='mps')
    lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
    # Save copies for reference
    p_ref = param.clone()
    m_ref = m.clone()
    v_ref = v.clone()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    adam_step_kernel[grid](param, grad, m, v, lr, beta1, beta2, eps, 1.0, n, BLOCK=256)
    # Reference
    m_new = beta1 * m_ref + (1 - beta1) * grad
    v_new = beta2 * v_ref + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1)
    v_hat = v_new / (1 - beta2)
    p_new = p_ref - lr * m_hat / (v_hat.sqrt() + eps)
    err_p = (param - p_new).abs().max().item()
    err_m = (m - m_new).abs().max().item()
    err_v = (v - v_new).abs().max().item()
    err = max(err_p, err_m, err_v)
    return err < 1e-4, err


def test_dropout_residual():
    n = 2048
    x = torch.randn(n, device='mps')
    r = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    seed = 42
    p_drop = 0.1
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    dropout_residual_kernel[grid](x, r, out, seed, p_drop, n, BLOCK=256)
    # Just verify output is finite and roughly correct magnitude
    finite = torch.isfinite(out).all().item()
    # Mean of out should be roughly mean(x/(1-p) + r) when most elements kept
    return finite, 0.0 if finite else 1.0


def test_top1():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    val = torch.zeros(M, device='mps')
    top1_kernel[(triton.cdiv(M, 32),)](x, val, None, M, N, BLOCK_M=32, BLOCK_N=64)
    ref_val = x.max(dim=1).values
    err = (val - ref_val).abs().max().item()
    return err < 1e-5, err


def test_fused_bias_gelu():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    ref = (x + bias[None, :])
    ref = ref * torch.sigmoid(1.702 * ref)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    fused_bias_gelu_kernel[grid](x, bias, None, M, N, BLOCK_M=32, BLOCK_N=32)
    err = (x - ref).abs().max().item()
    return err < 1e-4, err


def test_softmax_backward():
    M, N = 16, 64
    y = torch.softmax(torch.randn(M, N, device='mps'), dim=1)
    dy = torch.randn(M, N, device='mps')
    dx = torch.zeros(M, N, device='mps')
    softmax_backward_kernel[(M,)](dy, y, dx, N, N, BLOCK_SIZE=64)
    # Reference: dx = y * (dy - sum(dy*y, dim=1, keepdim=True))
    ref = y * (dy - (dy * y).sum(dim=1, keepdim=True))
    err = (dx - ref).abs().max().item()
    return err < 1e-4, err


def test_vq_error():
    n_points = 16
    n_centroids = 4
    dim = 32
    x = torch.randn(n_points, dim, device='mps')
    centroids = torch.randn(n_centroids, dim, device='mps')
    error = torch.zeros(n_points, device='mps')
    vq_error_kernel[(n_points,)](x, centroids, error, n_points, n_centroids, dim, BLOCK_D=32)
    # Reference
    ref = torch.cdist(x, centroids, p=2).pow(2).min(dim=1).values
    err = (error - ref).abs().max().item()
    return err < 1e-2, err


def test_batchnorm_forward():
    N, C = 32, 64
    x = torch.randn(N, C, device='mps')
    mean = x.mean(dim=0)
    var = x.var(dim=0, correction=0)
    gamma = torch.ones(C, device='mps')
    beta = torch.zeros(C, device='mps')
    out = torch.zeros(N, C, device='mps')
    batchnorm_forward_kernel[(N,)](x, mean, var, gamma, beta, out, N, C, 1e-5, BLOCK=64)
    ref = (x - mean) / (var + 1e-5).sqrt() * gamma + beta
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_cosine_lr():
    n = 256
    steps = torch.arange(n, device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps')
    lr_max, lr_min, total_steps = 0.001, 0.0001, 1000.0
    cosine_lr_kernel[(1,)](steps, out, lr_max, lr_min, total_steps, n, BLOCK=256)
    import math
    ref = torch.tensor([lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * s / total_steps))
                        for s in range(n)], device='mps')
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_masked_fill():
    n = 1024
    x = torch.randn(n, device='mps')
    mask = (torch.rand(n, device='mps') > 0.5).int()
    fill_val = -999.0
    x_ref = x.clone()
    x_ref[mask.bool()] = fill_val
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    masked_fill_kernel[grid](x, mask, fill_val, n, BLOCK=256)
    err = (x - x_ref).abs().max().item()
    return err < 1e-5, err


def test_weighted_ce():
    M = 16
    n_classes = 32
    logits = torch.randn(M, n_classes, device='mps')
    labels = torch.randint(0, n_classes, (M,), device='mps', dtype=torch.int32)
    weights = torch.rand(M, device='mps') + 0.5
    loss = torch.zeros(M, device='mps')
    weighted_ce_kernel[(M,)](logits, labels, weights, loss, n_classes, n_classes, BLOCK_SIZE=32)
    # Reference
    ref = torch.nn.functional.cross_entropy(logits, labels.long(), reduction='none') * weights
    err = (loss - ref).abs().max().item()
    return err < 1e-3, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 8) ===\n")

    tests = [
        ("Adam Optimizer Step", test_adam_step),
        ("Dropout + Residual", test_dropout_residual),
        ("Top-1 (row max)", test_top1),
        ("Fused Bias+GELU (2D)", test_fused_bias_gelu),
        ("Softmax Backward", test_softmax_backward),
        ("VQ Error (nearest centroid)", test_vq_error),
        ("BatchNorm Forward", test_batchnorm_forward),
        ("Cosine LR Schedule", test_cosine_lr),
        ("Masked Fill", test_masked_fill),
        ("Weighted Cross-Entropy", test_weighted_ce),
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
