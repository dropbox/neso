#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 26.

Targets: argmin, larger patterns, mixed types:
- tl.argmin (counterpart to argmax)
- Row-wise argmin
- tl.where with mixed int/float outputs
- Large 2D tile operations (64x64 element-wise)
- Fused multi-head attention score + causal mask
- Batch-wise softmax with temperature
- Double reduction (reduce 2D to scalar)
- Integer arithmetic chain
- Fused swish (x * sigmoid(beta * x))
- Row-wise log-sum-exp (numerically stable)
- Fused dropout + residual + layer norm (inference, p=0)
- Multi-output with different dtypes
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Row-wise argmin
@triton.jit
def argmin_kernel(x_ptr, idx_ptr, N, BLOCK_N: tl.constexpr):
    """idx[row] = argmin(x[row, :])."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=float('inf'))
    best_idx = tl.argmin(x, axis=0)
    tl.store(idx_ptr + row, best_idx)


# 2. Fused swish activation (x * sigmoid(beta * x))
@triton.jit
def swish_kernel(x_ptr, out_ptr, beta, n, BLOCK: tl.constexpr):
    """out = x * sigmoid(beta * x)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.sigmoid(beta * x)
    tl.store(out_ptr + offs, out, mask=mask)


# 3. Row-wise log-sum-exp
@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, N, BLOCK_N: tl.constexpr):
    """out[row] = log(sum(exp(x[row, :])))."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float('inf'))
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    lse = mx + tl.log(tl.sum(ex, axis=0))
    tl.store(out_ptr + row, lse)


# 4. Softmax with temperature
@triton.jit
def temp_softmax_kernel(x_ptr, out_ptr, temp, n_cols, BLOCK: tl.constexpr):
    """out = softmax(x / temp) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float('inf'))
    x = x / temp
    mx = tl.max(x, axis=0)
    ex = tl.exp(x - mx)
    sm = tl.sum(ex, axis=0)
    out = ex / sm
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 5. Integer arithmetic chain
@triton.jit
def int_arith_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = (a * b + a) % 256 + (b - a)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out = (a * b + a) % 256 + (b - a)
    tl.store(out_ptr + offs, out, mask=mask)


# 6. Batch-normalized inference (mean/var pre-computed)
@triton.jit
def batchnorm_infer_kernel(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr,
                             out_ptr, n_features, eps,
                             BLOCK: tl.constexpr):
    """out = gamma * (x - mean) / sqrt(var + eps) + beta."""
    sample = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_features
    x = tl.load(x_ptr + sample * n_features + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    var = tl.load(var_ptr + offs, mask=mask, other=1.0)
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    xn = (x - mean) / tl.sqrt(var + eps)
    out = gamma * xn + beta
    tl.store(out_ptr + sample * n_features + offs, out, mask=mask)


# 7. Squared differences (for MSE loss)
@triton.jit
def mse_kernel(pred_ptr, target_ptr, loss_ptr, n, BLOCK: tl.constexpr):
    """loss[i] = (pred[i] - target[i])^2."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(pred_ptr + offs, mask=mask)
    t = tl.load(target_ptr + offs, mask=mask)
    diff = p - t
    loss = diff * diff
    tl.store(loss_ptr + offs, loss, mask=mask)


# 8. Row-wise min and max (two reductions)
@triton.jit
def row_minmax_kernel(x_ptr, min_ptr, max_ptr, N, BLOCK_N: tl.constexpr):
    """min[row], max[row] = min/max over x[row, :]."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    x_for_min = tl.load(x_ptr + row * N + offs, mask=mask, other=float('inf'))
    x_for_max = tl.load(x_ptr + row * N + offs, mask=mask, other=-float('inf'))
    mn = tl.min(x_for_min, axis=0)
    mx = tl.max(x_for_max, axis=0)
    tl.store(min_ptr + row, mn)
    tl.store(max_ptr + row, mx)


# 9. Exponential linear unit (ELU)
@triton.jit
def elu_kernel(x_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):
    """ELU: out = x if x > 0 else alpha * (exp(x) - 1)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    pos = x
    neg = alpha * (tl.exp(x) - 1.0)
    out = tl.where(x > 0.0, pos, neg)
    tl.store(out_ptr + offs, out, mask=mask)


# 10. Fused bias + residual + ReLU
@triton.jit
def bias_residual_relu_kernel(x_ptr, bias_ptr, residual_ptr, out_ptr,
                                n_cols, BLOCK: tl.constexpr):
    """out = ReLU(x + bias + residual) per row."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    r = tl.load(residual_ptr + row * n_cols + offs, mask=mask, other=0.0)
    out = tl.maximum(x + b + r, 0.0)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)


# 11. Hardswish activation
@triton.jit
def hardswish_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """HardSwish: out = x * min(max(x+3, 0), 6) / 6."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    tl.store(out_ptr + offs, out, mask=mask)


# 12. Smooth L1 loss (Huber loss)
@triton.jit
def smooth_l1_kernel(pred_ptr, target_ptr, loss_ptr, beta, n,
                       BLOCK: tl.constexpr):
    """Smooth L1: loss = 0.5*d^2/beta if |d|<beta else |d|-0.5*beta."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    p = tl.load(pred_ptr + offs, mask=mask)
    t = tl.load(target_ptr + offs, mask=mask)
    d = p - t
    abs_d = tl.abs(d)
    loss = tl.where(abs_d < beta, 0.5 * d * d / beta, abs_d - 0.5 * beta)
    tl.store(loss_ptr + offs, loss, mask=mask)


# ============================================================
# Test runners
# ============================================================

def test_argmin():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    idx = torch.zeros(M, device='mps', dtype=torch.int32)
    argmin_kernel[(M,)](x, idx, N, BLOCK_N=128)
    ref = x.argmin(dim=1).int()
    err = (idx - ref).abs().max().item()
    return err == 0, float(err)


def test_swish():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    swish_kernel[grid](x, out, 1.5, n, BLOCK=256)
    ref = x * torch.sigmoid(1.5 * x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_logsumexp():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    logsumexp_kernel[(M,)](x, out, N, BLOCK_N=128)
    ref = torch.logsumexp(x, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_temp_softmax():
    M, N = 32, 64
    x = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    temp = 0.5
    temp_softmax_kernel[(M,)](x, out, temp, N, BLOCK=64)
    ref = torch.softmax(x / temp, dim=1)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_int_arith():
    n = 512
    a = torch.randint(1, 50, (n,), device='mps', dtype=torch.int32)
    b = torch.randint(1, 50, (n,), device='mps', dtype=torch.int32)
    out = torch.zeros(n, device='mps', dtype=torch.int32)
    int_arith_kernel[(2,)](a, b, out, n, BLOCK=256)
    ref = (a * b + a) % 256 + (b - a)
    err = (out - ref).abs().max().item()
    return err == 0, float(err)


def test_batchnorm_infer():
    batch = 64
    n_features = 128
    x = torch.randn(batch, n_features, device='mps')
    mean = torch.randn(n_features, device='mps')
    var = torch.rand(n_features, device='mps') + 0.01
    gamma = torch.randn(n_features, device='mps')
    beta = torch.randn(n_features, device='mps')
    out = torch.zeros(batch, n_features, device='mps')
    eps = 1e-5
    batchnorm_infer_kernel[(batch,)](x, mean, var, gamma, beta, out,
                                       n_features, eps, BLOCK=128)
    ref = gamma * (x - mean) / (var + eps).sqrt() + beta
    err = (out - ref).abs().max().item()
    return err < 1e-3, err


def test_mse():
    n = 2048
    pred = torch.randn(n, device='mps')
    target = torch.randn(n, device='mps')
    loss = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    mse_kernel[grid](pred, target, loss, n, BLOCK=256)
    ref = (pred - target) ** 2
    err = (loss - ref).abs().max().item()
    return err < 1e-5, err


def test_row_minmax():
    M, N = 64, 128
    x = torch.randn(M, N, device='mps')
    min_out = torch.zeros(M, device='mps')
    max_out = torch.zeros(M, device='mps')
    row_minmax_kernel[(M,)](x, min_out, max_out, N, BLOCK_N=128)
    ref_min = x.min(dim=1).values
    ref_max = x.max(dim=1).values
    err = max((min_out - ref_min).abs().max().item(),
              (max_out - ref_max).abs().max().item())
    return err < 1e-5, err


def test_elu():
    n = 2048
    x = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    elu_kernel[grid](x, out, 1.0, n, BLOCK=256)
    ref = torch.nn.functional.elu(x, alpha=1.0)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_bias_residual_relu():
    M, N = 32, 128
    x = torch.randn(M, N, device='mps')
    bias = torch.randn(N, device='mps')
    residual = torch.randn(M, N, device='mps')
    out = torch.zeros(M, N, device='mps')
    bias_residual_relu_kernel[(M,)](x, bias, residual, out, N, BLOCK=128)
    ref = torch.relu(x + bias + residual)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_hardswish():
    n = 2048
    x = torch.randn(n, device='mps') * 5
    out = torch.zeros(n, device='mps')
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    hardswish_kernel[grid](x, out, n, BLOCK=256)
    ref = torch.nn.functional.hardswish(x)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


def test_smooth_l1():
    n = 2048
    pred = torch.randn(n, device='mps')
    target = torch.randn(n, device='mps')
    loss = torch.zeros(n, device='mps')
    beta = 1.0
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    smooth_l1_kernel[grid](pred, target, loss, beta, n, BLOCK=256)
    ref = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none', beta=beta)
    err = (loss - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 26) ===\n")
    print("    (Argmin, activations, losses, batch norm)\n")

    tests = [
        ("Row Argmin", test_argmin),
        ("Swish (beta=1.5)", test_swish),
        ("LogSumExp (row)", test_logsumexp),
        ("Temp Softmax", test_temp_softmax),
        ("Int Arithmetic Chain", test_int_arith),
        ("BatchNorm Inference", test_batchnorm_infer),
        ("MSE Loss", test_mse),
        ("Row Min+Max", test_row_minmax),
        ("ELU Activation", test_elu),
        ("Bias+Residual+ReLU", test_bias_residual_relu),
        ("HardSwish", test_hardswish),
        ("Smooth L1 Loss", test_smooth_l1),
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
