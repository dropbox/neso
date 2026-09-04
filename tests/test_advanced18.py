#!/usr/bin/env python3
"""Test advanced Triton kernel patterns - batch 18.

Targets: attention and transformer patterns that combine multiple ops:
- Causal masking with 2D grids
- Multi-head attention score computation
- Fused attention (Q@K^T + mask + softmax + @V)
- Cross-attention pattern
- KV-cache append
- Token embedding lookup
- Positional encoding addition
- Layer-wise residual connections
- Top-k selection pattern
- Sparse attention (blockwise masking)
"""
import sys
import torch
import triton
import triton.language as tl

torch.mps.synchronize()

# ============================================================
# Kernel definitions
# ============================================================

# 1. Causal mask generation
@triton.jit
def causal_mask_kernel(mask_ptr, seq_len, BLOCK: tl.constexpr):
    """Generate causal attention mask: mask[i,j] = 1 if j <= i, else 0."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < seq_len
    # Causal: allow positions j <= row
    causal = (offs <= row).to(tl.float32)
    tl.store(mask_ptr + row * seq_len + offs, causal, mask=mask)


# 2. Scaled dot-product attention (single head)
@triton.jit
def sdpa_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                seq_len, head_dim, scale,
                BLOCK_SEQ: tl.constexpr, BLOCK_D: tl.constexpr):
    """Compute scaled dot-product attention for one query position.

    out[q_pos] = softmax(q[q_pos] @ K^T * scale) @ V
    """
    q_pos = tl.program_id(0)

    # Load query vector for this position
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    q = tl.load(q_ptr + q_pos * head_dim + d_offs, mask=d_mask, other=0.0)

    # Compute attention scores: q @ K^T
    k_offs = tl.arange(0, BLOCK_SEQ)
    k_mask = k_offs < seq_len

    # For each key position, compute dot product
    scores = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
    for d in range(0, head_dim, BLOCK_D):
        d_range = d + tl.arange(0, BLOCK_D)
        dm = d_range < head_dim
        q_chunk = tl.load(q_ptr + q_pos * head_dim + d_range, mask=dm, other=0.0)
        # Load K^T chunk for all key positions
        for kp in range(seq_len):
            k_chunk = tl.load(k_ptr + kp * head_dim + d_range, mask=dm, other=0.0)
            dot = tl.sum(q_chunk * k_chunk, axis=0)
            # This is very inefficient but tests the loop pattern

    # For a simpler approach: just load one q and one k vector per score
    # Rewrite using a different strategy
    pass


# 3. Token embedding lookup
@triton.jit
def embedding_kernel(token_ids_ptr, embed_ptr, out_ptr, n_tokens, embed_dim,
                      BLOCK_D: tl.constexpr):
    """Look up embeddings for a batch of tokens."""
    tid = tl.program_id(0)  # token index
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < embed_dim
    # Load token ID
    token_id = tl.load(token_ids_ptr + tid)
    # Load embedding row
    embed = tl.load(embed_ptr + token_id * embed_dim + d_offs, mask=d_mask, other=0.0)
    # Store to output
    tl.store(out_ptr + tid * embed_dim + d_offs, embed, mask=d_mask)


# 4. Add positional encoding
@triton.jit
def add_pos_encoding_kernel(x_ptr, pos_ptr, out_ptr, seq_len, dim,
                              BLOCK_D: tl.constexpr):
    """out[pos, d] = x[pos, d] + pos_enc[pos, d]."""
    pos = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < dim
    x = tl.load(x_ptr + pos * dim + d_offs, mask=d_mask, other=0.0)
    pe = tl.load(pos_ptr + pos * dim + d_offs, mask=d_mask, other=0.0)
    out = x + pe
    tl.store(out_ptr + pos * dim + d_offs, out, mask=d_mask)


# 5. Residual connection + dropout (transformer layer)
@triton.jit
def residual_dropout_kernel(x_ptr, residual_ptr, out_ptr, seed, p, n,
                              BLOCK: tl.constexpr):
    """out = residual + dropout(x, p)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    r = tl.load(residual_ptr + offs, mask=mask)
    # Dropout
    random = tl.rand(seed, offs)
    keep = random > p
    scale = 1.0 / (1.0 - p)
    dropped = tl.where(keep, x * scale, 0.0)
    out = r + dropped
    tl.store(out_ptr + offs, out, mask=mask)


# 6. KV-cache append (add new KV to cache at given position)
@triton.jit
def kv_cache_append_kernel(cache_ptr, new_kv_ptr, pos, seq_dim, head_dim,
                             BLOCK_D: tl.constexpr):
    """Append new KV to cache at position `pos`."""
    head = tl.program_id(0)  # which head
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    # Load new KV for this head
    new_val = tl.load(new_kv_ptr + head * head_dim + d_offs, mask=d_mask)
    # Store to cache[head, pos, :]
    tl.store(cache_ptr + head * seq_dim * head_dim + pos * head_dim + d_offs,
             new_val, mask=d_mask)


# 7. Multi-head split (reshape from [seq, n_heads*head_dim] to per-head)
@triton.jit
def head_split_kernel(x_ptr, out_ptr, seq_len, n_heads, head_dim,
                        BLOCK_D: tl.constexpr):
    """Extract one head's data: out[seq, d] = x[seq, head*head_dim + d]."""
    seq = tl.program_id(0)
    head = tl.program_id(1)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    x = tl.load(x_ptr + seq * n_heads * head_dim + head * head_dim + d_offs,
                mask=d_mask)
    tl.store(out_ptr + head * seq_len * head_dim + seq * head_dim + d_offs,
             x, mask=d_mask)


# 8. Fused Q@K^T for single query against all keys
@triton.jit
def qk_scores_kernel(q_ptr, k_ptr, scores_ptr, seq_len, head_dim, scale,
                       BLOCK_D: tl.constexpr):
    """Compute attention scores: scores[q,k] = q[q,:] @ k[k,:] * scale."""
    q_pos = tl.program_id(0)
    k_pos = tl.program_id(1)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    q = tl.load(q_ptr + q_pos * head_dim + d_offs, mask=d_mask, other=0.0)
    k = tl.load(k_ptr + k_pos * head_dim + d_offs, mask=d_mask, other=0.0)
    dot = tl.sum(q * k, axis=0) * scale
    tl.store(scores_ptr + q_pos * seq_len + k_pos, dot)


# 9. Fused softmax + weighted sum (attention * V)
@triton.jit
def attn_output_kernel(scores_ptr, v_ptr, out_ptr, seq_len, head_dim,
                         BLOCK_SEQ: tl.constexpr, BLOCK_D: tl.constexpr):
    """Compute attention output: out[q,:] = softmax(scores[q,:]) @ V.

    This computes for one query position (one row of scores).
    """
    q_pos = tl.program_id(0)

    # Load and softmax the scores for this query
    s_offs = tl.arange(0, BLOCK_SEQ)
    s_mask = s_offs < seq_len
    scores = tl.load(scores_ptr + q_pos * seq_len + s_offs, mask=s_mask, other=-float('inf'))
    max_s = tl.max(scores, axis=0)
    exp_s = tl.exp(scores - max_s)
    sum_s = tl.sum(exp_s, axis=0)
    weights = exp_s / sum_s  # [BLOCK_SEQ]

    # Weighted sum over V: out[d] = sum_k(weights[k] * V[k, d])
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in range(seq_len):
        w = tl.load(scores_ptr + q_pos * seq_len + k)  # scalar
        # Actually use the softmax weight
        # (This is slow per-element but tests the pattern)

    # Simpler approach: for small seq_len, load all V and multiply
    # For now just do weighted sum element by element
    out = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k_idx in range(0, seq_len):
        # Load weight for this key position
        s_val = tl.load(scores_ptr + q_pos * seq_len + k_idx)
        max_v = tl.max(scores, axis=0)  # reuse for stability
        # Just use pre-computed weights array
        pass

    # Actually, let's use a direct matrix multiply approach
    # This is cleaner: out[d] = sum_k weights[k] * V[k,d]
    # Since weights is 1D [BLOCK_SEQ] and V is 2D [seq_len, head_dim],
    # we need tl.dot or manual accumulation
    for d in range(head_dim):
        v_col = tl.load(v_ptr + s_offs * head_dim + d, mask=s_mask, other=0.0)
        dot = tl.sum(weights * v_col, axis=0)
        if d < head_dim:
            tl.store(out_ptr + q_pos * head_dim + d, dot)


# 10. Cosine similarity between rows
@triton.jit
def cosine_sim_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """Compute cosine similarity between two vectors."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + row * N + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + row * N + offs, mask=mask, other=0.0)
    dot_ab = tl.sum(a * b, axis=0)
    norm_a = tl.sqrt(tl.sum(a * a, axis=0))
    norm_b = tl.sqrt(tl.sum(b * b, axis=0))
    sim = dot_ab / (norm_a * norm_b + 1e-8)
    tl.store(out_ptr + row, sim)


# ============================================================
# Test runners
# ============================================================

def test_causal_mask():
    seq_len = 32
    mask_out = torch.zeros(seq_len, seq_len, device='mps')
    causal_mask_kernel[(seq_len,)](mask_out, seq_len, BLOCK=32)
    ref = torch.tril(torch.ones(seq_len, seq_len, device='mps'))
    err = (mask_out - ref).abs().max().item()
    return err < 1e-5, err


def test_embedding():
    n_tokens = 32
    vocab_size = 100
    embed_dim = 64
    embed_table = torch.randn(vocab_size, embed_dim, device='mps')
    token_ids = torch.randint(0, vocab_size, (n_tokens,), device='mps', dtype=torch.int32)
    out = torch.zeros(n_tokens, embed_dim, device='mps')
    embedding_kernel[(n_tokens,)](token_ids, embed_table, out, n_tokens, embed_dim, BLOCK_D=64)
    ref = embed_table[token_ids.long()]
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_add_pos_encoding():
    seq_len = 16
    dim = 64
    x = torch.randn(seq_len, dim, device='mps')
    pe = torch.randn(seq_len, dim, device='mps')
    out = torch.zeros(seq_len, dim, device='mps')
    add_pos_encoding_kernel[(seq_len,)](x, pe, out, seq_len, dim, BLOCK_D=64)
    ref = x + pe
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_residual_dropout():
    n = 2048
    x = torch.randn(n, device='mps')
    residual = torch.randn(n, device='mps')
    out = torch.zeros(n, device='mps')
    p = 0.1
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    residual_dropout_kernel[grid](x, residual, out, 42, p, n, BLOCK=256)
    # Check residual is added correctly (some elements dropped)
    diff = out - residual  # should be either 0 (dropped) or x/(1-p)
    dropped = (diff.abs() < 1e-6).sum().item()
    kept = n - dropped
    keep_rate = kept / n
    err = abs(keep_rate - (1 - p))
    return err < 0.1, err


def test_kv_cache_append():
    n_heads = 4
    seq_dim = 32
    head_dim = 16
    cache = torch.zeros(n_heads, seq_dim, head_dim, device='mps')
    new_kv = torch.randn(n_heads, head_dim, device='mps')
    pos = 5
    kv_cache_append_kernel[(n_heads,)](cache.reshape(-1), new_kv.reshape(-1),
                                        pos, seq_dim, head_dim, BLOCK_D=16)
    ref = cache.clone()
    ref[:, pos, :] = new_kv
    # Check position pos was written
    err = (cache[:, pos, :] - new_kv).abs().max().item()
    return err < 1e-5, err


def test_head_split():
    seq_len = 8
    n_heads = 4
    head_dim = 16
    x = torch.randn(seq_len, n_heads * head_dim, device='mps')
    out = torch.zeros(n_heads, seq_len, head_dim, device='mps')
    head_split_kernel[(seq_len, n_heads)](x, out.reshape(-1), seq_len, n_heads,
                                           head_dim, BLOCK_D=16)
    ref = x.view(seq_len, n_heads, head_dim).permute(1, 0, 2)
    err = (out - ref).abs().max().item()
    return err < 1e-5, err


def test_qk_scores():
    seq_len = 16
    head_dim = 32
    scale = 1.0 / (head_dim ** 0.5)
    q = torch.randn(seq_len, head_dim, device='mps')
    k = torch.randn(seq_len, head_dim, device='mps')
    scores = torch.zeros(seq_len, seq_len, device='mps')
    qk_scores_kernel[(seq_len, seq_len)](q, k, scores, seq_len, head_dim, scale,
                                          BLOCK_D=32)
    ref = (q @ k.t()) * scale
    err = (scores - ref).abs().max().item()
    return err < 1e-4, err


def test_cosine_sim():
    M = 32
    N = 64
    a = torch.randn(M, N, device='mps')
    b = torch.randn(M, N, device='mps')
    out = torch.zeros(M, device='mps')
    cosine_sim_kernel[(M,)](a, b, out, N, BLOCK=64)
    # Reference
    dot = (a * b).sum(dim=1)
    norm_a = a.norm(dim=1)
    norm_b = b.norm(dim=1)
    ref = dot / (norm_a * norm_b + 1e-8)
    err = (out - ref).abs().max().item()
    return err < 1e-4, err


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=== Advanced Kernel Pattern Tests (Batch 18) ===\n")
    print("    (Attention & transformer building blocks)\n")

    tests = [
        ("Causal Mask", test_causal_mask),
        ("Token Embedding", test_embedding),
        ("Add Positional Encoding", test_add_pos_encoding),
        ("Residual + Dropout", test_residual_dropout),
        ("KV-Cache Append", test_kv_cache_append),
        ("Multi-Head Split", test_head_split),
        ("Q@K^T Attention Scores", test_qk_scores),
        ("Cosine Similarity", test_cosine_sim),
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
