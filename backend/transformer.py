"""
Pharmakon Transformer – Pure NumPy, Character-Level, RoPE-based Decoder
v4.0 with KV caching, FlashAttention, gradient checkpointing.
No absolute positional embeddings — position is encoded solely via Rotary Position Embeddings (RoPE).
"""

import numpy as np

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def softmax(x, axis=-1):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len):
    """Upper triangular mask: 0 for allowed (i>=j), -1e10 for forbidden (i<j)."""
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float64), k=1) * -1e10
    return mask


def precompute_freqs(head_dim, max_seq_len, base=10000.0):
    """Precompute cos and sin for RoPE."""
    freqs = 1.0 / (base ** (np.arange(0, head_dim, 2).astype(np.float64) / head_dim))
    t = np.arange(max_seq_len, dtype=np.float64)
    freqs = np.outer(t, freqs)               # (max_seq_len, head_dim//2)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input. x: shape (..., dim)"""
    dim = x.shape[-1]
    x1 = x[..., :dim // 2]
    x2 = x[..., dim // 2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin, offset=0):
    """Applies RoPE to query and key tensors.
    q, k: shape (batch, n_heads, seq_len, head_dim)
    cos, sin: shape (max_seq_len, head_dim//2)
    """
    seq_len = q.shape[2]
    # cos_part, sin_part: (1, 1, seq_len, head_dim//2)
    cos_part = cos[offset:offset + seq_len, :][None, None, :, :]
    sin_part = sin[offset:offset + seq_len, :][None, None, :, :]
    # Duplicate for real/imaginary interleaving [a,b,a,b] (NOT repeat!)
    cos_part = np.concatenate([cos_part, cos_part], axis=-1)
    sin_part = np.concatenate([sin_part, sin_part], axis=-1)

    q_rot = q * cos_part + rotate_half(q) * sin_part
    k_rot = k * cos_part + rotate_half(k) * sin_part
    return q_rot, k_rot


def apply_rope_backward(dq_rot, dk_rot, cos, sin, offset=0):
    """Reverses the RoPE rotation for backpropagation."""
    seq_len = dq_rot.shape[2]
    cos_part = cos[offset:offset + seq_len, :][None, None, :, :]
    sin_part = sin[offset:offset + seq_len, :][None, None, :, :]
    cos_part = np.concatenate([cos_part, cos_part], axis=-1)
    sin_part = np.concatenate([sin_part, sin_part], axis=-1)

    dq = dq_rot * cos_part - rotate_half(dq_rot) * sin_part
    dk = dk_rot * cos_part - rotate_half(dk_rot) * sin_part
    return dq, dk


# -----------------------------------------------------------------------------
# FlashAttention (block-level, online softmax, deterministic dropout)
# -----------------------------------------------------------------------------
def _dropout_hash(B, H, S, j_start, j_end, p):
    """Vectorized deterministic dropout mask based on indices (avoids storing full mask)."""
    b_idx = np.arange(B)[:, None, None, None]
    h_idx = np.arange(H)[None, :, None, None]
    i_idx = np.arange(S)[None, None, :, None]
    j_idx = np.arange(j_start, j_end)[None, None, None, :]
    
    z = ( (b_idx * 0x9E3779B1).astype(np.uint32) ^
          (h_idx * 0x85EBCA77).astype(np.uint32) ^
          (i_idx * 0xC2B2AE35).astype(np.uint32) ^
          (j_idx * 0x27D4EB2F).astype(np.uint32) )
    val = (z % 10007).astype(np.float64) / 10007.0
    return np.where(val > p, 1.0 / (1.0 - p), 0.0)


def flash_attention_forward(Q, K, V, mask=None, dropout=0.0, training=False):
    """
    Q, K, V: (B, H, S, D)   mask: None or bool (S, S) causal mask that can be applied blockwise.
    Returns: output O (B,H,S,D), flash_cache dict for backward.
    Flash cache contains: O, LSE, Q, K, V, dropout_p, training flag.
    """
    # NOTE: Not worth optimizing/toggling below S=256; profiled non-attention overhead as the bottleneck, see walkthrough.md
    B, H, S, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    # Block size for tiling over key/value dimension
    Bc = min(32, S)

    # Initialize output and statistics
    O = np.zeros((B, H, S, D), dtype=np.float64)
    L = np.zeros((B, H, S), dtype=np.float64)   # sum of exp (un-normalised)
    m = np.full((B, H, S), -np.inf, dtype=np.float64)  # max logit

    # Tile over key/value blocks
    for j_start in range(0, S, Bc):
        j_end = min(j_start + Bc, S)
        Kj = K[:, :, j_start:j_end, :]      # (B, H, Bc, D)
        Vj = V[:, :, j_start:j_end, :]      # (B, H, Bc, D)

        # Compute scores block: (B, H, S, Bc)
        scores = np.matmul(Q, Kj.transpose(0, 1, 3, 2)) * scale

        # Apply causal mask blockwise
        if mask is not None:
            # mask is (S, S) boolean with True for -inf. We extract sub-block.
            mask_block = mask[:, j_start:j_end]   # (S, Bc)
            scores = scores + np.where(mask_block[None, None, :, :], -1e10, 0.0)

        # Online softmax update
        m_new = np.maximum(m, np.max(scores, axis=-1))          # (B,H,S)
        # Correction factor for previous sum
        alpha = np.exp(m - m_new)                               # (B,H,S)
        # Exponentiated scores for this block
        P = np.exp(scores - m_new[..., None])                   # (B,H,S,Bc)

        # Dropout on attention probabilities (deterministic hash)
        if training and dropout > 0.0:
            mask_drop = _dropout_hash(B, H, S, j_start, j_end, dropout)
            P = P * mask_drop

        # Update running sum and output
        L_new = alpha * L + np.sum(P, axis=-1)                 # (B,H,S)
        O = O * alpha[..., None] + np.matmul(P, Vj)           # (B,H,S,D)

        m = m_new
        L = L_new

    # Final divide by L (Clamp to prevent NaNs if all scores masked)
    O = O / np.maximum(L[..., None], 1e-12)
    # LSE = m + log(L)
    LSE = m + np.log(np.maximum(L, 1e-20))
    flash_cache = {'O': O, 'LSE': LSE, 'Q': Q, 'K': K, 'V': V,
                   'dropout': dropout, 'training': training, 'mask': mask}
    return O, flash_cache


def flash_attention_backward(dO, flash_cache):
    """Backward pass for FlashAttention using recomputation."""
    O = flash_cache['O']
    LSE = flash_cache['LSE']
    Q = flash_cache['Q']
    K = flash_cache['K']
    V = flash_cache['V']
    dropout = flash_cache['dropout']
    training = flash_cache['training']
    mask = flash_cache['mask']

    B, H, S, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    Bc = min(32, S)

    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    # Recompute block by block, same tiling as forward
    for j_start in range(0, S, Bc):
        j_end = min(j_start + Bc, S)
        Kj = K[:, :, j_start:j_end, :]
        Vj = V[:, :, j_start:j_end, :]

        scores = np.matmul(Q, Kj.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            mask_block = mask[:, j_start:j_end]
            scores = scores + np.where(mask_block[None, None, :, :], -1e10, 0.0)

        # Recompute P (softmax) using stored LSE
        P = np.exp(scores - LSE[..., None])        # (B,H,S,Bc)

        # Re-apply dropout mask exactly as in forward
        if training and dropout > 0.0:
            mask_drop = _dropout_hash(B, H, S, j_start, j_end, dropout)
            P = P * mask_drop

        # Gradients from output
        dV_j = np.matmul(P.transpose(0, 1, 3, 2), dO)    # (B,H,Bc,D)
        dV[:, :, j_start:j_end, :] += dV_j

        dP = np.matmul(dO, Vj.transpose(0, 1, 3, 2))    # (B,H,S,Bc)
        # Backward through softmax (using the fact that dP = P * (dS - sum(P*dS)))
        dS = P * (dP - np.sum(P * dP, axis=-1, keepdims=True))
        dS *= scale

        dQ += np.matmul(dS, Kj)                         # (B,H,S,D)
        dK_j = np.matmul(dS.transpose(0, 1, 3, 2), Q)  # (B,H,Bc,D)
        dK[:, :, j_start:j_end, :] += dK_j

    return dQ, dK, dV


# -----------------------------------------------------------------------------
# Layer Normalization (unchanged)
# -----------------------------------------------------------------------------
class LayerNorm:
    def __init__(self, dim, eps=1e-8):
        self.eps = eps
        self.gamma = np.ones(dim, dtype=np.float64)
        self.beta = np.zeros(dim, dtype=np.float64)
        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        """x: shape (batch, seq_len, dim) or (seq_len, dim)"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        sigma = np.sqrt(var + self.eps)
        x_hat = (x - mean) / sigma
        out = self.gamma * x_hat + self.beta
        return out, (x_hat, sigma)

    def backward(self, d_out, cache):
        """d_out: same shape as x in forward"""
        x_hat, sigma = cache
        dx_hat = d_out * self.gamma

        dx = (dx_hat
              - np.mean(dx_hat, axis=-1, keepdims=True)
              - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
              ) / sigma

        sum_axes = tuple(range(d_out.ndim - 1))
        self.dgamma = np.sum(d_out * x_hat, axis=sum_axes)
        self.dbeta = np.sum(d_out, axis=sum_axes)
        return dx


# -----------------------------------------------------------------------------
# Transformer Block with KV Cache, FlashAttention, Checkpointing
# -----------------------------------------------------------------------------
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, max_seq_len=1024):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Precompute RoPE frequencies
        self.cos, self.sin = precompute_freqs(self.head_dim, max_seq_len)

        # Attention weights (Xavier uniform)
        limit = np.sqrt(6.0 / (embed_dim + embed_dim))
        self.Wq = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)
        self.Wk = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)
        self.Wv = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)
        self.Wo = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)

        # Pre-norm layers
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

        # Feed-forward weights
        limit_ff1 = np.sqrt(6.0 / (embed_dim + ff_dim))
        self.W1 = np.random.uniform(-limit_ff1, limit_ff1, (embed_dim, ff_dim)).astype(np.float64)
        self.b1 = np.zeros(ff_dim, dtype=np.float64)

        limit_ff2 = np.sqrt(6.0 / (ff_dim + embed_dim))
        self.W2 = np.random.uniform(-limit_ff2, limit_ff2, (ff_dim, embed_dim)).astype(np.float64)
        self.b2 = np.zeros(embed_dim, dtype=np.float64)

        # Gradient buffers
        self.dWq = self.dWk = self.dWv = self.dWo = None
        self.dW1 = self.db1 = self.dW2 = self.db2 = None

    def forward(self, x, training=False, use_cache=False, kv_cache=None, checkpoint=False):
        """
        If use_cache=True (inference): returns (out, new_kv_cache).
        If training=True and checkpoint=True: returns (out, x_input) minimal cache.
        If training=True and checkpoint=False: returns (out, full_cache_dict).
        Else: returns out only.
        """
        B, S, D = x.shape

        # ----------------- Pre-norm & QKV projections -----------------
        h_norm1, ln1_cache = self.ln1.forward(x)
        Q_proj = np.dot(h_norm1, self.Wq)
        K_proj = np.dot(h_norm1, self.Wk)
        V_proj = np.dot(h_norm1, self.Wv)

        # ----------------- Handle KV cache -----------------
        if use_cache:
            # x should be a single token (B,1,D)
            assert S == 1, "KV cache mode only supports sequence length 1"
            # RoPE offset = current cache length
            offset = 0 if kv_cache is None else kv_cache['offset']
            Q = np.ascontiguousarray(Q_proj.reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
            K = np.ascontiguousarray(K_proj.reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
            V = np.ascontiguousarray(V_proj.reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
            Q_rot, K_rot = apply_rope(Q, K, self.cos, self.sin, offset=offset)

            if kv_cache is not None:
                # Keep at most 63 past tokens to match context window size of 64
                K_past = kv_cache['K']
                V_past = kv_cache['V']
                if K_past.shape[2] > 63:
                    K_past = K_past[:, :, -63:, :]
                    V_past = V_past[:, :, -63:, :]
                K_full = np.concatenate([K_past, K_rot], axis=2)  # (B,H,total,D)
                V_full = np.concatenate([V_past, V], axis=2)
            else:
                K_full, V_full = K_rot, V

            # Attention using full sequence (single query)
            scores = np.matmul(Q_rot, K_full.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
            attn_weights = softmax(scores, axis=-1)
            head_out = np.matmul(attn_weights, V_full)   # (B,H,1,D)
            concat = np.ascontiguousarray(head_out.transpose(0, 2, 1, 3).reshape(B, 1, D))
            attn_out = np.dot(concat, self.Wo)

            # Residual
            x_resid1 = x + attn_out

            # No dropout during inference (use_cache implies eval)
            # Feed-forward
            h_norm2, _ = self.ln2.forward(x_resid1)
            ff1 = np.maximum(0, np.dot(h_norm2, self.W1) + self.b1)
            ff2 = np.dot(ff1, self.W2) + self.b2
            out = x_resid1 + ff2

            # Update cache
            new_kv_cache = {
                'K': K_full,
                'V': V_full,
                'offset': offset + 1
            }
            return out, new_kv_cache

        # ----------------- Training / non-cache path -----------------
        # Split into heads
        Q = np.ascontiguousarray(Q_proj.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
        K = np.ascontiguousarray(K_proj.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
        V = np.ascontiguousarray(V_proj.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
        Q_rot, K_rot = apply_rope(Q, K, self.cos, self.sin, offset=0)

        # FlashAttention forward
        mask = create_causal_mask(S).astype(np.bool_)   # bool for efficiency
        attn_out_concat, flash_cache = flash_attention_forward(
            Q_rot, K_rot, V, mask=mask, dropout=self.dropout, training=training
        )
        # attn_out_concat: (B,H,S,D) -> concat (B,S,D)
        concat = np.ascontiguousarray(attn_out_concat.transpose(0, 2, 1, 3).reshape(B, S, D))
        attn_out = np.dot(concat, self.Wo)

        # Output projection dropout (same as before)
        if training and self.dropout > 0:
            attn_proj_mask = (np.random.rand(*attn_out.shape) > self.dropout).astype(attn_out.dtype)
            attn_out_dropped = attn_out * attn_proj_mask / (1.0 - self.dropout)
        else:
            attn_proj_mask = None
            attn_out_dropped = attn_out


        x_resid1 = x + attn_out_dropped

        # Pre-norm & Feed-Forward
        h_norm2, ln2_cache = self.ln2.forward(x_resid1)


        ff1_pre = np.dot(h_norm2, self.W1) + self.b1
        ff1 = np.maximum(0, ff1_pre)
        ff2 = np.dot(ff1, self.W2) + self.b2


        if training and self.dropout > 0:
            ffn_drop_mask = (np.random.rand(*ff2.shape) > self.dropout).astype(ff2.dtype)
            ff2_dropped = ff2 * ffn_drop_mask / (1.0 - self.dropout)
        else:
            ffn_drop_mask = None
            ff2_dropped = ff2


        out = x_resid1 + ff2_dropped

        # Cache construction for backward
        if training:
            if checkpoint:
                # Store only input x for recomputation
                return out, x
            else:
                cache = {
                    'x': x,
                    'h_norm1': h_norm1, 'ln1_cache': ln1_cache,
                    'Q_proj': Q_proj, 'K_proj': K_proj, 'V_proj': V_proj,
                    'Q': Q, 'K': K, 'V': V,
                    'Q_rot': Q_rot, 'K_rot': K_rot,
                    'flash_cache': flash_cache,    # for attention backward
                    'attn_out': attn_out, 'attn_proj_mask': attn_proj_mask,
                    'x_resid1': x_resid1,
                    'h_norm2': h_norm2, 'ln2_cache': ln2_cache,
                    'ff1_pre': ff1_pre, 'ff1': ff1,
                    'ff2': ff2, 'ffn_drop_mask': ffn_drop_mask
                }
                return out, cache
        return out

    def backward(self, d_out, cache_or_x):
        """
        Accepts either full cache dict (standard) or x_input if checkpointing.
        For checkpointing, recomputes forward and then backward.
        """
        if isinstance(cache_or_x, np.ndarray):
            # Checkpoint mode: recompute forward to get full cache
            x_input = cache_or_x
            _, full_cache = self.forward(x_input, training=True, checkpoint=False)
            return self._backward_from_cache(d_out, full_cache)
        else:
            return self._backward_from_cache(d_out, cache_or_x)

    def _backward_from_cache(self, d_out, cache):
        """Core backward logic using full cache."""
        B, S, D = d_out.shape


        x = cache['x']
        h_norm1 = cache['h_norm1']; ln1_cache = cache['ln1_cache']
        Q_proj = cache['Q_proj']; K_proj = cache['K_proj']; V_proj = cache['V_proj']
        Q = cache['Q']; K = cache['K']; V = cache['V']
        Q_rot = cache['Q_rot']; K_rot = cache['K_rot']
        flash_cache = cache['flash_cache']
        attn_proj_mask = cache['attn_proj_mask']
        x_resid1 = cache['x_resid1']
        h_norm2 = cache['h_norm2']; ln2_cache = cache['ln2_cache']
        ff1_pre = cache['ff1_pre']; ff1 = cache['ff1']
        ffn_drop_mask = cache['ffn_drop_mask']

        # ---- 1. Feed-Forward backward ----
        d_ff2_dropped = d_out
        if ffn_drop_mask is not None:
            d_ff2 = d_ff2_dropped * ffn_drop_mask / (1.0 - self.dropout)
        else:
            d_ff2 = d_ff2_dropped

        d_ff2_flat = d_ff2.reshape(B * S, D)
        ff1_flat = ff1.reshape(B * S, self.ff_dim)

        self.dW2 = np.dot(ff1_flat.T, d_ff2_flat)
        self.db2 = np.sum(d_ff2_flat, axis=0)

        d_ff1 = np.dot(d_ff2_flat, self.W2.T).reshape(B, S, self.ff_dim)


        d_ff1[ff1_pre <= 0] = 0.0

        d_ff1_flat = d_ff1.reshape(B * S, self.ff_dim)
        h_norm2_flat = h_norm2.reshape(B * S, D)

        self.dW1 = np.dot(h_norm2_flat.T, d_ff1_flat)
        self.db1 = np.sum(d_ff1_flat, axis=0)

        d_h_norm2 = np.dot(d_ff1_flat, self.W1.T).reshape(B, S, D)

        d_x_ffn_input = self.ln2.backward(d_h_norm2, ln2_cache)
    
        d_x_resid1 = d_out + d_x_ffn_input

        # ---- 2. Attention backward using FlashAttention backward ----
        d_attn_out_dropped = d_x_resid1

        if attn_proj_mask is not None:
            d_attn_out = d_attn_out_dropped * attn_proj_mask / (1.0 - self.dropout)
        else:
            d_attn_out = d_attn_out_dropped

        d_attn_out_flat = d_attn_out.reshape(B * S, D)
        concat_flat = flash_cache['O'].transpose(0, 2, 1, 3).reshape(B * S, D)
        self.dWo = np.dot(concat_flat.T, d_attn_out_flat)
        d_concat = np.dot(d_attn_out_flat, self.Wo.T).reshape(B, S, D)

        d_head_out = np.ascontiguousarray(
            d_concat.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        )
        # Call FlashAttention backward
        dQ_rot, dK_rot, dV = flash_attention_backward(d_head_out, flash_cache)

        # Reverse RoPE
        dQ, dK = apply_rope_backward(dQ_rot, dK_rot, self.cos, self.sin)

        # Merge heads
        dQ = np.ascontiguousarray(dQ.transpose(0, 2, 1, 3).reshape(B * S, D))
        dK = np.ascontiguousarray(dK.transpose(0, 2, 1, 3).reshape(B * S, D))
        dV = np.ascontiguousarray(dV.transpose(0, 2, 1, 3).reshape(B * S, D))

        h_norm1_flat = h_norm1.reshape(B * S, D)

        self.dWq = np.dot(h_norm1_flat.T, dQ)
        self.dWk = np.dot(h_norm1_flat.T, dK)
        self.dWv = np.dot(h_norm1_flat.T, dV)

        d_h_norm1 = (np.dot(dQ, self.Wq.T) + np.dot(dK, self.Wk.T) + np.dot(dV, self.Wv.T))
        d_h_norm1 = d_h_norm1.reshape(B, S, D)

        d_x_att_input = self.ln1.backward(d_h_norm1, ln1_cache)

        dx = d_x_resid1 + d_x_att_input
        return dx


# -----------------------------------------------------------------------------
# Full Pharmakon Transformer
# -----------------------------------------------------------------------------
class PharmakonTransformer:
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, ff_dim=128,
                 num_layers=2, max_seq_len=1024, dropout=0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers


        limit = np.sqrt(6.0 / (vocab_size + embed_dim))
        self.token_embedding = np.random.uniform(-limit, limit, (vocab_size, embed_dim)).astype(np.float64)

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout=dropout, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ]
        self.ln_final = LayerNorm(embed_dim)


        self.W_out = np.random.uniform(-limit, limit, (embed_dim, vocab_size)).astype(np.float64)


        self.dtoken_embedding = None
        self.dW_out = None

    def forward(self, idx, training=False, use_cache=False, kv_caches=None, checkpoint=False):
        """
        idx: (batch, seq_len) integer tokens.
        If use_cache: kv_caches list of per-layer caches or None; returns logits, new_kv_caches.
        If training and checkpoint=True: caches stored minimally (inputs per block).
        Returns appropriate output.
        """
        is_2d = (idx.ndim == 1)
        if is_2d:
            idx = idx[np.newaxis, :]

        B, S = idx.shape
        x = self.token_embedding[idx]

        block_caches = []
        new_kv_caches = [] if use_cache else None
        ln_final_cache = None

        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            if use_cache and new_kv_caches is not None:
                x, new_kv = block.forward(x, training=False, use_cache=True, kv_cache=kv_cache)
                new_kv_caches.append(new_kv)
            else:
                out = block.forward(x, training=training, checkpoint=checkpoint)
                if training:
                    if checkpoint:
                        x_out, cache_or_x = out
                        x = x_out
                        block_caches.append(cache_or_x)   # this is x_input (ndarray)
                    else:
                        x, cache = out
                        block_caches.append(cache)
                else:
                    x = out

        if training and not use_cache:
            x, ln_final_cache = self.ln_final.forward(x)
        else:
            x, _ = self.ln_final.forward(x)

        logits = np.dot(x, self.W_out)

        if is_2d:
            logits = logits[0]

        if use_cache:
            return logits, new_kv_caches

        if training:

            caches = (block_caches, ln_final_cache, x, idx)
            return logits, caches
        return logits

    def backward(self, d_logits, caches):

        block_caches, ln_final_cache, x_final, idx = caches


        is_2d = (d_logits.ndim == 2)
        if is_2d:
            d_logits = d_logits[np.newaxis, ...]

        B, S, V = d_logits.shape


        d_logits_flat = d_logits.reshape(B * S, V)
        h_norm_flat = x_final.reshape(B * S, self.embed_dim)

        self.dW_out = np.dot(h_norm_flat.T, d_logits_flat)
        d_x_final = np.dot(d_logits_flat, self.W_out.T).reshape(B, S, self.embed_dim)


        d_x = self.ln_final.backward(d_x_final, ln_final_cache)


        for i in reversed(range(self.num_layers)):
            d_x = self.blocks[i].backward(d_x, block_caches[i])


        self.dtoken_embedding = np.zeros_like(self.token_embedding)
        np.add.at(self.dtoken_embedding, idx, d_x)
        return d_x

    def get_params_and_grads(self):
        pairs = [
            (self.token_embedding, self.dtoken_embedding),
            (self.W_out, self.dW_out),
            (self.ln_final.gamma, self.ln_final.dgamma),
            (self.ln_final.beta, self.ln_final.dbeta),
        ]
        for block in self.blocks:
            pairs.extend([
                (block.Wq, block.dWq),
                (block.Wk, block.dWk),
                (block.Wv, block.dWv),
                (block.Wo, block.dWo),
                (block.ln1.gamma, block.ln1.dgamma),
                (block.ln1.beta, block.ln1.dbeta),
                (block.W1, block.dW1),
                (block.b1, block.db1),
                (block.W2, block.dW2),
                (block.b2, block.db2),
                (block.ln2.gamma, block.ln2.dgamma),
                (block.ln2.beta, block.ln2.dbeta),
            ])
        return pairs

    def load_weights(self, params_dict):

        self.token_embedding = params_dict['token_embedding']
        self.W_out = params_dict['W_out']
        self.ln_final.gamma = params_dict['ln_final_gamma']
        self.ln_final.beta = params_dict['ln_final_beta']
        for i, block in enumerate(self.blocks):
            prefix = f'block_{i}_'
            block.Wq = params_dict[prefix + 'Wq']
            block.Wk = params_dict[prefix + 'Wk']
            block.Wv = params_dict[prefix + 'Wv']
            block.Wo = params_dict[prefix + 'Wo']
            block.ln1.gamma = params_dict[prefix + 'ln1_gamma']
            block.ln1.beta = params_dict[prefix + 'ln1_beta']
            block.ln2.gamma = params_dict[prefix + 'ln2_gamma']
            block.ln2.beta = params_dict[prefix + 'ln2_beta']
            block.W1 = params_dict[prefix + 'W1']
            block.b1 = params_dict[prefix + 'b1']
            block.W2 = params_dict[prefix + 'W2']
            block.b2 = params_dict[prefix + 'b2']