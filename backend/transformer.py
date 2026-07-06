"""
Pharmakon Transformer – Pure NumPy, Character-Level, RoPE-based Decoder
Implements the exact Spec v3.1: pre-norm, multi-head causal attention,
contiguity optimizations, exact LayerNorm backward, and RoPE forward/backward in float64.
"""

import numpy as np

# ---------- Utility Functions ----------
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
    freqs = np.outer(t, freqs)  # (max_seq_len, head_dim//2)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    return cos, sin

def rotate_half(x):
    """Rotates half the hidden dims of the input.
    x: shape (..., dim)
    """
    dim = x.shape[-1]
    x1 = x[..., :dim//2]
    x2 = x[..., dim//2:]
    return np.concatenate([-x2, x1], axis=-1)

def apply_rope(q, k, cos, sin, offset=0):
    """Applies RoPE to query and key tensors.
    q, k: shape (batch, n_heads, seq_len, head_dim)
    cos, sin: shape (max_seq_len, head_dim//2)
    """
    seq_len = q.shape[2]
    cos_part = cos[offset:offset+seq_len, :][None, None, :, :]  # (1, 1, seq_len, head_dim//2)
    sin_part = sin[offset:offset+seq_len, :][None, None, :, :]  # (1, 1, seq_len, head_dim//2)
    cos_part = np.concatenate([cos_part, cos_part], axis=-1)  # (1, 1, seq_len, head_dim)
    sin_part = np.concatenate([sin_part, sin_part], axis=-1)  # (1, 1, seq_len, head_dim)
    
    q_rot = q * cos_part + rotate_half(q) * sin_part
    k_rot = k * cos_part + rotate_half(k) * sin_part
    return q_rot, k_rot

def apply_rope_backward(dq_rot, dk_rot, cos, sin, offset=0):
    """Reverses the RoPE rotation for backpropagation.
    dq_rot, dk_rot: shape (batch, n_heads, seq_len, head_dim)
    cos, sin: shape (max_seq_len, head_dim//2)
    """
    seq_len = dq_rot.shape[2]
    cos_part = cos[offset:offset+seq_len, :][None, None, :, :]
    sin_part = sin[offset:offset+seq_len, :][None, None, :, :]
    cos_part = np.concatenate([cos_part, cos_part], axis=-1)
    sin_part = np.concatenate([sin_part, sin_part], axis=-1)
    
    dq = dq_rot * cos_part - rotate_half(dq_rot) * sin_part
    dk = dk_rot * cos_part - rotate_half(dk_rot) * sin_part
    return dq, dk

def get_positional_encoding(seq_len, embed_dim):
    """Precompute sinusoidal positional encodings."""
    pe = np.zeros((seq_len, embed_dim), dtype=np.float64)
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_dim, 2).astype(np.float64) * -(np.log(10000.0) / embed_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

# ---------- Layer Normalization (Forward & Backward) ----------
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
        
        # Exact closed-form derivative
        dx = (dx_hat - np.mean(dx_hat, axis=-1, keepdims=True) - x_hat * np.mean(dx_hat * x_hat, axis=-1, keepdims=True)) / sigma
        
        # Accumulate parameter gradients (summed over all axes except the last feature dimension)
        sum_axes = tuple(range(d_out.ndim - 1))
        self.dgamma = np.sum(d_out * x_hat, axis=sum_axes)
        self.dbeta = np.sum(d_out, axis=sum_axes)
        return dx

# ---------- Transformer Block ----------
class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, max_seq_len=256):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        # Precompute RoPE frequencies
        self.cos, self.sin = precompute_freqs(self.head_dim, max_seq_len)

        # Attention weights
        limit = np.sqrt(6.0 / (embed_dim + embed_dim))
        self.Wq = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)
        self.Wk = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)
        self.Wv = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)
        self.Wo = np.random.uniform(-limit, limit, (embed_dim, embed_dim)).astype(np.float64)

        # Layer norms
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

        # Feed-forward weights
        limit_ff1 = np.sqrt(6.0 / (embed_dim + ff_dim))
        self.W1 = np.random.uniform(-limit_ff1, limit_ff1, (embed_dim, ff_dim)).astype(np.float64)
        self.b1 = np.zeros(ff_dim, dtype=np.float64)
        
        limit_ff2 = np.sqrt(6.0 / (ff_dim + embed_dim))
        self.W2 = np.random.uniform(-limit_ff2, limit_ff2, (ff_dim, embed_dim)).astype(np.float64)
        self.b2 = np.zeros(embed_dim, dtype=np.float64)

        # Gradients
        self.dWq, self.dWk, self.dWv, self.dWo = None, None, None, None
        self.dW1, self.db1, self.dW2, self.db2 = None, None, None, None

    def forward(self, x, training=False):
        # x: (batch, seq_len, embed_dim)
        B, S, D = x.shape

        # Pre-norm for attention
        h_norm1, ln1_cache = self.ln1.forward(x)

        # Project Q, K, V
        Q_proj = np.dot(h_norm1, self.Wq)  # (B, S, D)
        K_proj = np.dot(h_norm1, self.Wk)
        V_proj = np.dot(h_norm1, self.Wv)

        # Split into heads & enforce contiguity
        Q = np.ascontiguousarray(Q_proj.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
        K = np.ascontiguousarray(K_proj.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
        V = np.ascontiguousarray(V_proj.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))

        # Apply RoPE
        Q_rot, K_rot = apply_rope(Q, K, self.cos, self.sin)

        # Causal Attention Matrix
        scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        mask = create_causal_mask(S)
        scores = scores + mask[None, None, :, :]  # Broad-casted to (B, H, S, S)
        attn_weights_raw = softmax(scores, axis=-1)

        # Attention Dropout
        if training and self.dropout > 0:
            drop_mask = (np.random.rand(*attn_weights_raw.shape) > self.dropout).astype(attn_weights_raw.dtype)
            attn_weights = attn_weights_raw * drop_mask / (1.0 - self.dropout)
        else:
            drop_mask = None
            attn_weights = attn_weights_raw

        # Output aggregation & contiguity
        head_out = np.matmul(attn_weights, V)  # (B, H, S, head_dim)
        concat = np.ascontiguousarray(head_out.transpose(0, 2, 1, 3).reshape(B, S, D))
        attn_out = np.dot(concat, self.Wo)

        # Post-attention Dropout
        if training and self.dropout > 0:
            attn_proj_mask = (np.random.rand(*attn_out.shape) > self.dropout).astype(attn_out.dtype)
            attn_out_dropped = attn_out * attn_proj_mask / (1.0 - self.dropout)
        else:
            attn_proj_mask = None
            attn_out_dropped = attn_out

        # First residual connection
        x_resid1 = x + attn_out_dropped

        # Pre-norm for FFN
        h_norm2, ln2_cache = self.ln2.forward(x_resid1)

        # FFN: linear -> relu -> linear
        ff1_pre = np.dot(h_norm2, self.W1) + self.b1
        ff1 = np.maximum(0, ff1_pre)
        ff2 = np.dot(ff1, self.W2) + self.b2

        # Post-FFN Dropout
        if training and self.dropout > 0:
            ffn_drop_mask = (np.random.rand(*ff2.shape) > self.dropout).astype(ff2.dtype)
            ff2_dropped = ff2 * ffn_drop_mask / (1.0 - self.dropout)
        else:
            ffn_drop_mask = None
            ff2_dropped = ff2

        # Second residual connection
        out = x_resid1 + ff2_dropped

        if training:
            cache = {
                'x': x,
                'h_norm1': h_norm1,
                'ln1_cache': ln1_cache,
                'Q_proj': Q_proj,
                'K_proj': K_proj,
                'V_proj': V_proj,
                'Q': Q,
                'K': K,
                'V': V,
                'Q_rot': Q_rot,
                'K_rot': K_rot,
                'scores': scores,
                'attn_weights_raw': attn_weights_raw,
                'attn_weights': attn_weights,
                'drop_mask': drop_mask,
                'head_out': head_out,
                'concat': concat,
                'attn_out': attn_out,
                'attn_proj_mask': attn_proj_mask,
                'x_resid1': x_resid1,
                'h_norm2': h_norm2,
                'ln2_cache': ln2_cache,
                'ff1_pre': ff1_pre,
                'ff1': ff1,
                'ff2': ff2,
                'ffn_drop_mask': ffn_drop_mask
            }
            return out, cache
        return out

    def backward(self, d_out, cache):
        """Backward pass through a single block."""
        B, S, D = d_out.shape

        # Retrieve cached forward matrices
        x = cache['x']
        h_norm1 = cache['h_norm1']
        ln1_cache = cache['ln1_cache']
        Q_proj = cache['Q_proj']
        K_proj = cache['K_proj']
        V_proj = cache['V_proj']
        Q = cache['Q']
        K = cache['K']
        V = cache['V']
        Q_rot = cache['Q_rot']
        K_rot = cache['K_rot']
        attn_weights_raw = cache['attn_weights_raw']
        attn_weights = cache['attn_weights']
        drop_mask = cache['drop_mask']
        head_out = cache['head_out']
        concat = cache['concat']
        attn_proj_mask = cache['attn_proj_mask']
        x_resid1 = cache['x_resid1']
        h_norm2 = cache['h_norm2']
        ln2_cache = cache['ln2_cache']
        ff1_pre = cache['ff1_pre']
        ff1 = cache['ff1']
        ffn_drop_mask = cache['ffn_drop_mask']

        # ----------------------------------------------------
        # BACKPROP THROUGH FEED-FORWARD LAYER (FFN)
        # ----------------------------------------------------
        # Upstream gradient from block output into the FFN residual
        d_ff2_dropped = d_out  # residual skip-connection copy
        
        if ffn_drop_mask is not None:
            d_ff2 = d_ff2_dropped * ffn_drop_mask / (1.0 - self.dropout)
        else:
            d_ff2 = d_ff2_dropped
            
        d_ff2_flat = d_ff2.reshape(B * S, D)
        ff1_flat = ff1.reshape(B * S, self.ff_dim)
        
        self.dW2 = np.dot(ff1_flat.T, d_ff2_flat)
        self.db2 = np.sum(d_ff2_flat, axis=0)
        
        d_ff1 = np.dot(d_ff2_flat, self.W2.T).reshape(B, S, self.ff_dim)
        
        # ReLU derivative
        d_ff1[ff1_pre <= 0] = 0.0
        
        d_ff1_flat = d_ff1.reshape(B * S, self.ff_dim)
        h_norm2_flat = h_norm2.reshape(B * S, D)
        
        self.dW1 = np.dot(h_norm2_flat.T, d_ff1_flat)
        self.db1 = np.sum(d_ff1_flat, axis=0)
        
        d_h_norm2 = np.dot(d_ff1_flat, self.W1.T).reshape(B, S, D)
        
        # LayerNorm 2 Backward
        d_x_ffn_input = self.ln2.backward(d_h_norm2, ln2_cache)
        
        # Combine residual gradient flows (FFN input path + FFN output skip)
        d_x_resid1 = d_out + d_x_ffn_input

        # ----------------------------------------------------
        # BACKPROP THROUGH ATTENTION LAYER
        # ----------------------------------------------------
        d_attn_out_dropped = d_x_resid1
        
        if attn_proj_mask is not None:
            d_attn_out = d_attn_out_dropped * attn_proj_mask / (1.0 - self.dropout)
        else:
            d_attn_out = d_attn_out_dropped
            
        d_attn_out_flat = d_attn_out.reshape(B * S, D)
        concat_flat = concat.reshape(B * S, D)
        
        self.dWo = np.dot(concat_flat.T, d_attn_out_flat)
        d_concat = np.dot(d_attn_out_flat, self.Wo.T).reshape(B, S, D)
        
        # Reverse heads split & merge transposes
        d_head_out = np.ascontiguousarray(d_concat.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3))
        
        # matmul: head_out = attn_weights @ V
        dV = np.matmul(attn_weights.transpose(0, 1, 3, 2), d_head_out)
        d_attn_weights = np.matmul(d_head_out, V.transpose(0, 1, 3, 2))
        
        if drop_mask is not None:
            d_attn_weights = d_attn_weights * drop_mask / (1.0 - self.dropout)
            
        # Softmax backward
        d_scores = attn_weights_raw * (d_attn_weights - np.sum(attn_weights_raw * d_attn_weights, axis=-1, keepdims=True))
        
        # Block causal mask gradient flows
        mask = create_causal_mask(S)
        d_scores = np.where(mask[None, None, :, :] == -1e10, 0.0, d_scores)
        d_scores /= np.sqrt(self.head_dim)
        
        # matmul: scores = Q_rot @ K_rot^T
        dQ_rot = np.matmul(d_scores, K_rot)
        dK_rot = np.matmul(d_scores.transpose(0, 1, 3, 2), Q_rot)
        
        # Reverse RoPE rotation
        dQ, dK = apply_rope_backward(dQ_rot, dK_rot, self.cos, self.sin)
        
        # Force contiguity before project maps
        dQ = np.ascontiguousarray(dQ.transpose(0, 2, 1, 3).reshape(B * S, D))
        dK = np.ascontiguousarray(dK.transpose(0, 2, 1, 3).reshape(B * S, D))
        dV = np.ascontiguousarray(dV.transpose(0, 2, 1, 3).reshape(B * S, D))
        
        h_norm1_flat = h_norm1.reshape(B * S, D)
        
        self.dWq = np.dot(h_norm1_flat.T, dQ)
        self.dWk = np.dot(h_norm1_flat.T, dK)
        self.dWv = np.dot(h_norm1_flat.T, dV)
        
        d_h_norm1 = np.dot(dQ, self.Wq.T) + np.dot(dK, self.Wk.T) + np.dot(dV, self.Wv.T)
        d_h_norm1 = d_h_norm1.reshape(B, S, D)
        
        # LayerNorm 1 Backward
        d_x_att_input = self.ln1.backward(d_h_norm1, ln1_cache)
        
        # Final residual join for the block input
        dx = d_x_resid1 + d_x_att_input
        return dx

# ---------- Full Pharmakon Transformer ----------
class PharmakonTransformer:
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, ff_dim=128,
                 num_layers=2, max_seq_len=256, dropout=0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Token embedding
        limit = np.sqrt(6.0 / (vocab_size + embed_dim))
        self.token_embedding = np.random.uniform(-limit, limit, (vocab_size, embed_dim)).astype(np.float64)

        # Transformer layers
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, dropout=dropout, max_seq_len=max_seq_len)
                       for _ in range(num_layers)]

        # Final layer norm
        self.ln_final = LayerNorm(embed_dim)

        # Output projection
        self.W_out = np.random.uniform(-limit, limit, (embed_dim, vocab_size)).astype(np.float64)

        # Parameter gradients
        self.dtoken_embedding = None
        self.dW_out = None

    def forward(self, idx, training=False):
        """idx: (batch, seq_len) integer token indices.
        Returns: logits (batch, seq_len, vocab_size) and caches (if training=True)
        """
        # Ensure 3D input format: shape (B, S)
        is_2d = (idx.ndim == 1)
        if is_2d:
            idx = idx[np.newaxis, :]
            
        B, S = idx.shape

        x = self.token_embedding[idx]  # (B, S, embed_dim)

        # Apply positional encodings
        pe = get_positional_encoding(S, self.embed_dim)
        x = x + pe[np.newaxis, :, :]

        block_caches = []
        for block in self.blocks:
            if training:
                x, cache = block.forward(x, training=True)
                block_caches.append(cache)
            else:
                x = block.forward(x, training=False)

        # Final normalization
        if training:
            x, ln_final_cache = self.ln_final.forward(x)
        else:
            x, _ = self.ln_final.forward(x)

        # Final linear projection
        logits = np.dot(x, self.W_out)  # (B, S, vocab_size)

        if is_2d:
            logits = logits[0]  # Return 2D if inputs were 2D

        if training:
            # Package caches list along with input indices for embedding gradients
            caches = (block_caches, ln_final_cache, x, idx)
            return logits, caches
        return logits

    def backward(self, d_logits, caches):
        """Backward pass through the entire network."""
        block_caches, ln_final_cache, x_final, idx = caches
        
        # Ensure 3D shape format
        is_2d = (d_logits.ndim == 2)
        if is_2d:
            d_logits = d_logits[np.newaxis, ...]
            
        B, S, V = d_logits.shape

        # Output projection backward
        d_logits_flat = d_logits.reshape(B * S, V)
        h_norm_flat = x_final.reshape(B * S, self.embed_dim)
        
        self.dW_out = np.dot(h_norm_flat.T, d_logits_flat)
        d_x_final = np.dot(d_logits_flat, self.W_out.T).reshape(B, S, self.embed_dim)

        # Final LayerNorm backward
        d_x = self.ln_final.backward(d_x_final, ln_final_cache)

        # Backprop through blocks in reverse order
        for i in reversed(range(self.num_layers)):
            d_x = self.blocks[i].backward(d_x, block_caches[i])

        # Token embedding backward using scatter-add
        self.dtoken_embedding = np.zeros_like(self.token_embedding)
        np.add.at(self.dtoken_embedding, idx, d_x)
        return d_x

    def get_params_and_grads(self):
        """Collects parameter matrices and accumulated gradient matrices for training."""
        pairs = []
        pairs.append((self.token_embedding, self.dtoken_embedding))
        pairs.append((self.W_out, self.dW_out))
        pairs.append((self.ln_final.gamma, self.ln_final.dgamma))
        pairs.append((self.ln_final.beta, self.ln_final.dbeta))
        for block in self.blocks:
            pairs.append((block.Wq, block.dWq))
            pairs.append((block.Wk, block.dWk))
            pairs.append((block.Wv, block.dWv))
            pairs.append((block.Wo, block.dWo))
            pairs.append((block.ln1.gamma, block.ln1.dgamma))
            pairs.append((block.ln1.beta, block.ln1.dbeta))
            pairs.append((block.W1, block.dW1))
            pairs.append((block.b1, block.db1))
            pairs.append((block.W2, block.dW2))
            pairs.append((block.b2, block.db2))
            pairs.append((block.ln2.gamma, block.ln2.dgamma))
            pairs.append((block.ln2.beta, block.ln2.dbeta))
        return pairs

    def load_weights(self, params_dict):
        """Loads weight configuration maps."""
        self.token_embedding = params_dict['token_embedding']
        self.W_out = params_dict['W_out']
        self.ln_final.gamma = params_dict['ln_final_gamma']
        self.ln_final.beta = params_dict['ln_final_beta']
        for i, block in enumerate(self.blocks):
            prefix = f'block_{i}_'
            block.Wq = params_dict[prefix+'Wq']
            block.Wk = params_dict[prefix+'Wk']
            block.Wv = params_dict[prefix+'Wv']
            block.Wo = params_dict[prefix+'Wo']
            block.ln1.gamma = params_dict[prefix+'ln1_gamma']
            block.ln1.beta = params_dict[prefix+'ln1_beta']
            block.ln2.gamma = params_dict[prefix+'ln2_gamma']
            block.ln2.beta = params_dict[prefix+'ln2_beta']
            block.W1 = params_dict[prefix+'W1']
            block.b1 = params_dict[prefix+'b1']
            block.W2 = params_dict[prefix+'W2']
            block.b2 = params_dict[prefix+'b2']
