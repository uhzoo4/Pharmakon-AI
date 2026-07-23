import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def create_causal_mask(seq_len):
    mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.float32), k=1) * -1e10
    return mask

def precompute_freqs(head_dim, max_seq_len, base=10000.0):
    freqs = 1.0 / (base ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    return cos, sin

def rotate_half(x):
    dim = x.shape[-1]
    x1 = x[..., :dim // 2]
    x2 = x[..., dim // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rope(q, k, cos, sin, offset=0):
    seq_len = q.shape[2]
    cos_part = jax.lax.dynamic_slice(cos, (offset, 0), (seq_len, cos.shape[1]))[None, None, :, :]
    sin_part = jax.lax.dynamic_slice(sin, (offset, 0), (seq_len, sin.shape[1]))[None, None, :, :]
    
    cos_part = jnp.concatenate([cos_part, cos_part], axis=-1)
    sin_part = jnp.concatenate([sin_part, sin_part], axis=-1)

    q_rot = q * cos_part + rotate_half(q) * sin_part
    k_rot = k * cos_part + rotate_half(k) * sin_part
    return q_rot, k_rot

def layer_norm_forward(x, gamma, beta, eps=1e-8):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_hat + beta

# -----------------------------------------------------------------------------
# Pure Functional Transformer Forward Pass
# -----------------------------------------------------------------------------
def pharmakon_forward(
    params: Dict[str, jnp.ndarray], 
    idx: jnp.ndarray, 
    cos: jnp.ndarray, 
    sin: jnp.ndarray,
    num_heads: int, 
    head_dim: int, 
    num_layers: int, 
    kv_caches=None, 
    use_cache: bool = False, 
    rng=None, 
    dropout: float = 0.0
):
    # 1. Token embedding
    x = jnp.take(params['token_embedding'], idx, axis=0)
    
    B, S, D = x.shape
    
    mask = None if (use_cache or S == 1) else create_causal_mask(S)
    
    new_kv_caches = [] if use_cache else None
    
    # 2. Blocks
    for i in range(num_layers):
        prefix = f"block_{i}_"
        
        # Pre-norm
        h_norm1 = layer_norm_forward(x, params[prefix + "ln1_gamma"], params[prefix + "ln1_beta"])
        
        # QKV
        Q_proj = jnp.dot(h_norm1, params[prefix + "Wq"])
        K_proj = jnp.dot(h_norm1, params[prefix + "Wk"])
        V_proj = jnp.dot(h_norm1, params[prefix + "Wv"])
        
        # Split heads
        Q = Q_proj.reshape(B, S, num_heads, head_dim).transpose((0, 2, 1, 3))
        K = K_proj.reshape(B, S, num_heads, head_dim).transpose((0, 2, 1, 3))
        V = V_proj.reshape(B, S, num_heads, head_dim).transpose((0, 2, 1, 3))
        
        if use_cache:
            assert new_kv_caches is not None
            offset = 0 if kv_caches is None else kv_caches[i]['offset']
            Q_rot, K_rot = apply_rope(Q, K, cos, sin, offset=offset)
            
            if kv_caches is not None:
                K_past = kv_caches[i]['K']
                V_past = kv_caches[i]['V']
                if K_past.shape[2] > 63:
                    K_past = K_past[:, :, -63:, :]
                    V_past = V_past[:, :, -63:, :]
                K_full = jnp.concatenate([K_past, K_rot], axis=2)
                V_full = jnp.concatenate([V_past, V], axis=2)
            else:
                K_full, V_full = K_rot, V
                
            new_kv_caches.append({
                'K': K_full,
                'V': V_full,
                'offset': offset + 1
            })
        else:
            Q_rot, K_rot = apply_rope(Q, K, cos, sin, offset=0)
            K_full, V_full = K_rot, V
            
        # Attention
        scores = jnp.matmul(Q_rot, K_full.transpose((0, 1, 3, 2))) / jnp.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask
            
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        if rng is not None and dropout > 0.0:
            rng, drop_rng = jax.random.split(rng)
            # In JAX, mask out values based on uniform distribution
            keep_prob = 1.0 - dropout
            drop_mask = jax.random.bernoulli(drop_rng, p=keep_prob, shape=attn_weights.shape)
            attn_weights = jnp.where(drop_mask, attn_weights / keep_prob, 0.0)
            
        head_out = jnp.matmul(attn_weights, V_full)
        concat = head_out.transpose((0, 2, 1, 3)).reshape(B, S, D)
        
        attn_out = jnp.dot(concat, params[prefix + "Wo"])
        
        if rng is not None and dropout > 0.0:
            rng, drop_rng = jax.random.split(rng)
            keep_prob = 1.0 - dropout
            drop_mask = jax.random.bernoulli(drop_rng, p=keep_prob, shape=attn_out.shape)
            attn_out = jnp.where(drop_mask, attn_out / keep_prob, 0.0)
            
        x_resid1 = x + attn_out
        
        # FF
        h_norm2 = layer_norm_forward(x_resid1, params[prefix + "ln2_gamma"], params[prefix + "ln2_beta"])
        ff1 = jnp.maximum(0.0, jnp.dot(h_norm2, params[prefix + "W1"]) + params[prefix + "b1"])
        ff2 = jnp.dot(ff1, params[prefix + "W2"]) + params[prefix + "b2"]
        
        if rng is not None and dropout > 0.0:
            rng, drop_rng = jax.random.split(rng)
            keep_prob = 1.0 - dropout
            drop_mask = jax.random.bernoulli(drop_rng, p=keep_prob, shape=ff2.shape)
            ff2 = jnp.where(drop_mask, ff2 / keep_prob, 0.0)
            
        x = x_resid1 + ff2
        
    # Final norm
    x = layer_norm_forward(x, params["ln_final_gamma"], params["ln_final_beta"])
    logits = jnp.dot(x, params["W_out"])
    
    if use_cache:
        return logits, new_kv_caches
    return logits

# -----------------------------------------------------------------------------
# OOP Wrapper for compatibility with main.py
# -----------------------------------------------------------------------------
class PharmakonTransformer:
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, ff_dim=128,
                 num_layers=2, max_seq_len=1024, dropout=0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        
        self.cos, self.sin = precompute_freqs(self.head_dim, max_seq_len)
        self.params = {}
        
        # Initialize default weights (using numpy so we can lazily move to JAX if needed, or directly to JAX)
        import numpy as np
        limit = np.sqrt(6.0 / (vocab_size + embed_dim))
        self.params["token_embedding"] = jnp.array(np.random.uniform(-limit, limit, (vocab_size, embed_dim)).astype(np.float32))
        self.params["W_out"] = jnp.array(np.random.uniform(-limit, limit, (embed_dim, vocab_size)).astype(np.float32))
        self.params["ln_final_gamma"] = jnp.ones(embed_dim, dtype=jnp.float32)
        self.params["ln_final_beta"] = jnp.zeros(embed_dim, dtype=jnp.float32)

        limit_attn = np.sqrt(6.0 / (2.0 * embed_dim))
        limit_ff1 = np.sqrt(6.0 / (embed_dim + ff_dim))
        limit_ff2 = np.sqrt(6.0 / (ff_dim + embed_dim))

        for i in range(num_layers):
            prefix = f"block_{i}_"
            self.params[prefix + "Wq"] = jnp.array(np.random.uniform(-limit_attn, limit_attn, (embed_dim, embed_dim)).astype(np.float32))
            self.params[prefix + "Wk"] = jnp.array(np.random.uniform(-limit_attn, limit_attn, (embed_dim, embed_dim)).astype(np.float32))
            self.params[prefix + "Wv"] = jnp.array(np.random.uniform(-limit_attn, limit_attn, (embed_dim, embed_dim)).astype(np.float32))
            self.params[prefix + "Wo"] = jnp.array(np.random.uniform(-limit_attn, limit_attn, (embed_dim, embed_dim)).astype(np.float32))
            self.params[prefix + "ln1_gamma"] = jnp.ones(embed_dim, dtype=jnp.float32)
            self.params[prefix + "ln1_beta"] = jnp.zeros(embed_dim, dtype=jnp.float32)
            self.params[prefix + "ln2_gamma"] = jnp.ones(embed_dim, dtype=jnp.float32)
            self.params[prefix + "ln2_beta"] = jnp.zeros(embed_dim, dtype=jnp.float32)
            self.params[prefix + "W1"] = jnp.array(np.random.uniform(-limit_ff1, limit_ff1, (embed_dim, ff_dim)).astype(np.float32))
            self.params[prefix + "b1"] = jnp.zeros(ff_dim, dtype=jnp.float32)
            self.params[prefix + "W2"] = jnp.array(np.random.uniform(-limit_ff2, limit_ff2, (ff_dim, embed_dim)).astype(np.float32))
            self.params[prefix + "b2"] = jnp.zeros(embed_dim, dtype=jnp.float32)
        
    def load_weights(self, params_dict: Dict[str, Any]):
        # Convert all numpy arrays to JAX arrays
        self.params = {k: jnp.array(v) for k, v in params_dict.items()}
        
    def forward(self, idx, training=False, use_cache=False, kv_caches=None, checkpoint=False, rng=None):
        # We must convert idx to JAX array if it's numpy
        idx_jnp = jnp.array(idx, dtype=jnp.int32)
        if idx_jnp.ndim == 0:
            idx_jnp = idx_jnp.reshape(1, 1)
        elif idx_jnp.ndim == 1:
            idx_jnp = idx_jnp.reshape(1, -1)
        
        # If training is False, force dropout to 0.0
        current_dropout = self.dropout if training else 0.0
        
        result = pharmakon_forward(
            self.params, 
            idx_jnp, 
            self.cos, 
            self.sin,
            self.num_heads, 
            self.head_dim, 
            self.num_layers, 
            kv_caches=kv_caches,
            use_cache=use_cache,
            rng=rng,
            dropout=current_dropout
        )
        
        if use_cache:
            logits, new_kv_caches = result
            # Convert logits back to numpy for EntmaxSampler
            import numpy as np
            return np.array(logits), new_kv_caches
        
        # Convert logits back to numpy
        import numpy as np
        return np.array(result)

    def get_params_and_grads(self):
        raise NotImplementedError("JAX handles gradients automatically via jax.grad. Remove calls to get_params_and_grads().")