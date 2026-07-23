from __future__ import annotations

import functools
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from transformer import pharmakon_forward

_tree_map = getattr(getattr(jax, 'tree', None), 'map', jax.tree_util.tree_map)
_tree_leaves = getattr(getattr(jax, 'tree', None), 'leaves', jax.tree_util.tree_leaves)

# -----------------------------------------------------------------------------
# JAX Optimizer: AdamW
# -----------------------------------------------------------------------------
def init_adamw_state(params: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize m and v buffers for all parameters."""
    m = _tree_map(lambda x: jnp.zeros_like(x), params)
    v = _tree_map(lambda x: jnp.zeros_like(x), params)
    return {'m': m, 'v': v, 't': 0}

def adamw_update(params: Dict[str, Any], grads: Dict[str, Any], opt_state: Dict[str, Any], lr: float, weight_decay: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t = opt_state['t'] + 1
    
    def update_param(p, g, m, v):
        # Decoupled weight decay
        p = p * (1 - lr * weight_decay)
        # Adam update
        m_new = beta1 * m + (1 - beta1) * g
        v_new = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        p_new = p - lr * m_hat / (jnp.sqrt(jnp.maximum(v_hat, 0.0)) + eps)
        return p_new, m_new, v_new

    new_params = {}
    new_m = {}
    new_v = {}
    for k in params:
        p, m, v = update_param(params[k], grads[k], opt_state['m'][k], opt_state['v'][k])
        new_params[k] = p
        new_m[k] = m
        new_v[k] = v
    
    new_opt_state = {'m': new_m, 'v': new_v, 't': t}
    return new_params, new_opt_state

# -----------------------------------------------------------------------------
# Scheduler: cosine decay with linear warmup
# -----------------------------------------------------------------------------
class CosineDecayWithWarmup:
    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.step_num = 0
        self.current_lr = base_lr

    def step(self) -> float:
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            # linear warmup
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            # cosine decay
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        self.current_lr = lr
        return lr


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)

def get_batch(data: Any, batch_size: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data)
    idx = np.random.randint(0, n - seq_len - 1, size=(batch_size,))
    x = np.stack([data[i:i+seq_len] for i in idx])
    y = np.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y

def extract_weights(model: Any) -> Dict[str, Any]:
    """Extract model parameters for saving as numpy arrays."""
    return {k: np.array(v, dtype=np.float32) for k, v in model.params.items()}

# -----------------------------------------------------------------------------
# JIT Compiled Train Step
# -----------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=['num_heads', 'head_dim', 'num_layers', 'dropout'], donate_argnums=(0, 1))
def train_step_jit(params, opt_state, x, y, cos, sin, num_heads, head_dim, num_layers, dropout, rng, lr, weight_decay):
    
    def loss_fn(p, rng_key):
        logits = pharmakon_forward(
            p, x, cos, sin, num_heads, head_dim, num_layers,
            use_cache=False, rng=rng_key, dropout=dropout
        )
        B, S, V = logits.shape
        logits_flat = logits.reshape(B*S, V)
        y_flat = y.reshape(B*S)
        
        # Cross entropy
        max_logits = jnp.max(logits_flat, axis=-1, keepdims=True)
        shifted = logits_flat - max_logits
        log_probs = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
        
        target_log_probs = jnp.take_along_axis(log_probs, y_flat[:, None], axis=-1)
        loss = -jnp.mean(target_log_probs)
        return loss

    rng, step_rng = jax.random.split(rng)
    loss, grads = jax.value_and_grad(loss_fn)(params, step_rng)
    
    # Global gradient clipping
    def get_sq(g): return jnp.sum(g ** 2)
    sq_tree = _tree_map(get_sq, grads)
    total_norm = jnp.sqrt(jnp.sum(jnp.array(_tree_leaves(sq_tree))))
    
    # Clip if norm > 1.0
    scale = jnp.where(total_norm > 1.0, 1.0 / total_norm, 1.0)
    grads = _tree_map(lambda g: g * scale, grads)
    
    new_params, new_opt_state = adamw_update(params, grads, opt_state, lr, weight_decay)
    
    return loss, new_params, new_opt_state, rng, total_norm

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train(
    model, 
    data, 
    char_to_idx=None, 
    epochs=10, 
    batch_size=16, 
    seq_len=64, 
    lr=1e-3, 
    weight_decay=0.01,
    warmup_steps=100,
    use_checkpoint=True,
    resume_state=None
):
    if isinstance(data, str):
        if char_to_idx is None:
            raise ValueError("char_to_idx required if data is a string.")
        encoded = np.array([char_to_idx[c] for c in data], dtype=np.int32)
    else:
        encoded = np.array(data, dtype=np.int32)

    total_batches = (len(encoded) // (batch_size * seq_len)) * epochs
    steps_per_epoch = len(encoded) // (batch_size * seq_len)

    opt_state = init_adamw_state(model.params)
    scheduler = CosineDecayWithWarmup(lr, warmup_steps, total_batches)
    
    # Initialize JAX rng
    rng = jax.random.PRNGKey(42)
    
    print(f"Training {epochs} epochs, {steps_per_epoch} batches/epoch, total steps {total_batches}")

    start_epoch = 1
    start_step = 0
    if resume_state:
        start_epoch = resume_state.get("epoch", 1)
        start_step = resume_state.get("step", 0)
        scheduler.step_num = resume_state.get("scheduler_step", 0)

    epoch_losses: list[float] = []
    for epoch in range(start_epoch, epochs+1):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            if epoch == start_epoch and step < start_step:
                continue

            x_np, y_np = get_batch(encoded, batch_size, seq_len)
            x = jnp.array(x_np)
            y = jnp.array(y_np)
            
            current_lr = scheduler.current_lr
            
            # Execute compiled JAX step
            loss, model.params, opt_state, rng, norm = train_step_jit(
                model.params, opt_state, x, y, 
                model.cos, model.sin, 
                model.num_heads, model.head_dim, model.num_layers,
                model.dropout, rng, current_lr, weight_decay
            )
            
            # Convert scalar loss back to python float
            loss_val = float(loss)
            norm_val = float(norm)
            
            if np.isnan(norm_val) or np.isinf(norm_val):
                print(f"  [FATAL] Epoch {epoch} | Step {step+1}: NaNs/Infs detected in gradients! Triggering Self-Kill (Exit 88)")
                sys.exit(88)

            scheduler.step()
            epoch_loss += loss_val

            if (step + 1) % 10 == 0 or step == 0:
                print(f"Epoch {epoch} | Step {step+1}/{steps_per_epoch} | Loss: {loss_val:.4f} | LR: {current_lr:.2e}")
                
            if (step + 1) % 1000 == 0:
                print(f"[System] Auto-saving checkpoint at step {step+1}...")
                chk_path = "backend/weights/the_leviathan_checkpoint.npz"
                tmp_chk = chk_path + ".tmp.npz"
                try:
                    weights = extract_weights(model)
                    np.savez_compressed(tmp_chk, **dict(weights))
                    os.replace(tmp_chk, chk_path)
                except Exception as e:
                    print(f"Failed to auto-save checkpoint: {e}")

        avg_loss = epoch_loss / steps_per_epoch
        epoch_losses.append(avg_loss)
        print(f"=== Epoch {epoch}/{epochs} Complete | Avg Loss: {avg_loss:.4f} ===")
        
        if epoch % 3 == 0:
            slot = epoch % 3
            slot_path = f"backend/weights/model_slot_{slot}.npz"
            tmp = slot_path + ".tmp.npz"
            print(f"[System] Saving robust checkpoint to slot {slot} ({slot_path})...")
            try:
                weights = extract_weights(model)
                np.savez_compressed(tmp, **dict(weights))
                os.replace(tmp, slot_path)
                print(f"  [OK] Checkpoint slot {slot} saved successfully.")
            except Exception as e:
                print(f"  [ERROR] Checkpoint failed: {e}")

    print("Training finished.")
    return epoch_losses