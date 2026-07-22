"""
Pharmakon Training Script – AdamW optimizer, cosine decay with warmup, gradient checkpointing.
Uses the refactored transformer with FlashAttention.
"""

import numpy as np
import time
import os
import gc
import json
import sys
from transformer import PharmakonTransformer

# -----------------------------------------------------------------------------
# Optimizer: AdamW with decoupled weight decay
# -----------------------------------------------------------------------------
class AdamW:
    def __init__(self, model, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        # Fetch parameter shapes to initialize momentum and velocity buffers
        params_grads = model.get_params_and_grads()
        self.m = [np.zeros_like(p) for p, _ in params_grads]
        self.v = [np.zeros_like(p) for p, _ in params_grads]

    def step(self):
        self.t += 1
        params_grads = self.model.get_params_and_grads()
        for i, (param, grad) in enumerate(params_grads):
            # Decoupled weight decay: param = param - lr * weight_decay * param
            param *= (1 - self.lr * self.weight_decay)
            # Adam update
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(np.maximum(v_hat, 0.0)) + self.eps)

    def zero_grad(self):
        for _, grad in self.model.get_params_and_grads():
            if grad is not None:
                grad.fill(0.0)


# -----------------------------------------------------------------------------
# Scheduler: cosine decay with linear warmup
# -----------------------------------------------------------------------------
class CosineDecayWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.step_num = 0
        self.current_lr = optimizer.lr

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            # linear warmup
            lr = self.base_lr * (self.step_num / self.warmup_steps)
        else:
            # cosine decay
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        self.optimizer.lr = lr
        self.current_lr = lr


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char, len(chars)

def get_batch(data, batch_size, seq_len):
    """Randomly sample a batch of sequences."""
    n = len(data)
    idx = np.random.randint(0, n - seq_len - 1, size=(batch_size,))
    x = np.stack([data[i:i+seq_len] for i in idx])
    y = np.stack([data[i+1:i+seq_len+1] for i in idx])
    return x, y


# -----------------------------------------------------------------------------
# Checkpoint Helpers
# -----------------------------------------------------------------------------
def extract_weights(model):
    """Extract model parameters for saving."""
    params_dict = {}
    params_dict["token_embedding"] = model.token_embedding.astype(np.float32)
    params_dict["W_out"] = model.W_out.astype(np.float32)
    params_dict["ln_final_gamma"] = model.ln_final.gamma.astype(np.float32)
    params_dict["ln_final_beta"] = model.ln_final.beta.astype(np.float32)

    for i, block in enumerate(model.blocks):
        prefix = f"block_{i}_"
        params_dict[prefix + "Wq"] = block.Wq.astype(np.float32)
        params_dict[prefix + "Wk"] = block.Wk.astype(np.float32)
        params_dict[prefix + "Wv"] = block.Wv.astype(np.float32)
        params_dict[prefix + "Wo"] = block.Wo.astype(np.float32)
        params_dict[prefix + "ln1_gamma"] = block.ln1.gamma.astype(np.float32)
        params_dict[prefix + "ln1_beta"] = block.ln1.beta.astype(np.float32)
        params_dict[prefix + "ln2_gamma"] = block.ln2.gamma.astype(np.float32)
        params_dict[prefix + "ln2_beta"] = block.ln2.beta.astype(np.float32)
        params_dict[prefix + "W1"] = block.W1.astype(np.float32)
        params_dict[prefix + "b1"] = block.b1.astype(np.float32)
        params_dict[prefix + "W2"] = block.W2.astype(np.float32)
        params_dict[prefix + "b2"] = block.b2.astype(np.float32)
    return params_dict


def save_checkpoint(model, epoch, base_path="checkpoints"):
    os.makedirs(base_path, exist_ok=True)
    slot = epoch % 3
    path = os.path.join(base_path, f"model_slot_{slot}.npz")
    tmp_path = path + ".tmp.npz"
    
    weights = extract_weights(model)
    np.savez_compressed(tmp_path, **weights)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved rolling checkpoint for epoch {epoch} to slot {slot}")


def save_training_state(epoch, step, scheduler_step, current_lr, base_path="checkpoints"):
    os.makedirs(base_path, exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "scheduler_step": scheduler_step,
        "learning_rate": current_lr
    }
    path = os.path.join(base_path, "training_state.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved JSON training state (Epoch {epoch}, Step {step})")


def load_checkpoint(model, epoch, base_path="checkpoints"):
    # Scan rolling slots in reverse order starting from the previous epoch
    for offset in [1, 2, 3]:
        prev_epoch = epoch - offset
        if prev_epoch < 0:
            continue
        slot = prev_epoch % 3
        path = os.path.join(base_path, f"model_slot_{slot}.npz")
        if os.path.exists(path):
            print(f"[Checkpoint] Restoring weights from clean checkpoint: {path} (slot {slot})")
            try:
                weights = dict(np.load(path))
                # Cast parameters to float64
                for key in weights:
                    weights[key] = weights[key].astype(np.float64)
                model.load_weights(weights)
                return True
            except Exception as e:
                print(f"[Checkpoint] Failed to load checkpoint {path}: {e}")
    return False


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def train(
    model, 
    data, # Can be str (raw text) or List[int] (BPE tokens)
    char_to_idx=None, # Only needed if data is str
    epochs=10, 
    batch_size=16, 
    seq_len=64, 
    lr=1e-3, 
    weight_decay=0.01,
    warmup_steps=100,
    use_checkpoint=True,
    resume_state=None
):
    """
    Trains the PharmakonTransformer.
    """
    if isinstance(data, str):
        if char_to_idx is None:
            raise ValueError("char_to_idx required if data is a string.")
        print("[Dataset] Converting text to character IDs...")
        encoded = np.array([char_to_idx[c] for c in data], dtype=np.int32)
    else:
        print("[Dataset] Using pre-encoded BPE tokens...")
        encoded = np.array(data, dtype=np.int32)

    total_batches = (len(encoded) // (batch_size * seq_len)) * epochs
    steps_per_epoch = len(encoded) // (batch_size * seq_len)

    # Initialize optimizer with model references
    optimizer = AdamW(model, lr=lr, weight_decay=weight_decay)
    scheduler = CosineDecayWithWarmup(optimizer, warmup_steps, total_batches)

    logits = None
    caches = None
    logits_flat = None
    targets_flat = None
    max_logits = None
    shifted = None
    exp_shifted = None
    probs = None
    d_logits_flat = None
    d_logits = None

    epoch_losses = []

    print(f"Training {epochs} epochs, {steps_per_epoch} batches/epoch, total steps {total_batches}")

    start_epoch = 1
    start_step = 0
    if resume_state:
        start_epoch = resume_state.get("epoch", 1)
        start_step = resume_state.get("step", 0)
        scheduler.step_num = resume_state.get("scheduler_step", 0)
        print(f"[System] Resuming from epoch {start_epoch}, step {start_step}")

    for epoch in range(start_epoch, epochs+1):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            if epoch == start_epoch and step < start_step:
                continue

            x, y = get_batch(encoded, batch_size, seq_len)

            # Forward (with gradient checkpointing if enabled)
            logits, caches = model.forward(x, training=True, checkpoint=use_checkpoint)
            B, S, V = logits.shape

            # Compute loss and logits gradient
            # Softmax cross-entropy
            logits_flat = logits.reshape(-1, V)
            targets_flat = y.reshape(-1)
            # numerically stable softmax with log
            max_logits = np.max(logits_flat, axis=1, keepdims=True)
            shifted = logits_flat - max_logits
            exp_shifted = np.exp(shifted)
            probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
            # Loss
            loss = -np.mean(np.log(probs[np.arange(B*S), targets_flat] + 1e-12))
            # Gradient of loss w.r.t. logits
            d_logits_flat = probs.copy()
            d_logits_flat[np.arange(B*S), targets_flat] -= 1.0
            d_logits_flat /= (B * S)
            d_logits = d_logits_flat.reshape(B, S, V)

            # Backward pass
            optimizer.zero_grad()
            model.backward(d_logits, caches)

            # Gradient clipping (global L2 norm)
            params_grads = model.get_params_and_grads()
            total_norm = 0.0
            has_nan_or_inf = False
            for _, grad in params_grads:
                if grad is not None:
                    if not np.isfinite(grad).all():
                        has_nan_or_inf = True
                    total_norm += np.sum(grad ** 2)
            total_norm = np.sqrt(total_norm)

            if has_nan_or_inf or np.isnan(total_norm) or np.isinf(total_norm):
                print(f"  [FATAL] Epoch {epoch} | Step {step+1}: NaNs/Infs detected in gradients! Triggering Self-Kill (Exit 88)")
                sys.exit(88)

            max_norm = 1.0
            if total_norm > max_norm:
                scale = max_norm / total_norm
                for _, grad in params_grads:
                    if grad is not None:
                        grad *= scale

            # Update weights and learning rate
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss

            if (step + 1) % 10 == 0 or step == 0:
                print(f"Epoch {epoch} | Step {step+1}/{steps_per_epoch} | Loss: {loss:.4f} | LR: {scheduler.current_lr:.2e}")
                
            if (step + 1) % 1000 == 0:
                print(f"[System] Auto-saving checkpoint at step {step+1}...")
                chk_path = "backend/weights/the_leviathan_checkpoint.npz"
                tmp_chk = chk_path + ".tmp.npz"
                try:
                    weights = extract_weights(model)
                    np.savez_compressed(tmp_chk, **weights)
                    os.replace(tmp_chk, chk_path)
                    save_training_state(epoch, step+1, scheduler.step_num, scheduler.current_lr, "checkpoints")
                except Exception as exc:
                    print(f"  [WARNING] Auto-saving at step {step+1} failed: {exc}")

        avg_loss = epoch_loss / steps_per_epoch
        
        # Check for NaNs/Infs
        if np.isnan(avg_loss) or np.isinf(avg_loss):
            print(f"Execution halted: Numerical divergence detected! Epoch loss is {avg_loss}. Triggering Self-Kill (Exit 88)")
            sys.exit(88)

        print(f"Epoch {epoch:2d} | loss: {avg_loss:.4f} | lr: {optimizer.lr:.2e}")
        epoch_losses.append(avg_loss)

        # Save a clean rolling checkpoint
        try:
            save_checkpoint(model, epoch, base_path="checkpoints")
            save_training_state(epoch, 0, scheduler.step_num, optimizer.lr, "checkpoints")
        except Exception as exc:
            print(f"  [WARNING] Failed to save rolling checkpoint: {exc}")

        # Force garbage collection at the end of the epoch
        logits = None
        caches = None
        logits_flat = None
        targets_flat = None
        max_logits = None
        shifted = None
        exp_shifted = None
        probs = None
        d_logits_flat = None
        d_logits = None
        gc.collect()
        
    print("Training complete.")
    return epoch_losses