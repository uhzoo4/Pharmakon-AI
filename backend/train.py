"""
Pharmakon Training Script – AdamW optimizer, cosine decay with warmup, gradient checkpointing.
Uses the refactored transformer with FlashAttention.
"""

import numpy as np
import time
from backend.transformer import PharmakonTransformer

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
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

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
# Training loop
# -----------------------------------------------------------------------------
def train(model, data, char_to_idx, epochs=50, batch_size=32, seq_len=64,
          lr=3e-4, weight_decay=0.01, warmup_steps=100, use_checkpoint=True):
    vocab_size = len(char_to_idx)
    # Prepare integer data
    encoded = np.array([char_to_idx[ch] for ch in data], dtype=np.int32)
    total_batches = (len(encoded) // (batch_size * seq_len)) * epochs
    steps_per_epoch = len(encoded) // (batch_size * seq_len)

    # Initialize optimizer with model references
    optimizer = AdamW(model, lr=lr, weight_decay=weight_decay)
    scheduler = CosineDecayWithWarmup(optimizer, warmup_steps, total_batches)

    print(f"Training {epochs} epochs, {steps_per_epoch} batches/epoch, total steps {total_batches}")

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
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
            for _, grad in params_grads:
                total_norm += np.sum(grad ** 2)
            total_norm = np.sqrt(total_norm)
            max_norm = 1.0
            if total_norm > max_norm:
                scale = max_norm / total_norm
                for _, grad in params_grads:
                    grad *= scale

            # Update weights and learning rate
            optimizer.step()
            scheduler.step()

            epoch_loss += loss

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch:2d} | loss: {avg_loss:.4f} | lr: {optimizer.lr:.2e}")

    print("Training complete.")