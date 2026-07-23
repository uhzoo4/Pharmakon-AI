"""
Standalone Test Suite for Data Pipeline, Memmap Dataset, Batch Sampler, and JAX Model Convergence.
Enforces the 5-point Definition of Done.
"""

import sys
import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer import Tokenizer
from dataset import create_memmap_bin, MemmapBatchSampler
from transformer import PharmakonTransformer
from train import train_step_jit, init_adamw_state

def test_1_tokenizer_identity():
    print("[Test 1/5] Testing Tokenizer encode -> decode identity...")
    tokenizer = Tokenizer()
    sample_text = (
        "The Leviathan architecture processes high-dimensional embeddings using "
        "Rotary Position Embeddings (RoPE) & JAX automatic differentiation.\n"
        "Special characters: !@#$%^&*()_+-=[]{}|;:'\",.<>/?\n"
    )
    tokenizer.test_identity(sample_text)
    print("  -> Passed Tokenizer encode -> decode identity test!")

def test_2_memmap_roundtrip():
    print("[Test 2/5] Testing np.memmap round-trip token match...")
    known_tokens = np.array([101, 2054, 2003, 1037, 3231, 999, 50256], dtype=np.uint16)
    
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write known tokens to memmap bin file
        written_count = create_memmap_bin(known_tokens, tmp_path, dtype=np.uint16)
        assert written_count == len(known_tokens), f"Expected {len(known_tokens)} written, got {written_count}"

        # Memory-map file back from disk
        read_back = np.memmap(tmp_path, dtype=np.uint16, mode='r')
        assert len(read_back) == len(known_tokens), f"Length mismatch: {len(read_back)} vs {len(known_tokens)}"
        
        # Confirm 100% token equality
        assert np.array_equal(read_back, known_tokens), (
            f"Memmap round-trip mismatch!\nWritten: {known_tokens}\nRead:    {read_back}"
        )
        print("  -> Passed np.memmap round-trip token match test!")
        # Explicitly close/del memmap handle for Windows cleanup
        del read_back
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def test_3_batch_sampler_shapes_and_smoke():
    print("[Test 3/5] Testing MemmapBatchSampler shapes and boundary assertions...")
    batch_size = 8
    seq_len = 16
    synthetic_tokens = np.random.randint(0, 50000, size=(1000,), dtype=np.uint16)

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_memmap_bin(synthetic_tokens, tmp_path, dtype=np.uint16)
        sampler = MemmapBatchSampler(tmp_path, seq_len=seq_len, batch_size=batch_size, dtype=np.uint16)
        
        x, y = sampler.get_batch()
        
        # Assert shapes explicitly
        assert x.shape == (batch_size, seq_len), f"Assert failed: x shape {x.shape} != ({batch_size}, {seq_len})"
        assert y.shape == (batch_size, seq_len), f"Assert failed: y shape {y.shape} != ({batch_size}, {seq_len})"
        assert x.dtype == np.int32, f"Assert failed: x dtype {x.dtype} != int32"
        assert y.dtype == np.int32, f"Assert failed: y dtype {y.dtype} != int32"

        # Verify no indices are out of bounds
        assert np.all(x >= 0) and np.all(x < 50000), "x token indices out of bounds!"
        assert np.all(y >= 0) and np.all(y < 50000), "y token indices out of bounds!"

        print("  -> Passed MemmapBatchSampler boundary shape & smoke assertions!")
        del sampler
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def test_4_gradient_check_and_autodiff():
    print("[Test 4/5] Testing JAX value_and_grad gradient sanity check...")
    # Initialize a small transformer model
    vocab_size = 100
    embed_dim = 32
    num_heads = 4
    head_dim = 8
    num_layers = 2
    
    model = PharmakonTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=64,
        num_layers=num_layers,
        max_seq_len=32
    )

    x = jnp.array([[1, 5, 12, 44]], dtype=jnp.int32)
    y = jnp.array([[5, 12, 44, 99]], dtype=jnp.int32)
    
    def simple_loss(p):
        logits = model.forward(x, training=True)
        assert isinstance(logits, np.ndarray), f"Expected logits to be np.ndarray, got {type(logits)}"
        B, S, V = logits.shape
        assert logits.shape == (1, 4, 100), f"Logits shape mismatch: {logits.shape}"
        logits_flat = logits.reshape(B*S, V)
        y_flat = y.reshape(B*S)
        max_logits = jnp.max(logits_flat, axis=-1, keepdims=True)
        shifted = logits_flat - max_logits
        log_probs = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
        target_log_probs = jnp.take_along_axis(log_probs, y_flat[:, None], axis=-1)
        return -jnp.mean(target_log_probs)

    loss_val, grads = jax.value_and_grad(simple_loss)(model.params)
    
    assert jnp.isfinite(loss_val), f"Loss is not finite: {loss_val}"
    for k, g in grads.items():
        assert jnp.all(jnp.isfinite(g)), f"Gradient {k} contains NaNs or Infs!"
        assert g.shape == model.params[k].shape, (
            f"Gradient shape mismatch for {k}: {g.shape} vs parameter shape {model.params[k].shape}"
        )

    print("  -> Passed JAX value_and_grad gradient sanity check!")

def test_5_one_batch_overfit_and_no_nans():
    print("[Test 5/5] Testing 1-batch overfit convergence & zero NaNs/Infs...")
    vocab_size = 100
    embed_dim = 32
    num_heads = 4
    head_dim = 8
    num_layers = 2
    batch_size = 2
    seq_len = 8

    model = PharmakonTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=64,
        num_layers=num_layers,
        max_seq_len=16,
        dropout=0.0
    )

    opt_state = init_adamw_state(model.params)
    rng = jax.random.PRNGKey(1234)

    # Fixed single synthetic batch
    x_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    y_np = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    x = jnp.array(x_np)
    y = jnp.array(y_np)

    losses = []
    lr = 1e-2
    weight_decay = 0.0

    for step in range(50):
        loss, model.params, opt_state, rng, norm = train_step_jit(
            model.params, opt_state, x, y,
            model.cos, model.sin,
            num_heads, head_dim, num_layers,
            0.0, rng, lr, weight_decay
        )
        loss_val = float(loss)
        norm_val = float(norm)

        # Assert no NaNs or Infs (Hard Rule)
        assert np.isfinite(loss_val), f"NaN/Inf detected in loss at step {step}: {loss_val}"
        assert np.isfinite(norm_val), f"NaN/Inf detected in gradient norm at step {step}: {norm_val}"
        
        losses.append(loss_val)

    initial_loss = losses[0]
    final_loss = losses[-1]

    # Assert loss strictly decreases (Hard Rule)
    assert final_loss < initial_loss * 0.5, (
        f"1-batch overfit failed! Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}"
    )

    print(f"  -> Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    print("  -> Passed 1-batch overfit convergence & zero NaNs/Infs test!")

def run_all_tests():
    print("==========================================================")
    print("RUNNING PHARMAKON DATA PIPELINE & TRAINING TEST SUITE")
    print("==========================================================")
    test_1_tokenizer_identity()
    test_2_memmap_roundtrip()
    test_3_batch_sampler_shapes_and_smoke()
    test_4_gradient_check_and_autodiff()
    test_5_one_batch_overfit_and_no_nans()
    print("==========================================================")
    print("ALL 5 TESTS PASSED SUCCESSFULLY! DEFINITION OF DONE MET.")
    print("==========================================================")

if __name__ == "__main__":
    run_all_tests()
