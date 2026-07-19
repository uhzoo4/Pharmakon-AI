"""
Pharmakon – Batch Training Script
==================================
Trains the three personality models using the literary texts from the data/ folder.
Each personality is trained on a curated subset of texts.

Usage:
    python train_personalities.py
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from transformer import PharmakonTransformer
from weight_manager import WeightManager
from clean_corpus import clean_text
import train as train_module

# --- Configuration ---
DATA_DIR = Path(__file__).parent.parent / "data"
WEIGHTS_DIR = Path(__file__).parent / "weights"

VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
VOCAB_SIZE = len(VOCAB_CHARS)
char_to_idx = {char: idx for idx, char in enumerate(VOCAB_CHARS)}

# Define which texts belong to which personality
PERSONALITY_TEXTS = {
    "kafkaesque": [
        "The Trial.txt",
        "Metamorphosis.txt",
        "Die Verwandlung.txt",
    ],
    "camus_stranger": [
        "camus_stranger.txt",
    ],
    "dark_romance": [
        "Wuthering Heights.txt",
        "The Monk A Romance.txt",
        "The Mysteries of Udolpho.txt",
        "Therèse Raquin.txt",
        "White nights.txt",
    ],
}

# Training hyperparameters
EPOCHS = 30         # Character-level models need many epochs to learn word structure
BATCH_SIZE = 16
SEQ_LEN = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_CHARS = 200_000  # Limit corpus size per personality to keep training manageable


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


def load_and_clean_texts(filenames):
    """Load and concatenate multiple text files, then clean."""
    combined = ""
    for fname in filenames:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        raw = fpath.read_text(encoding="utf-8", errors="ignore")
        combined += raw + "\n\n"
        print(f"  [OK] Loaded {fname} ({len(raw):,} chars)")
    
    # Clean the combined text
    cleaned, dropped = clean_text(combined)
    print(f"  Combined: {len(combined):,} chars -> Cleaned: {len(cleaned):,} chars")
    
    # Truncate if too large
    if len(cleaned) > MAX_CHARS:
        cleaned = cleaned[:MAX_CHARS]
        print(f"  Truncated to {MAX_CHARS:,} chars")
    
    return cleaned


def train_personality(name, text_files):
    """Train a single personality model."""
    print(f"\n{'='*60}")
    print(f"  TRAINING: {name}")
    print(f"{'='*60}")
    
    # Load texts
    corpus = load_and_clean_texts(text_files)
    if len(corpus) < 100:
        print(f"  [ERROR] Corpus too small ({len(corpus)} chars). Skipping.")
        return
    
    # Create model
    model = PharmakonTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_layers=2,
        max_seq_len=64,
        dropout=0.0
    )
    
    # Calculate warmup steps
    warmup_steps = min(100, len(corpus) // (BATCH_SIZE * SEQ_LEN))
    
    start = time.time()
    
    # Train
    train_module.train(
        model=model,
        data=corpus,
        char_to_idx=char_to_idx,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        use_checkpoint=True
    )
    
    elapsed = time.time() - start
    print(f"  Training complete in {elapsed:.1f}s")
    
    # Save weights
    weights = extract_weights(model)
    save_path = WEIGHTS_DIR / f"{name}.npz"
    np.savez_compressed(save_path, **weights)
    print(f"  [SAVED] {save_path}")


if __name__ == "__main__":
    print("Pharmakon – Batch Personality Training")
    print(f"Data directory: {DATA_DIR}")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LR}")
    
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, files in PERSONALITY_TEXTS.items():
        train_personality(name, files)
    
    print("\n" + "="*60)
    print("  ALL TRAINING COMPLETE")
    print("="*60)
