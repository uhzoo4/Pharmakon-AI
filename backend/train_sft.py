import sys
import time
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Setup paths to import backend modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer import PharmakonTransformer
from clean_corpus import clean_text
import train as train_module

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run as worker")
    parser.add_argument("--resume_from", type=str, default="", help="Path to JSON state to resume from")
    args = parser.parse_args()

    resume_state = None
    if args.resume_from:
        print(f"[Worker] Loading resume state from {args.resume_from}...")
        with open(args.resume_from, "r") as f:
            resume_state = json.load(f)
    VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
    char_to_idx = {c: i for i, c in enumerate(VOCAB_CHARS)}
    VOCAB_SIZE = len(VOCAB_CHARS)

    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_DIR / "sft_dataset.txt"
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        return

    raw_text = dataset_path.read_text(encoding="utf-8", errors="ignore")
    
    # We duplicate the small dataset to ensure we have enough data to fill the batch/seq_len tensors
    # In a real SFT pipeline, you would use thousands of unique examples.
    raw_text = (raw_text + "\n") * 100 
    
    cleaned, _ = clean_text(raw_text)
    print(f"Loaded SFT dataset: {len(cleaned):,} characters (after duplication & cleaning)")

    # Initialize Model
    model = PharmakonTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=64, 
        num_heads=4, 
        ff_dim=128, 
        num_layers=2, 
        max_seq_len=64, 
        dropout=0.0
    )
    
    warmup = min(100, len(cleaned) // (16 * 64))
    print("Starting Supervised Fine-Tuning (SFT)...")
    start = time.time()

    # Train
    train_module.train(
        model=model, 
        data=cleaned, 
        char_to_idx=char_to_idx, 
        epochs=40, # slightly higher epochs for small dataset to enforce formatting
        batch_size=16, 
        seq_len=64, 
        lr=3e-4, 
        weight_decay=0.01, 
        warmup_steps=warmup, 
        use_checkpoint=True,
        resume_state=resume_state
    )

    elapsed = time.time() - start
    print(f"SFT Training complete in {elapsed:.1f}s")

    # Extract and save weights
    params: Dict[str, Any] = train_module.extract_weights(model)

    save_path = WEIGHTS_DIR / 'the_assistant.npz'
    np.savez_compressed(file=save_path, **dict(params))
    print(f"[SAVED] Assistant model saved to {save_path}")

if __name__ == "__main__":
    main()
