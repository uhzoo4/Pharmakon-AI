"""
Pharmakon – Multi-Personality Training Pipeline
================================================
Scans the data/ directory for any .txt files, trains the character-level model,
and updates the corresponding .npz weights inside backend/weights/ dynamically.
Supports incremental fine-tuning if weights already exist.
"""

import sys
from pathlib import Path
import numpy as np

# Adjust sys.path to resolve backend imports
sys.path.append(str(Path(__file__).parent / "backend"))

from transformer import PharmakonTransformer
import train

# Predefined Vocabulary configuration
VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
VOCAB_SIZE = len(VOCAB_CHARS)
char_to_idx = {char: idx for idx, char in enumerate(VOCAB_CHARS)}


def run_pipeline(epochs=20, batch_size=32, seq_len=64, lr=3e-4, use_checkpoint=True):
    data_dir = Path("data")
    weights_dir = Path("backend/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        print("No training corpus files found in data/ folder.")
        print("Please add files like 'kafkaesque.txt' or 'camus_stranger.txt' to data/ and run again.")
        return

    print(f"Found {len(txt_files)} training corpus files: {[f.name for f in txt_files]}")

    for file_path in txt_files:
        name = file_path.stem
        print("\n" + "=" * 60)
        print(f"Starting training pipeline for personality: '{name}'")
        print("=" * 60)

        # 1. Read and validate corpus
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        except Exception as exc:
            print(f"  [WARNING] Failed to read {file_path}: {exc}")
            continue

        if len(text_content) < seq_len + 2:
            print(f"  [WARNING] Text file '{file_path.name}' is too small (must be at least {seq_len + 2} chars). Skipping.")
            continue

        print(f"Loaded {len(text_content):,} characters of training corpus.")

        # 2. Instantiate v4.0 model architecture
        model = PharmakonTransformer(
            vocab_size=VOCAB_SIZE,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            num_layers=2,
            max_seq_len=seq_len,
            dropout=0.0  # Set dropout to 0.0 for stable loss calculations
        )

        # 3. Check for existing weights for incremental training / fine-tuning
        npz_path = weights_dir / f"{name}.npz"
        if npz_path.exists():
            print(f"Found existing weights file: {npz_path.name}")
            print("Loading weights to resume training (incremental fine-tuning)...")
            try:
                # Force casting parameters back to float64 to maintain high-precision gradient checks
                weights_dict = dict(np.load(npz_path))
                for key in weights_dict:
                    weights_dict[key] = weights_dict[key].astype(np.float64)
                model.load_weights(weights_dict)
                print("  [OK] Loaded weights successfully.")
            except Exception as exc:
                print(f"  [WARNING] Failed to load existing weights (will train from scratch): {exc}")
        else:
            print(f"No existing weights found for '{name}'. Training from scratch...")

        # 4. Trigger training loop
        # We set warmup steps dynamically based on batch volume
        try:
            train.train(
                model=model,
                data=text_content,
                char_to_idx=char_to_idx,
                epochs=epochs,
                batch_size=batch_size,
                seq_len=seq_len,
                lr=lr,
                weight_decay=0.01,
                warmup_steps=min(100, len(text_content) // (batch_size * seq_len)),
                use_checkpoint=use_checkpoint
            )
        except Exception as exc:
            print(f"  [WARNING] Training crashed: {exc}")
            continue

        # 5. Extract trained parameters and save back to directory
        print(f"Saving optimized parameters to checkpoint: {npz_path}")
        try:
            # Gather parameters from model
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

            np.savez_compressed(npz_path, **params_dict)
            print(f"  [OK] Saved '{name}' parameters successfully.")
        except Exception as exc:
            print(f"  [WARNING] Failed to save weights to file: {exc}")

    print("\nTraining run complete.")


if __name__ == "__main__":
    # Standard training hyper-parameters
    run_pipeline(epochs=10, batch_size=16, seq_len=64, lr=3e-4, use_checkpoint=True)
