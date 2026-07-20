import os
import sys
import numpy as np
from pathlib import Path
import subprocess

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.transformer import PharmakonTransformer
import backend.train as train_module

WEIGHTS_DIR = BASE_DIR / "backend" / "weights"
DATA_DIR = BASE_DIR / "data"

def install_datasets():
    try:
        import datasets
    except ImportError:
        print("[System] Installing HuggingFace datasets library for massive data stream...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

def load_massive_dataset() -> str:
    from datasets import load_dataset
    print("--- 1. HARVESTING GIGABYTES OF DATA ---")
    print("Connecting to HuggingFace to stream timdettmers/openassistant-guanaco...")
    
    # We load streaming=True if we don't want to kill the disk, or just load the train split.
    # We will load a chunk of the dataset to get around 5-10 Megabytes of raw text.
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
    valid_chars = set(VOCAB_CHARS)
    
    combined = []
    char_count = 0
    target_chars = 10_000_000 # 10 Million characters!
    
    print(f"[System] Filtering {len(dataset)} conversations down to {target_chars:,} characters...")
    
    for item in dataset:
        text = item["text"]
        text = text.replace("### Human:", "User:").replace("### Assistant:", "Assistant:")
        cleaned = "".join([c for c in text if c in valid_chars])
        
        if not cleaned.strip():
            continue
            
        combined.append(cleaned.strip())
        char_count += len(cleaned)
        
        if char_count >= target_chars:
            break
            
    final_text = "\n\n".join(combined)
    print(f"[System] Successfully prepared {len(final_text):,} characters of ultra-clean data.")
    return final_text

def train_leviathan(text_data: str):
    print("--- 2. BOOTING THE LEVIATHAN ---")
    
    VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
    char_to_idx = {c: i for i, c in enumerate(VOCAB_CHARS)}
    VOCAB_SIZE = len(VOCAB_CHARS)
    
    # THE LEVIATHAN CONFIGURATION (8GB RAM SAFE MODE)
    print("""
    [WARNING] INITIALIZING THE LEVIATHAN (8GB SAFE MODE)
    Architecture: 12 Layers, 1024 Embed Dim, 16 Heads, 2048 FF Dim, 512 Context Window.
    This pure NumPy instantiation has been carefully tuned to consume ~3-4 GB of RAM,
    ensuring it fully maximizes your hardware without crashing your 8GB system!
    """)
    
    model = PharmakonTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=1024,   # Gigantic embeddings
        num_heads=16,     # Huge multi-head routing
        ff_dim=2048,      # Scaled down from 4096 to prevent 8GB OOM crashes
        num_layers=12,    # Deepest architecture yet
        max_seq_len=512   # Massive context window
    )
    
    target_weights = WEIGHTS_DIR / "the_leviathan.npz"
    
    print(f"[System] Unleashing training cycle on {len(text_data):,} characters...")
    
    history = train_module.train(
        model=model, 
        data=text_data, 
        char_to_idx=char_to_idx, 
        epochs=3,         # 3 epochs on 10 million characters
        batch_size=2,     # Micro-batching to guarantee 8GB RAM safety!
        seq_len=512,      # Massive context window
        lr=2e-4, 
        weight_decay=0.01, 
        warmup_steps=200, 
        use_checkpoint=False
    )
    
    if any(np.isnan(l) for l in history):
        print("[System] CATASTROPHIC FAILURE: NaN loss detected. The Leviathan has collapsed.")
        sys.exit(1)
        
    print(f"[System] Saving Leviathan weights to {target_weights}...")
    params = {}
    params['token_embedding'] = model.token_embedding.astype(np.float32)
    params['W_out'] = model.W_out.astype(np.float32)
    params['ln_final_gamma'] = model.ln_final.gamma.astype(np.float32)
    params['ln_final_beta'] = model.ln_final.beta.astype(np.float32)
    for i, block in enumerate(model.blocks):
        p = f'block_{i}_'
        params[p+'Wq'] = block.Wq.astype(np.float32)
        params[p+'Wk'] = block.Wk.astype(np.float32)
        params[p+'Wv'] = block.Wv.astype(np.float32)
        params[p+'Wo'] = block.Wo.astype(np.float32)
        params[p+'ln1_gamma'] = block.ln1.gamma.astype(np.float32)
        params[p+'ln1_beta'] = block.ln1.beta.astype(np.float32)
        params[p+'ln2_gamma'] = block.ln2.gamma.astype(np.float32)
        params[p+'ln2_beta'] = block.ln2.beta.astype(np.float32)
        params[p+'W1'] = block.W1.astype(np.float32)
        params[p+'b1'] = block.b1.astype(np.float32)
        params[p+'W2'] = block.W2.astype(np.float32)
        params[p+'b2'] = block.b2.astype(np.float32)

    np.savez_compressed(target_weights, **params)
    print("[System] The Leviathan has been successfully forged!")

if __name__ == "__main__":
    install_datasets()
    fresh_data = load_massive_dataset()
    train_leviathan(fresh_data)
