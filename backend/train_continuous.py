import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.transformer import PharmakonTransformer
import backend.train as train_module

DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "backend" / "weights"

def run_harvesters():
    print("--- 1. RUNNING AUTOMATED HARVESTERS ---")
    hackernews_script = BASE_DIR / "tools" / "harvest_hackernews.py"
    discord_script = BASE_DIR / "tools" / "harvest_discord.py"
    
    # Run Hacker News (No auth required!)
    print("[System] Triggering Hacker News Harvester (Keyless)...")
    subprocess.run([sys.executable, str(hackernews_script)])
        
    # Run Discord
    if os.environ.get("DISCORD_TOKEN"):
        print("[System] Triggering Discord Harvester...")
        subprocess.run([sys.executable, str(discord_script)])
    else:
        print("[System] Skipping Discord Harvester (Missing credentials in ENV)")

def load_and_merge_data() -> str:
    print("--- 2. MERGING FRESH DATA ---")
    combined = ""
    
    reddit_file = DATA_DIR / "reddit_corpus.txt"
    if reddit_file.exists():
        combined += reddit_file.read_text(encoding="utf-8") + "\n\n"
        
    discord_file = DATA_DIR / "discord_corpus.txt"
    if discord_file.exists():
        combined += discord_file.read_text(encoding="utf-8") + "\n\n"
        
    if not combined.strip():
        print("[System] No new data found. Make sure you set your API keys!")
        sys.exit(0)
        
    return combined

def continuous_fine_tune(text_data: str):
    print("--- 3. CONTINUOUS FINE-TUNING ---")
    
    # Hyper-scaled architecture for the_pinnacle.npz
    VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
    char_to_idx = {c: i for i, c in enumerate(VOCAB_CHARS)}
    VOCAB_SIZE = len(VOCAB_CHARS)
    
    model = PharmakonTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=128, 
        num_heads=8, 
        ff_dim=256, 
        num_layers=4, 
        max_seq_len=128
    )
    
    target_weights = WEIGHTS_DIR / "the_pinnacle.npz"
    if not target_weights.exists():
        print(f"[ERROR] Target model {target_weights} not found to fine-tune!")
        sys.exit(1)
        
    print(f"[System] Loading {target_weights.name} for continuous adaptation...")
    # Load weights
    weights = np.load(target_weights)
    model.token_embedding = weights['token_embedding']
    model.W_out = weights['W_out']
    model.ln_final.gamma = weights['ln_final_gamma']
    model.ln_final.beta = weights['ln_final_beta']
    for i, block in enumerate(model.blocks):
        p = f'block_{i}_'
        block.Wq = weights[p+'Wq']
        block.Wk = weights[p+'Wk']
        block.Wv = weights[p+'Wv']
        block.Wo = weights[p+'Wo']
        block.ln1.gamma = weights[p+'ln1_gamma']
        block.ln1.beta = weights[p+'ln1_beta']
        block.ln2.gamma = weights[p+'ln2_gamma']
        block.ln2.beta = weights[p+'ln2_beta']
        block.W1 = weights[p+'W1']
        block.b1 = weights[p+'b1']
        block.W2 = weights[p+'W2']
        block.b2 = weights[p+'b2']

    # Train on fresh data (low learning rate so it doesn't forget its core identity)
    print(f"[System] Fine-tuning on {len(text_data)} characters of fresh data...")
    history = train_module.train(
        model=model, 
        data=text_data, 
        char_to_idx=char_to_idx, 
        epochs=20, # Hyper-stress testing
        batch_size=32, # Increased batch size
        seq_len=128, # Double context window for the_pinnacle
        lr=5e-6, # Extremely low learning rate to prevent shock without Adam state
        weight_decay=0.01, 
        warmup_steps=5, 
        use_checkpoint=False
    )
    
    # NaN Protection
    if any(np.isnan(l) for l in history):
        print("[System] CATASTROPHIC FAILURE: NaN loss detected. Aborting save to protect weights.")
        sys.exit(1)
        
    # Save back
    print(f"[System] Saving updated weights to {target_weights}...")
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
    print("[System] Continuous Training Cycle Complete! Model adapted to fresh internet data.")

if __name__ == "__main__":
    run_harvesters()
    fresh_data = load_and_merge_data()
    continuous_fine_tune(fresh_data)
