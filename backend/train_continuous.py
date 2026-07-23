import os
import sys
import time
import subprocess
import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, Any

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

def continuous_fine_tune(text_data: str, resume_state: dict | None = None):
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
        max_seq_len=64
    )
    
    WEIGHTS_DIR = Path("backend/weights")
    target_weights = WEIGHTS_DIR / "the_pinnacle.npz"

    if not target_weights.exists():
        print(f"[ERROR] Target weights {target_weights} not found!")
        sys.exit(1)
        
    print(f"[System] Loading {target_weights.name} for continuous adaptation...")
    # Load weights
    weights = np.load(target_weights)
    model.load_weights(dict(weights))

    # Train on fresh data (low learning rate so it doesn't forget its core identity)
    print(f"[System] Fine-tuning on {len(text_data)} characters of fresh data...")
    history = train_module.train(
        model=model, 
        data=text_data, 
        char_to_idx=char_to_idx, 
        epochs=20, # Hyper-stress testing
        batch_size=16, # Halved batch size to prevent OOM
        seq_len=64, # Halved context window to prevent OOM
        lr=5e-6, # Extremely low learning rate to prevent shock without Adam state
        weight_decay=0.01, 
        warmup_steps=5, 
        use_checkpoint=True,
        resume_state=resume_state
    )
    
    # NaN Protection
    if any(np.isnan(l) for l in history):
        print("[System] CATASTROPHIC FAILURE: NaN loss detected. Aborting save to protect weights.")
        sys.exit(1)
        
    # Save back
    print(f"[System] Saving updated weights to {target_weights}...")
    params: Dict[str, Any] = train_module.extract_weights(model)

    np.savez_compressed(target_weights, **dict(params))
    print("[System] Continuous Training Cycle Complete! Model adapted to fresh internet data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true", help="Run as worker")
    parser.add_argument("--resume_from", type=str, default="", help="Path to JSON state to resume from")
    args = parser.parse_args()

    resume_state = None
    if args.resume_from:
        print(f"[Worker] Loading resume state from {args.resume_from}...")
        with open(args.resume_from, "r") as f:
            resume_state = json.load(f)

    # Only run harvesters if we are NOT resuming from a crash midway through
    if not resume_state:
        run_harvesters()
    else:
        print("[System] Resuming from crash: Skipping harvester to prevent dataset ballooning.")

    fresh_data = load_and_merge_data()
    continuous_fine_tune(fresh_data, resume_state=resume_state)
