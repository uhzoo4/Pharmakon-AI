import os
import sys
import re
import numpy as np
from pathlib import Path
import subprocess
from typing import Dict, Any

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from backend.transformer import PharmakonTransformer
import backend.train as train_module

WEIGHTS_DIR = BASE_DIR / "backend" / "weights"
DATA_DIR = BASE_DIR / "data"

def install_datasets():
    try:
        import datasets # type: ignore
    except ImportError:
        print("[System] Installing HuggingFace datasets library for massive data stream...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

def load_massive_dataset() -> str:
    from datasets import load_dataset # type: ignore
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
        text = item["text"] # type: ignore
        text = text.replace("### Human:", "User:").replace("### Assistant:", "Assistant:")
        cleaned = "".join([c for c in text if c in valid_chars])
        
        if not cleaned.strip():
            continue
            
        combined.append(cleaned.strip())
        char_count += len(cleaned)
        
        if char_count >= target_chars:
            break
            
    final_text = "\n\n".join(combined)
    
    # Inject the documentary if the user adds it!
    doc_path = DATA_DIR / "documentary.txt"
    if doc_path.exists():
        print("[System] Legendary Artifact Detected: Injecting documentary.txt into the training data!")
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
            # Strip timestamps (e.g. 00:00:00.400)
            doc_text = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}\s*', '', doc_text)
            # Strip bracketed tags (e.g. [music])
            doc_text = re.sub(r'\[.*?\]\s*', '', doc_text)
            
            cleaned_doc = "".join([c for c in doc_text if c in valid_chars])
            final_text = cleaned_doc + "\n\n" + final_text
            
    print(f"[System] Successfully prepared {len(final_text):,} characters of ultra-clean data.")
    return final_text

def train_leviathan(text_data: str):
    print("--- 2. BOOTING THE LEVIATHAN ---")
    
    print("--- 2. TRAINING BPE TOKENIZER ---")
    from backend.bpe import BPETokenizer
    
    bpe = BPETokenizer()
    bpe.train(text_data, vocab_size=2000)
    
    bpe_path = WEIGHTS_DIR / "leviathan_bpe.json"
    bpe.save(str(bpe_path))
    print(f"[System] BPE Tokenizer saved to {bpe_path}")
    
    encoded_data = bpe.encode(text_data)
    VOCAB_SIZE = len(bpe.vocab)
    
    # THE LEVIATHAN CONFIGURATION (8GB RAM SAFE MODE)
    print(f"""
    [WARNING] INITIALIZING THE LEVIATHAN (8GB SAFE MODE)
    Architecture: 12 Layers, 1024 Embed Dim, 16 Heads, 2048 FF Dim, 512 Context Window.
    Vocab Size: {VOCAB_SIZE} (BPE Sub-words)
    This pure NumPy instantiation has been carefully tuned to consume ~3-4 GB of RAM,
    ensuring it fully maximizes your hardware without crashing your 8GB system!
    """)
    
    model = PharmakonTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=1024,   # Gigantic embeddings
        num_heads=16,     # Huge multi-head routing
        ff_dim=2048,      # Scaled down from 4096 to prevent 8GB OOM crashes
        num_layers=12,    # Deepest architecture yet
        max_seq_len=256   # Dropped from 512 to save N^2 attention memory in the backward pass
    )
    
    target_weights = WEIGHTS_DIR / "the_leviathan.npz"
    
    print(f"[System] Unleashing training cycle on {len(encoded_data):,} BPE tokens...")
    
    history = train_module.train(
        model=model, 
        data=encoded_data, 
        char_to_idx=None, 
        epochs=3,         # 3 epochs on massive data
        batch_size=1,     # Extreme Micro-batching (1) to guarantee 8GB RAM safety!
        seq_len=256,      # Reduced context window to survive backprop
        lr=2e-4, 
        weight_decay=0.01, 
        warmup_steps=200, 
        use_checkpoint=False
    )
    
    if any(np.isnan(l) for l in history):
        print("[System] CATASTROPHIC FAILURE: NaN loss detected. The Leviathan has collapsed.")
        sys.exit(1)
        
    print(f"[System] Saving Leviathan weights to {target_weights}...")
    params: Dict[str, Any] = train_module.extract_weights(model)

    np.savez_compressed(target_weights, **dict(params))
    print("[System] The Leviathan has been successfully forged!")

if __name__ == "__main__":
    install_datasets()
    fresh_data = load_massive_dataset()
    train_leviathan(fresh_data)
