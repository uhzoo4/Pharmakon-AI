"""
HN + HF Data Preparation Pipeline
Parses, cleans, deduplicates, tokenizes, and converts raw text into `np.memmap` binary files (train.bin, val.bin).
"""

import os
import re
import numpy as np
from tokenizer import Tokenizer
from dataset import create_memmap_bin

def clean_text(text: str) -> str:
    """Cleans control sequences and normalizes whitespace."""
    # Replace non-printable ASCII except newlines and tabs
    text = re.sub(r'[^\x09\x0A\x20-\x7E]', '', text)
    # Remove empty line spam
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def deduplicate_lines(text: str) -> str:
    """Simple exact line-level deduplication preserving structural order."""
    seen = set()
    unique_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            unique_lines.append("")
            continue
        if stripped not in seen:
            seen.add(stripped)
            unique_lines.append(line)
    return "\n".join(unique_lines)

def process_data_sources(
    source_dir: str = "data",
    out_dir: str = "backend/data_bin",
    val_ratio: float = 0.1,
    vocab_size: int = 50257
):
    print(f"[Pipeline] Processing text sources from {source_dir}...")
    tokenizer = Tokenizer()
    
    all_text_chunks = []
    
    # Read files in data directory
    if os.path.exists(source_dir):
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".txt"):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                            raw = f.read()
                            cleaned = clean_text(raw)
                            deduped = deduplicate_lines(cleaned)
                            all_text_chunks.append(deduped)
                    except Exception as e:
                        print(f"[Pipeline] Warning: Skipping {filepath} due to read error: {e}")

    full_corpus = "\n\n".join(all_text_chunks)
    assert len(full_corpus) > 0, "Corpus is empty after loading text sources!"
    print(f"[Pipeline] Loaded total text corpus of {len(full_corpus):,} characters.")
    
    print("[Pipeline] Running tokenizer encode...")
    tokens = tokenizer.encode(full_corpus)
    total_tokens = len(tokens)
    print(f"[Pipeline] Total token count: {total_tokens:,}")
    
    # Split train / val
    val_size = int(total_tokens * val_ratio)
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]
    
    train_bin_path = os.path.join(out_dir, "train.bin")
    val_bin_path = os.path.join(out_dir, "val.bin")
    
    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.int32
    
    print(f"[Pipeline] Writing {len(train_tokens):,} tokens to {train_bin_path} (dtype={dtype})...")
    create_memmap_bin(train_tokens, train_bin_path, dtype=dtype)
    
    print(f"[Pipeline] Writing {len(val_tokens):,} tokens to {val_bin_path} (dtype={dtype})...")
    create_memmap_bin(val_tokens, val_bin_path, dtype=dtype)
    
    print("[Pipeline] Data preparation complete!")

if __name__ == "__main__":
    process_data_sources()
