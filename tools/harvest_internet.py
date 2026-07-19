import os
import sys
import subprocess
from pathlib import Path

def install_deps():
    try:
        import datasets
    except ImportError:
        print("Installing HuggingFace datasets library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])

def harvest_and_clean():
    from datasets import load_dataset
    
    print("Harvesting massive internet conversational dataset...")
    # 'timdettmers/openassistant-guanaco' is a parquet-based modern dataset
    print("Downloading 'timdettmers/openassistant-guanaco' dataset...")
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "world_corpus.txt"
    
    print(f"Formatting {len(dataset)} conversations...")
    
    VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
    valid_chars = set(VOCAB_CHARS)
    
    with open(out_path, "w", encoding="utf-8") as f:
        # We only want to process enough to reach ~1 million characters to save time
        char_count = 0
        for item in dataset:
            text = item["text"]
            
            # Format: '### Human: ... ### Assistant: ...'
            # We want to replace '### Human:' with 'User:' and '### Assistant:' with 'Assistant:'
            text = text.replace("### Human:", "User:").replace("### Assistant:", "Assistant:")
            
            # Clean utterance to our 97-char vocab
            cleaned = "".join([c for c in text if c in valid_chars])
            if not cleaned.strip():
                continue
                
            f.write(cleaned.strip() + "\n\n")
            char_count += len(cleaned)
            
            if char_count > 1200000: # Slightly over 1MB
                break
            
    print(f"Successfully harvested and saved to {out_path}")
    print(f"File size: {out_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    install_deps()
    harvest_and_clean()

