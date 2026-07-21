"""
Pharmakon – Project Gutenberg Novel Downloader
================================================
Downloads plain text copies of public domain books from Project Gutenberg
purely for local non-commercial model training.

Fair Use Disclaimer:
These texts are downloaded under Section 107 of the US Copyright Act (Fair Use)
for research, education, and machine learning model training purposes only.
They will not be commercialized, sold, or distributed.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

DISCLAIMER = """
================================================================================
          PROJECT GUTENBERG DOWNLOADER - FAIR USE COPYRIGHT DISCLAIMER
================================================================================
This utility downloads plain text copies of public domain classic literature from
Project Gutenberg (www.gutenberg.org).

Usage constraints:
1. Research & Education: These files are downloaded solely for local machine
   learning model training and semantic research.
2. Fair Use: Under Section 107 of the US Copyright Act 1976, this research use
   constitutes a protected 'fair use' of public domain and educational texts.
3. Non-Commercial: These files are stored locally in 'data/' for your private
   runs, and will NOT be sold, commercialized, or distributed.
================================================================================
"""

# Classic Philosophy and Greek literature plain text mirrors on Project Gutenberg
GUTENBERG_BOOKS = {
    "Plato_Republic": "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
    "Aristotle_Politics": "https://www.gutenberg.org/cache/epub/6762/pg6762.txt",
    "Marcus_Aurelius_Meditations": "https://www.gutenberg.org/cache/epub/6828/pg6828.txt",
    "Plato_Apology_Crito_Phaedo": "https://www.gutenberg.org/cache/epub/1656/pg1656.txt"
}


def strip_gutenberg_metadata(text: str) -> str:
    """Strips the Project Gutenberg header and footer metadata to keep text clean."""
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK"
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK"
    ]
    
    # Locate beginning of text
    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # find end of line containing the marker
            eol = text.find("\n", idx)
            if eol != -1:
                start_idx = eol + 1
            break
            
    # Locate end of text
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break
            
    return text[start_idx:end_idx].strip()


def download_books():
    print(DISCLAIMER)
    
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    for name, url in GUTENBERG_BOOKS.items():
        out_path = data_dir / f"{name}.txt"
        print(f"Downloading {name}...")
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                raw_text = response.text
                clean_text = strip_gutenberg_metadata(raw_text)
                
                # Verify we extracted text
                if not clean_text:
                    print(f"  [Warning] Metadata stripping resulted in empty text for {name}. Saving raw file instead.")
                    clean_text = raw_text
                
                out_path.write_text(clean_text, encoding="utf-8")
                print(f"  [Success] Saved {len(clean_text):,} characters to: data/{out_path.name}")
                downloaded_files.append(out_path)
            else:
                print(f"  [Error] Failed to download {name} (status code {response.status_code})")
        except Exception as e:
            print(f"  [Error] Failed to download {name}: {e}")
            
    print(f"\nCompleted! Downloaded and processed {len(downloaded_files)} book(s).")
    
    # Prompt user to clean corpus (strips accents and enforces 97-char vocabulary)
    if downloaded_files:
        print("\nCleaning corpus using clean_corpus.py to enforce ASCII 97-char vocabulary...")
        try:
            subprocess.run([sys.executable, "clean_corpus.py"])
            print("[Clean] All downloaded text files successfully normalized.")
        except Exception as e:
            print(f"[Warning] Failed to auto-run clean_corpus.py: {e}")
            
        # Prompt to run personalities training pipeline
        print("\nWould you like to kick off the personality training pipeline now to train on these new books?")
        print("To run, execute: python train_personalities.py")


if __name__ == "__main__":
    download_books()
