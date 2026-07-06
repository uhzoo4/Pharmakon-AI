r"""
Pharmakon – Corpus Normalization & Validation Engine
=====================================================
Applies mathematical text normalization, unicode decomposition (NFKD),
and character filtering to guarantee 100% vocabulary coverage.

Mathematical Formulation:
1. Unicode Decomposition (NFKD):
   For any character c, NFKD(c) = (c_base, m_1, m_2, ...) where c_base is the 
   base ASCII character and m_i are combining diacritical marks. We strip m_i.
   
2. Normalization Mapping:
   Norm(c) maps common non-ASCII punctuation to their ASCII equivalents:
     - Left/Right Single Curly Quotes (’ or ‘) -> '
     - Left/Right Double Curly Quotes (” or “) -> "
     - Em/En Dashes (— or –) -> -
     - Whitespace (e.g., non-breaking space \xa0) -> ' '
     - Carriage Returns (\r\n) -> \n
     
3. Vocabulary Coverage Fraction (\eta):
   \eta = (1 / N) * \sum_{i=1}^N I(c_i \in \mathcal{V})
   Where \mathcal{V} is the 97-token target vocabulary and I is the indicator function.
   This script guarantees \eta = 1.0 post-processing.
   
4. Shannon Entropy of the Corpus (H):
   H = - \sum_{v \in \mathcal{V}} P(v) * \log_2 P(v)
   Where P(v) is the frequency probability of vocabulary character v in the text.
"""

import sys
import unicodedata
from pathlib import Path
import math

# Define target 97-token vocabulary (matching train_personalities.py)
VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
VOCAB_SET = set(VOCAB_CHARS)


def clean_text(text: str) -> tuple[str, dict[str, int]]:
    """
    Applies unicode normalization, smart-character mapping, and strips combining accents.
    
    Returns:
        - The fully cleaned ASCII-compatible text string.
        - A dictionary containing counts of replaced/dropped out-of-vocabulary characters.
    """
    # 1. Map common smart punctuation and whitespace
    mappings = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "—": "-",
        "–": "-",
        "\r\n": "\n",
        "\xa0": " ",
    }
    for orig, rep in mappings.items():
        text = text.replace(orig, rep)

    # 2. Decompose unicode characters into base form + diacritics
    normalized_chars = []
    dropped_chars: dict[str, int] = {}

    # Decompose text to separate letters from accents (NFKD)
    decomposed = unicodedata.normalize("NFKD", text)
    
    for char in decomposed:
        # Check if it's a combining diacritical mark (category 'Mn') and skip it
        if unicodedata.category(char) == "Mn":
            continue
            
        if char in VOCAB_SET:
            normalized_chars.append(char)
        else:
            # Increment frequency of out-of-vocab characters that are dropped
            dropped_chars[char] = dropped_chars.get(char, 0) + 1

    cleaned_text = "".join(normalized_chars)
    return cleaned_text, dropped_chars


def compute_metrics(text: str) -> tuple[float, float]:
    r"""
    Calculates corpus statistics:
        - Vocabulary Coverage Fraction (\eta)
        - Shannon Entropy (H) in bits/character
    """
    if not text:
        return 0.0, 0.0
        
    n = len(text)
    
    # 1. Vocabulary Coverage
    in_vocab_count = sum(1 for c in text if c in VOCAB_SET)
    eta = in_vocab_count / n
    
    # 2. Character Probability & Shannon Entropy
    freqs: dict[str, int] = {}
    for char in text:
        freqs[char] = freqs.get(char, 0) + 1
        
    entropy = 0.0
    for char, count in freqs.items():
        p = count / n
        entropy -= p * math.log2(p)
        
    return eta, entropy


def process_corpus_directory(data_dir: Path = Path("data")) -> None:
    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"No corpus files found in '{data_dir}/'. Place your raw datasets there.")
        return

    print("=" * 70)
    print("PHARMAKON – MATHEMATICAL CORPUS CLEANING & VALIDATION RUN")
    print("=" * 70)

    for file_path in txt_files:
        print(f"\nProcessing File: {file_path.name}")
        
        # Read raw content
        raw_text = file_path.read_text(encoding="utf-8")
        raw_len = len(raw_text)
        
        if raw_len == 0:
            print("  [WARNING] File is empty. Skipping.")
            continue
            
        # Analyze raw state
        raw_coverage, raw_entropy = compute_metrics(raw_text)
        print(f"  - Raw length: {raw_len:,} characters")
        print(f"  - Raw Vocab Coverage (\\eta): {raw_coverage * 100:.4f}%")
        print(f"  - Raw Shannon Entropy (H): {raw_entropy:.4f} bits/char")
        
        # Clean text
        cleaned_text, dropped = clean_text(raw_text)
        cleaned_len = len(cleaned_text)
        
        # Analyze clean state
        clean_coverage, clean_entropy = compute_metrics(cleaned_text)
        
        # Print dropped characters report
        if dropped:
            print("  - Replaced / Dropped out-of-vocabulary characters:")
            for char, count in sorted(dropped.items(), key=lambda x: x[1], reverse=True)[:10]:
                char_repr = repr(char)
                print(f"    * Character {char_repr:<6} : {count:,} times")
            if len(dropped) > 10:
                print(f"    * ... and {len(dropped) - 10} other unique non-ASCII characters.")
                
        print(f"  - Cleaned length: {cleaned_len:,} characters (shrunk by {raw_len - cleaned_len:,} chars)")
        print(f"  - Cleaned Vocab Coverage (\\eta): {clean_coverage * 100:.4f}%")
        print(f"  - Cleaned Shannon Entropy (H): {clean_entropy:.4f} bits/char")
        
        # Guarantee mathematical coverage is perfect
        assert clean_coverage == 1.0, f"Error: Cleaned coverage is {clean_coverage}, expected 1.0"
        
        # Create a backup of the original raw file before writing
        backup_path = file_path.with_suffix(".txt.orig")
        if not backup_path.exists():
            file_path.rename(backup_path)
            print(f"  [OK] Saved backup of original file to: {backup_path.name}")
        else:
            # If a backup already exists, just overwrite the main file
            file_path.unlink()
            
        # Save cleaned file
        file_path.write_text(cleaned_text, encoding="utf-8")
        print(f"  [OK] Successfully wrote clean corpus: {file_path.name}")
        
    print("\n" + "=" * 70)
    print("ALL CORPUS FILES CLEANED & VALIDATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    process_corpus_directory()
