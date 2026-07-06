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
import math
import unicodedata
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, FrozenSet

# -----------------------------------------------------------------------------
# Constants & Type Definitions
# -----------------------------------------------------------------------------

# Target Vocabulary V = 97: Newline, Tab, Printable ASCII (32-126)
VOCAB_CHARS: Tuple[str, ...] = ("\n", "\t") + tuple(chr(i) for i in range(32, 127))
VOCAB_SET: FrozenSet[str] = frozenset(VOCAB_CHARS)
VOCAB_SIZE: int = len(VOCAB_CHARS)  # 97

# Smart Punctuation Mapping (Pre-NFKD normalization)
# Maps specific Unicode codepoints to their ASCII equivalents.
SMART_MAP: Dict[str, str] = {
    "\u2019": "'",  # Right Single Quotation Mark
    "\u2018": "'",  # Left Single Quotation Mark
    "\u201C": '"',  # Left Double Quotation Mark
    "\u201D": '"',  # Right Double Quotation Mark
    "\u2014": "-",  # Em Dash
    "\u2013": "-",  # En Dash
    "\u00A0": " ",  # Non-breaking Space
}

@dataclass(frozen=True, slots=True)
class CorpusMetrics:
    """Immutable container for corpus statistical metrics."""
    length: int
    coverage: float  # \eta
    entropy: float   # H (bits/char)

@dataclass(frozen=True, slots=True)
class CleaningReport:
    """Immutable report for a single file processing run."""
    file_name: str
    raw_metrics: CorpusMetrics
    clean_metrics: CorpusMetrics
    dropped_counts: Dict[str, int]
    chars_removed: int

# -----------------------------------------------------------------------------
# Core Mathematical Operations
# -----------------------------------------------------------------------------

def compute_metrics(text: str) -> CorpusMetrics:
    r"""
    Calculates corpus statistics in a single pass O(N).
    
    Returns:
        CorpusMetrics containing length, vocabulary coverage (\eta), 
        and Shannon Entropy (H).
    """
    n = len(text)
    if n == 0:
        return CorpusMetrics(length=0, coverage=0.0, entropy=0.0)

    freq: Dict[str, int] = {}
    in_vocab_count = 0

    # Single pass: Frequency count + Vocab coverage
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
        if ch in VOCAB_SET:
            in_vocab_count += 1

    # Vocabulary Coverage Fraction (\eta)
    eta = in_vocab_count / n

    # Shannon Entropy H = - \sum P(x) log2 P(x)
    inv_n = 1.0 / n
    entropy = 0.0
    for count in freq.values():
        p = count * inv_n
        # math.log2 is faster and more accurate than log(p, 2)
        entropy -= p * math.log2(p)

    return CorpusMetrics(length=n, coverage=eta, entropy=entropy)


def clean_text(text: str) -> Tuple[str, Dict[str, int]]:
    r"""
    Applies the normalization pipeline:
    1. Smart Punctuation Substitution (Norm)
    2. NFKD Decomposition + Combining Mark Stripping
    3. Vocabulary Filtering (Projection \pi)
    
    Returns:
        Tuple of (cleaned_text, dropped_character_counts).
    """
    # 1. Smart Replacement (Norm)
    # str.translate is highly optimized in CPython for 1:1 char mapping
    # We handle multi-char sequences (\r\n) manually before translate.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(str.maketrans(SMART_MAP))

    # 2. NFKD Decomposition & Filtering
    # NFKD separates base chars (e.g. 'e') from combining marks (e.g. '\u0301').
    # Category 'Mn' (Mark, Non-Spacing) identifies diacritics to strip.
    decomposed = unicodedata.normalize("NFKD", text)
    
    # Pre-allocate list for performance (approx size)
    cleaned_chars = []
    cleaned_chars_append = cleaned_chars.append
    dropped: Dict[str, int] = {}
    dropped_get = dropped.get

    for ch in decomposed:
        # Strip combining diacritical marks (accents, umlauts, etc.)
        if unicodedata.category(ch) == "Mn":
            continue
        
        # Projection \pi: Keep if in Vocab, else Drop
        if ch in VOCAB_SET:
            cleaned_chars_append(ch)
        else:
            dropped[ch] = dropped_get(ch, 0) + 1

    return "".join(cleaned_chars), dropped


# -----------------------------------------------------------------------------
# Pipeline Orchestration
# -----------------------------------------------------------------------------

def process_file(file_path: Path) -> CleaningReport:
    """Processes a single corpus file: Read -> Analyze -> Clean -> Validate -> Write."""
    # --- Read ---
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"Failed to decode {file_path} as UTF-8: {e}") from e

    # --- Analyze Raw ---
    raw_metrics = compute_metrics(raw_text)

    # --- Clean ---
    cleaned_text, dropped_counts = clean_text(raw_text)

    # --- Analyze Clean ---
    clean_metrics = compute_metrics(cleaned_text)

    # --- Mathematical Guarantee Verification ---
    # The projection \pi guarantees \eta = 1.0 by construction (filtering).
    # We assert this as a runtime sanity check.
    if clean_metrics.coverage != 1.0:
        # This should be mathematically impossible if VOCAB_SET is correct
        raise AssertionError(
            f"Invariant Violation: Cleaned coverage is {clean_metrics.coverage:.6f}, "
            f"expected 1.0. File: {file_path.name}"
        )

    # --- Atomic Write with Backup ---
    # Strategy: Write to temp file -> Backup original -> Atomic Replace
    # This prevents data loss if the process is killed mid-write.
    
    # 1. Write cleaned content to a secure temporary file in the same directory
    #    (ensures atomic replace works across filesystems).
    with tempfile.NamedTemporaryFile(
        mode="w", 
        encoding="utf-8", 
        dir=file_path.parent, 
        delete=False, 
        suffix=".tmp"
    ) as tmp_f:
        tmp_f.write(cleaned_text)
        tmp_path = Path(tmp_f.name)

    backup_path = file_path.with_suffix(file_path.suffix + ".orig")
    try:
        # 2. Backup original: rename original -> .orig
        if backup_path.exists():
            backup_path.unlink() # Remove stale backup if exists
        file_path.rename(backup_path)

        # 3. Atomic commit: rename temp -> original name
        tmp_path.replace(file_path)
    except OSError:
        # Rollback: If replace fails, restore original name from backup
        if backup_path.exists():
            backup_path.rename(file_path)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return CleaningReport(
        file_name=file_path.name,
        raw_metrics=raw_metrics,
        clean_metrics=clean_metrics,
        dropped_counts=dropped_counts,
        chars_removed=raw_metrics.length - clean_metrics.length
    )


def format_dropped_report(dropped: Dict[str, int], limit: int = 10) -> str:
    """Formats the top-N dropped characters for logging."""
    if not dropped:
        return "  (No characters dropped)"
    
    lines = []
    sorted_items = sorted(dropped.items(), key=lambda x: x[1], reverse=True)
    for ch, count in sorted_items[:limit]:
        # repr() safely escapes control chars and quotes for logging
        lines.append(f"    * {repr(ch):<8} : {count:>10,} occurrences")
    
    if len(sorted_items) > limit:
        lines.append(f"    * ... and {len(sorted_items) - limit} other unique characters.")
    return "\n".join(lines)


def print_report(report: CleaningReport) -> None:
    """Pretty prints the cleaning report to stdout."""
    r = report
    print(f"\n{'='*20} Processing: {r.file_name} {'='*20}")
    print(f"  Raw Length      : {r.raw_metrics.length:>12,} chars")
    print(f"  Raw Coverage    : {r.raw_metrics.coverage*100:>12.4f}% (eta)")
    print(f"  Raw Entropy     : {r.raw_metrics.entropy:>12.4f} bits/char (H)")
    print(f"  Clean Length    : {r.clean_metrics.length:>12,} chars")
    print(f"  Clean Coverage  : {r.clean_metrics.coverage*100:>12.4f}% (eta) [GUARANTEED 100%]")
    print(f"  Clean Entropy   : {r.clean_metrics.entropy:>12.4f} bits/char (H)")
    print(f"  Chars Removed   : {r.chars_removed:>12,} chars")
    print(f"  Dropped Chars   :")
    print(format_dropped_report(r.dropped_counts))
    print(f"  Backup Saved    : {r.file_name}.orig")
    print(f"  Status          : [OK] Validated & Written")


def main(data_dir: Path = Path("data")) -> int:
    """Main entry point. Returns exit code."""
    print("=" * 70)
    print("PHARMAKON - MATHEMATICAL CORPUS CLEANING & VALIDATION ENGINE")
    print(f"Target Vocabulary Size: {VOCAB_SIZE} tokens")
    print("=" * 70)

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"\n[WARNING] No *.txt files found in '{data_dir.resolve()}'.")
        print("Place raw corpus files in this directory and re-run.")
        return 0

    error_count = 0
    for file_path in txt_files:
        # Skip backup files generated by previous runs
        if file_path.suffix == ".orig":
            continue
        
        try:
            report = process_file(file_path)
            print_report(report)
        except Exception as e:
            print(f"\n[FATAL ERROR] Processing {file_path.name}: {e}", file=sys.stderr)
            error_count += 1

    print("\n" + "=" * 70)
    if error_count == 0:
        print("ALL CORPUS FILES CLEANED & VALIDATED SUCCESSFULLY!")
    else:
        print(f"COMPLETED WITH {error_count} ERROR(S). CHECK LOGS ABOVE.")
    print("=" * 70)
    
    return 1 if error_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
