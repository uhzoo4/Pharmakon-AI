"""
Pharmakon – Text Translation Utility
======================================
Translates novels and text corpora from foreign languages (German, French, Spanish, etc.)
into English. Ensures the text is fully ASCII-compatible for training characters.

Usage:
    python translator.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# -----------------------------------------------------------------------------
# Dependency Resolution
# -----------------------------------------------------------------------------
def install_deep_translator():
    try:
        import deep_translator
    except ImportError:
        print("[System] Installing open-source 'deep-translator' library for fallback translation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "deep-translator"])
            print("[System] Successfully installed deep-translator.")
        except Exception as e:
            print(f"[Error] Failed to install deep-translator: {e}")
            print("Please run: pip install deep-translator")
            sys.exit(1)

# Ensure dependencies are available
install_deep_translator()

from deep_translator import GoogleTranslator
import requests

# -----------------------------------------------------------------------------
# Core Translation Engine
# -----------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_API_KEY")

def translate_chunk_hf(text: str, token: str) -> str:
    """Translates a text chunk using Hugging Face Serverless Inference API."""
    # We use the Helsinki-NLP multilingual-to-English translation model
    API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-mul-en"
    headers = {"Authorization": f"Bearer {token}"}
    
    # Retry logic for model loading (since HF model may be sleeping)
    for attempt in range(5):
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=30)
        if response.status_code == 200:
            res_json = response.json()
            if isinstance(res_json, list) and len(res_json) > 0 and "translation_text" in res_json[0]:
                return res_json[0]["translation_text"]
            elif isinstance(res_json, dict) and "translation_text" in res_json:
                return res_json["translation_text"]
            else:
                raise ValueError(f"Unexpected response format: {res_json}")
        elif response.status_code == 503:
            # Model is loading, wait and retry
            print(f"  [HF API] Model is loading, waiting 15s (attempt {attempt+1}/5)...")
            time.sleep(15)
        else:
            raise RuntimeError(f"HF Inference API returned status {response.status_code}: {response.text}")
            
    raise RuntimeError("HF Inference API failed to load the model after several attempts.")


def translate_chunk_google(text: str) -> str:
    """Translates a text chunk using keyless Google Translate API via deep-translator."""
    return GoogleTranslator(source="auto", target="en").translate(text)


def translate_text(text: str, chunk_size: int = 3000) -> str:
    """Splits text into chunks, translates them, and joins them back."""
    # Split text into paragraphs or words to maintain semantic boundary
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        if current_length + len(para) + 1 > chunk_size:
            chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += len(para) + 1
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    print(f"Divided text into {len(chunks)} chunks for translation.")
    
    translated_chunks = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            translated_chunks.append(chunk)
            continue
            
        print(f"  Translating chunk {idx+1}/{len(chunks)} ({len(chunk)} chars)...")
        translated_text = ""
        
        # Try Hugging Face first if token is available
        if HF_TOKEN:
            try:
                translated_text = translate_chunk_hf(chunk, HF_TOKEN)
                print("    [Success] Translated using Hugging Face Inference API")
            except Exception as hf_err:
                print(f"    [Warning] HF Inference API failed: {hf_err}. Falling back to Google Translate...")
                
        # Fallback to Google Translate if HF failed or wasn't configured
        if not translated_text:
            try:
                translated_text = translate_chunk_google(chunk)
                print("    [Success] Translated using Google Translate (deep-translator)")
                # Sleep briefly to avoid Google Translate rate limits
                time.sleep(0.5)
            except Exception as g_err:
                print(f"    [Error] Google Translate failed: {g_err}")
                print("    Skipping translation for this chunk.")
                translated_text = chunk # Fallback to original text chunk
                
        translated_chunks.append(translated_text)
        
    return "\n".join(translated_chunks)

# -----------------------------------------------------------------------------
# Main Orchestrator
# -----------------------------------------------------------------------------
def main():
    root_dir = Path(__file__).resolve().parent
    untranslated_dir = root_dir / "data" / "untranslated"
    data_dir = root_dir / "data"
    
    # Create directories if they do not exist
    untranslated_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan for .txt, .docx, and .pdf files
    supported_extensions = ["*.txt", "*.docx", "*.pdf"]
    input_files = []
    for ext in supported_extensions:
        input_files.extend(list(untranslated_dir.glob(ext)))
        
    if not input_files:
        print("\n=== PHARMAKON TRANSLATOR ===")
        print(f"No untranslated files found in: {untranslated_dir}")
        print("Please drop your foreign language (.txt, .docx, or .pdf) novels or books in that folder.")
        print("Example: data/untranslated/german_book.txt")
        return
        
    print(f"Found {len(input_files)} file(s) ready for translation.")
    if HF_TOKEN:
        print("Hugging Face API token detected. Ready to use serverless inference.")
    else:
        print("No Hugging Face token found. Using open-source Google Translate fallback.")
        
    for file_path in input_files:
        print("\n" + "-"*60)
        print(f"Processing: {file_path.name}")
        print("-"*60)
        
        raw_text = ""
        ext = file_path.suffix.lower()
        
        try:
            if ext == ".txt":
                raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
            elif ext == ".docx":
                import docx2txt  # type: ignore
                raw_text = docx2txt.process(str(file_path))
            elif ext == ".pdf":
                import pypdf  # type: ignore
                reader = pypdf.PdfReader(str(file_path))
                raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            else:
                print(f"[Warning] Unsupported file extension {ext}. Skipping.")
                continue
        except Exception as e:
            print(f"[Error] Failed to read/parse {file_path}: {e}")
            continue
            
        if len(raw_text.strip()) == 0:
            print("[Warning] File has no extractable text. Skipping.")
            continue
            
        print(f"Loaded {len(raw_text):,} characters of foreign text.")
        start_time = time.time()
        
        # Translate
        translated_text = translate_text(raw_text)
        
        # Save output
        out_name = f"translated_{file_path.stem}.txt"
        out_path = data_dir / out_name
        
        try:
            out_path.write_text(translated_text, encoding="utf-8")
            elapsed = time.time() - start_time
            print(f"[Done] Saved translated English text to: {out_path.name}")
            print(f"Elapsed Time: {elapsed:.2f} seconds")
        except Exception as e:
            print(f"[Error] Failed to write translated text: {e}")

if __name__ == "__main__":
    main()
