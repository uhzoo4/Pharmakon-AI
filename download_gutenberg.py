"""
Pharmakon – Project Gutenberg Novel Catalog Search & Downloader
================================================================
Queries the Gutendex API to dynamically search for and download public domain
classic literature in philosophy, mythology, and gothic/dark romance.

Fair Use Disclaimer:
These texts are downloaded under Section 107 of the US Copyright Act (Fair Use)
for research, education, and machine learning model training purposes only.
They will not be commercialized, sold, or distributed.
"""

import os
import sys
import subprocess
import urllib.parse
import requests
from pathlib import Path

DISCLAIMER = """
================================================================================
          PROJECT GUTENBERG DOWNLOADER - FAIR USE COPYRIGHT DISCLAIMER
================================================================================
This utility downloads plain text copies of public domain classic literature from
Project Gutenberg (www.gutenberg.org) using the Gutendex Catalog API.

Usage constraints:
1. Research & Education: These files are downloaded solely for local machine
   learning model training and semantic research.
2. Fair Use: Under Section 107 of the US Copyright Act 1976, this research use
   constitutes a protected 'fair use' of public domain and educational texts.
3. Non-Commercial: These files are stored locally in 'data/' for your private
   runs, and will NOT be sold, commercialized, or distributed.
================================================================================
"""

# Predefined categories mapping to search keywords
CATEGORIES = {
    "1": ("Greek/Roman Mythology & Legends", "greek mythology OR roman mythology OR legends of greece"),
    "2": ("Philosophy (Existential, Ancient & Modern)", "philosophy OR existentialism OR metaphysics OR ethics"),
    "3": ("Gothic Novels & Horror", "gothic novel OR gothic fiction OR ghost stories"),
    "4": ("Dark Romance & Literary Romance", "gothic romance OR classic romance")
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


def sanitize_filename(name: str) -> str:
    """Creates a filesystem-safe filename from a string."""
    keep_chars = ("-", "_", " ")
    cleaned = "".join(c for c in name if c.isalnum() or c in keep_chars).rstrip()
    return cleaned.replace(" ", "_")


def search_gutendex(query: str) -> list[dict]:
    """Queries Gutendex API for the search term and parses results."""
    encoded_query = urllib.parse.quote(query)
    api_url = f"https://gutendex.com/books/?search={encoded_query}"
    
    print(f"Querying Project Gutenberg catalog for: '{query}'...")
    try:
        response = requests.get(api_url, timeout=20)
        if response.status_code == 200:
            results = response.json().get("results", [])
            return results
        else:
            print(f"[Error] Catalog search API returned status {response.status_code}")
    except Exception as e:
        print(f"[Error] Failed to connect to catalog API: {e}")
    return []


def process_and_download(books: list[dict], max_downloads: int = 3):
    """Filters results, downloads missing plain text books, and normalizes them."""
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    
    for book in books:
        if downloaded_count >= max_downloads:
            break
            
        title = book.get("title", "Unknown Title")
        authors_list = book.get("authors", [])
        author = authors_list[0].get("name", "Unknown Author") if authors_list else "Unknown Author"
        
        # Check for plain text formats
        formats = book.get("formats", {})
        download_url = (
            formats.get("text/plain; charset=utf-8") or 
            formats.get("text/plain") or 
            formats.get("text/plain; charset=us-ascii")
        )
        
        if not download_url:
            # Skip if no plain text format is available
            continue
            
        safe_title = sanitize_filename(title)
        out_path = data_dir / f"{safe_title}.txt"
        
        # Check if book is already downloaded
        if out_path.exists():
            print(f"Skipping '{title}' (already exists locally at data/{out_path.name})")
            continue
            
        print(f"\nDownloading: '{title}' by {author}...")
        print(f"  URL: {download_url}")
        
        try:
            response = requests.get(download_url, timeout=30)
            if response.status_code == 200:
                raw_text = response.text
                clean_text = strip_gutenberg_metadata(raw_text)
                
                # Guard against empty stripped content
                if not clean_text:
                    clean_text = raw_text
                    
                out_path.write_text(clean_text, encoding="utf-8")
                print(f"  [Success] Saved {len(clean_text):,} characters to: data/{out_path.name}")
                downloaded_count += 1
            else:
                print(f"  [Error] Failed to download text (HTTP status {response.status_code})")
        except Exception as e:
            print(f"  [Error] Failed to download: {e}")
            
    print(f"\nCompleted! Discovered and downloaded {downloaded_count} new book(s).")
    
    # Run corpus cleaning if books were downloaded
    if downloaded_count > 0:
        print("\nCleaning corpus using clean_corpus.py to enforce ASCII 97-char vocabulary...")
        try:
            subprocess.run([sys.executable, "clean_corpus.py"])
            print("[Clean] All downloaded text files successfully normalized.")
            # Remove generated backup files immediately to keep directories pristine
            subprocess.run(["powershell", "-Command", "Remove-Item data\\*.orig -ErrorAction SilentlyContinue"])
            print("[Clean] Stale backup files (.orig) successfully removed.")
        except Exception as e:
            print(f"[Warning] Failed to normalize downloads: {e}")
            
        print("\nWould you like to kick off the training pipeline to train on these new books?")
        print("To run, execute: python train_personalities.py")


def main():
    print(DISCLAIMER)
    
    print("Choose a category to search and download books:")
    for num, (name, _) in CATEGORIES.items():
        print(f"  {num}. {name}")
    print("  5. Custom Search Query")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    search_query = ""
    if choice in CATEGORIES:
        search_query = CATEGORIES[choice][1]
    elif choice == "5":
        search_query = input("Enter your custom search query (e.g. Schopenhauer, dark romance, greek myths): ").strip()
    else:
        print("[Error] Invalid option. Exiting.")
        return
        
    if not search_query:
        print("[Error] Empty search query. Exiting.")
        return
        
    # Search catalog
    results = search_gutendex(search_query)
    
    if not results:
        print("No matching public domain books found in the catalog. Please try a different query.")
        return
        
    print(f"\nDiscovered {len(results)} matches in the catalog.")
    
    # Show the top 5 discovered books
    print("\nTop discovered titles:")
    valid_books = []
    for book in results:
        title = book.get("title", "Unknown Title")
        formats = book.get("formats", {})
        has_txt = any(k.startswith("text/plain") for k in formats.keys())
        if has_txt:
            valid_books.append(book)
            if len(valid_books) <= 5:
                authors = [a.get("name", "Unknown") for a in book.get("authors", [])]
                author_str = ", ".join(authors) if authors else "Unknown"
                print(f"  * '{title}' by {author_str}")
                
    if not valid_books:
        print("None of the discovered books have plain text files available. Skipping.")
        return
        
    confirm = input(f"\nProceed to automatically download up to 3 new books from this list? (y/n): ").strip().lower()
    if confirm == "y" or confirm == "yes":
        process_and_download(valid_books, max_downloads=3)
    else:
        print("Download aborted.")


if __name__ == "__main__":
    main()
