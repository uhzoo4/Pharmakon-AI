import sys
import time
import requests
from pathlib import Path

def clean_text(text: str) -> str:
    """Strip everything except our 97-character vocab and remove HTML."""
    if not text:
        return ""
    # Basic HTML removal since HN API returns raw HTML for comments
    text = text.replace("<p>", "\n\n").replace("&#x27;", "'").replace("&quot;", '"').replace("&gt;", ">").replace("&lt;", "<").replace("&#x2F;", "/")
    
    valid_chars = set(["\n", "\t"] + [chr(i) for i in range(32, 127)])
    cleaned = "".join([c for c in text if c in valid_chars])
    
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()

def harvest_hackernews():
    print("Connecting to Hacker News API (No Auth Required)...")
    
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "reddit_corpus.txt" # We append to the same file so train_continuous picks it up
    
    # 1. Get Top Stories
    try:
        res = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json")
        res.raise_for_status()
        top_ids = res.json()[:30] # Get top 30 stories
    except Exception as e:
        print(f"[ERROR] Failed to fetch top stories: {e}")
        return

    print(f"Harvesting {len(top_ids)} live threads from Hacker News...")

    with open(out_path, "a", encoding="utf-8") as f:
        for story_id in top_ids:
            try:
                story_res = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
                story_data = story_res.json()
                
                title = clean_text(story_data.get("title", ""))
                kids = story_data.get("kids", [])
                
                if not title or not kids:
                    continue
                    
                # 2. Get the top comment
                top_comment_id = kids[0]
                comment_res = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{top_comment_id}.json")
                comment_data = comment_res.json()
                
                body = clean_text(comment_data.get("text", ""))
                
                if not body or body == "[deleted]":
                    continue
                    
                # We format the post title as the User prompt
                # And the top comment as the Assistant reply
                f.write(f"User: {title}\nAssistant: {body}\n\n")
                print(f"-> Harvested thread: {title[:50]}...")
                
                # Be polite to the API
                time.sleep(0.5)
            except Exception as e:
                print(f"Skipping thread {story_id} due to error: {e}")
                continue

    print(f"[OK] Hacker News harvest complete! Data appended to {out_path}")

if __name__ == "__main__":
    harvest_hackernews()
