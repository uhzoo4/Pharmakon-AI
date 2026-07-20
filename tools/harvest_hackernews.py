import sys
import asyncio
from pathlib import Path

def install_deps():
    try:
        import aiohttp
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])

def clean_text(text: str) -> str:
    """Strip everything except our 97-character vocab and remove HTML."""
    if not text:
        return ""
    text = text.replace("<p>", "\n\n").replace("&#x27;", "'").replace("&quot;", '"').replace("&gt;", ">").replace("&lt;", "<").replace("&#x2F;", "/")
    
    valid_chars = set(["\n", "\t"] + [chr(i) for i in range(32, 127)])
    cleaned = "".join([c for c in text if c in valid_chars])
    
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()

async def fetch_item(session, item_id, semaphore):
    async with semaphore:
        try:
            async with session.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
    return None

async def process_story(session, story_id, semaphore, out_file):
    story_data = await fetch_item(session, story_id, semaphore)
    if not story_data: return 0
    
    title = clean_text(story_data.get("title", ""))
    kids = story_data.get("kids", [])
    
    if not title or not kids: return 0
    
    # Fetch up to 5 top level comments for this story
    top_kids = kids[:5]
    tasks = [fetch_item(session, kid_id, semaphore) for kid_id in top_kids]
    comments = await asyncio.gather(*tasks)
    
    harvested = 0
    for comment_data in comments:
        if not comment_data: continue
        body = clean_text(comment_data.get("text", ""))
        if not body or body == "[deleted]": continue
        
        # Write format
        out_file.write(f"User: {title}\nAssistant: {body}\n\n")
        harvested += len(title) + len(body)
        
    return harvested

async def harvest_hackernews_async():
    print("Connecting to Hacker News API (No Auth Required) for MASSIVE SCALE HARVEST...")
    
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "reddit_corpus.txt"
    
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        # Get all 500 top stories
        async with session.get("https://hacker-news.firebaseio.com/v0/topstories.json") as resp:
            top_ids = await resp.json()
            
        print(f"Discovered {len(top_ids)} live threads. Booting up massive parallel ingestion...")
        
        # Limit concurrent connections to avoid crashing or getting IP banned (Hacker News Firebase allows quite a bit)
        semaphore = asyncio.Semaphore(50)
        
        total_chars = 0
        with open(out_path, "a", encoding="utf-8") as f:
            tasks = [process_story(session, sid, semaphore, f) for sid in top_ids]
            
            # Use asyncio.as_completed for progress bar effect
            completed = 0
            for coro in asyncio.as_completed(tasks):
                chars = await coro
                total_chars += chars
                completed += 1
                if completed % 50 == 0:
                    print(f"Processed {completed}/500 threads... Harvested {total_chars} characters so far.")

        print(f"[OK] Massive Harvest Complete! {total_chars} total characters appended to {out_path}")

def harvest_hackernews():
    install_deps()
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(harvest_hackernews_async())

if __name__ == "__main__":
    harvest_hackernews()
