import os
import sys
import time
from pathlib import Path

def install_deps():
    try:
        import praw
    except ImportError:
        print("Installing praw (Python Reddit API Wrapper)...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "praw"])

def clean_text(text: str) -> str:
    """Strip everything except our 97-character vocab."""
    if not text:
        return ""
    valid_chars = set(["\n", "\t"] + [chr(i) for i in range(32, 127)])
    # Remove weird zero-width spaces or markdown links if possible
    cleaned = "".join([c for c in text if c in valid_chars])
    # Prevent excessive newlines
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()

def harvest_reddit():
    import praw

    # Check for credentials
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "PharmakonDataHarvester/1.0")

    if not client_id or not client_secret:
        print("[ERROR] Reddit API credentials missing!")
        print("To get these:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'create another app...' at the bottom.")
        print("3. Select 'script', name it 'Pharmakon', set redirect uri to 'http://localhost:8080'.")
        print("4. Set REDDIT_CLIENT_ID (under the name) and REDDIT_CLIENT_SECRET in your terminal environment.")
        sys.exit(1)

    print("Connecting to Reddit API...")
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "reddit_corpus.txt"

    # Subreddits good for conversational data
    subreddits = ["casualconversation", "askreddit", "explainlikeimfive", "tifu"]
    
    print(f"Harvesting from: {', '.join(subreddits)}")
    
    with open(out_path, "a", encoding="utf-8") as f:
        for sub_name in subreddits:
            print(f"-> r/{sub_name}")
            subreddit = reddit.subreddit(sub_name)
            
            # Fetch hot posts
            for submission in subreddit.hot(limit=20):
                # Avoid stickied mod posts or megathreads
                if submission.stickied:
                    continue
                    
                title = clean_text(submission.title)
                if not title:
                    continue
                    
                # We format the post title as the User prompt
                # And the top comment as the Assistant reply
                
                # Fetch comments
                submission.comments.replace_more(limit=0) # Flatten tree, ignore "load more"
                if not submission.comments:
                    continue
                    
                top_comment = submission.comments[0]
                body = clean_text(top_comment.body)
                
                if not body or body == "[deleted]" or body == "[removed]":
                    continue
                    
                # Write to corpus
                f.write(f"User: {title}\nAssistant: {body}\n\n")
                
                # Sleep to respect rate limits
                time.sleep(0.5)

    print(f"[OK] Reddit harvest complete! Data appended to {out_path}")

if __name__ == "__main__":
    install_deps()
    harvest_reddit()
