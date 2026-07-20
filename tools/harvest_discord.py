import os
import sys
import asyncio
from pathlib import Path

def install_deps():
    try:
        import discord
    except ImportError:
        print("Installing discord.py...")
        import subprocess
        # use discord.py self-bot support if needed, but standard discord.py is usually fine for basic bot usage
        subprocess.check_call([sys.executable, "-m", "pip", "install", "discord.py"])

def clean_text(text: str) -> str:
    """Strip everything except our 97-character vocab."""
    if not text:
        return ""
    valid_chars = set(["\n", "\t"] + [chr(i) for i in range(32, 127)])
    cleaned = "".join([c for c in text if c in valid_chars])
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()

def run_discord_harvester():
    import discord

    token = os.environ.get("DISCORD_TOKEN")
    channel_id_str = os.environ.get("DISCORD_CHANNEL_ID")
    
    # Note: Setting IS_USER_TOKEN to True violates Discord ToS.
    # It is recommended to use a standard Bot Token.
    is_user_token = os.environ.get("DISCORD_IS_USER_TOKEN", "false").lower() == "true"

    if not token or not channel_id_str:
        print("[ERROR] Discord credentials missing!")
        print("Required Env Vars:")
        print("- DISCORD_TOKEN (Your bot token or user token)")
        print("- DISCORD_CHANNEL_ID (The ID of the channel to scrape)")
        print("- DISCORD_IS_USER_TOKEN (Set to 'true' if using a personal account token)")
        sys.exit(1)

    channel_id = int(channel_id_str)

    # Initialize client
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "discord_corpus.txt"

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        channel = client.get_channel(channel_id)
        if not channel:
            print(f"Could not find channel with ID {channel_id}")
            await client.close()
            return

        print(f"Harvesting history from #{channel.name}...")
        
        # Pull the last 500 messages
        messages = [message async for message in channel.history(limit=500)]
        messages.reverse() # chronological order
        
        with open(out_path, "a", encoding="utf-8") as f:
            for i in range(len(messages) - 1):
                msg1 = messages[i]
                msg2 = messages[i+1]
                
                # If msg1 is from User A, and msg2 is from User B, it's a conversation!
                if msg1.author != msg2.author and not msg1.author.bot and not msg2.author.bot:
                    user_text = clean_text(msg1.content)
                    assistant_text = clean_text(msg2.content)
                    
                    if user_text and assistant_text and len(user_text) > 5 and len(assistant_text) > 5:
                        # Avoid dumping URLs or bot commands
                        if "http" not in user_text and "http" not in assistant_text:
                            f.write(f"User: {user_text}\nAssistant: {assistant_text}\n\n")

        print(f"[OK] Discord harvest complete! Data appended to {out_path}")
        await client.close()

    # Run the client
    try:
        client.run(token, bot=not is_user_token)
    except Exception as e:
        print(f"Failed to connect to Discord: {e}")

if __name__ == "__main__":
    install_deps()
    run_discord_harvester()
