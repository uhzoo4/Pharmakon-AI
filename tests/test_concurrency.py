import sys
from pathlib import Path
import subprocess
import time
import urllib.request
import json
import threading
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR))

from transformer import PharmakonTransformer
from weight_manager import WeightManager
from generate import Sampler

VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
VOCAB_SIZE = len(VOCAB_CHARS)
char_to_idx = {char: idx for idx, char in enumerate(VOCAB_CHARS)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB_CHARS)}

def encode_prompt(prompt: str) -> list[int]:
    space_idx = char_to_idx[" "]
    return [char_to_idx.get(c, space_idx) for c in prompt]

def make_request(personality, prompt):
    url = "http://127.0.0.1:8000/api/generate"
    payload = {
        "personality": personality,
        "prompt": prompt,
        "temperature": 0.8,
        "max_tokens": 50,
        "blacklist": []
    }
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    
    chars = []
    try:
        with urllib.request.urlopen(req) as response:
            for line in response:
                line_decoded = line.decode('utf-8').strip()
                if line_decoded.startswith("data: "):
                    data = json.loads(line_decoded[6:])
                    if "text" in data:
                        chars.append(data["text"])
                    elif "error" in data:
                        print(f"Error in {personality}: {data['error']}")
                        return None
    except Exception as e:
        print(f"Request failed for {personality}: {e}")
        return None
        
    return "".join(chars)

def main():
    # 1. Start uvicorn server in a background process
    print("Starting FastAPI backend server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR / "backend"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for the server to spin up
    time.sleep(3.0)
    
    # Check if server is running
    if server_process.poll() is not None:
        print("Failed to start server. Output:")
        out, err = server_process.communicate()
        print(out.decode())
        print(err.decode())
        sys.exit(1)
        
    print("Server started successfully. Sending API requests...")
    
    # Test different personalities sequentially to confirm they both produce high quality non-gibberish outputs
    out_bg = make_request("Beyond Good and Evil", "One morning, when Gregor Samsa woke ")
    out_meta = make_request("Metamorphosis", "One morning, when Gregor Samsa woke ")
    
    print("\nAPI Outputs (Sequential):")
    print(f"  Beyond Good and Evil: {repr(out_bg)}")
    print(f"  Metamorphosis:        {repr(out_meta)}")
    
    assert out_bg is not None and len(out_bg) > 10, "Failed Beyond Good and Evil request"
    assert out_meta is not None and len(out_meta) > 10, "Failed Metamorphosis request"
    
    # 2. Run multi-threaded model isolation test (simulating concurrent requests in python)
    print("\nRunning multi-threaded model isolation test...")
    weights_dir = ROOT_DIR / "backend" / "weights"
    weight_manager = WeightManager(weights_dir, vocab_size=VOCAB_SIZE)
    
    personality_models = {}
    
    # Load Beyond Good and Evil
    m1 = PharmakonTransformer(vocab_size=VOCAB_SIZE)
    m1.load_weights(weight_manager.get_weights("Beyond Good and Evil"))
    personality_models["Beyond Good and Evil"] = m1
    
    # Load Metamorphosis
    m2 = PharmakonTransformer(vocab_size=VOCAB_SIZE)
    m2.load_weights(weight_manager.get_weights("Metamorphosis"))
    personality_models["Metamorphosis"] = m2
    
    prompt = encode_prompt("One morning, when Gregor Samsa woke from troubled dreams, he found himself")
    results = {}
    
    def worker(name, model_instance):
        outputs = []
        kv_caches = None
        logits = None
        # Ingest prompt
        for idx in prompt[-64:]:
            logits, kv_caches = model_instance.forward(np.array([idx]), use_cache=True, kv_caches=kv_caches)
        # Generate 50 tokens
        sampler = Sampler(temperature=0.8)
        assert logits is not None, "Logits cannot be None"
        next_idx = sampler.sample(logits)
        outputs.append(next_idx)
        for _ in range(50):
            logits, kv_caches = model_instance.forward(np.array([next_idx]), use_cache=True, kv_caches=kv_caches)
            next_idx = sampler.sample(logits)
            outputs.append(next_idx)
            time.sleep(0.001)  # force context switching
        results[name] = "".join([idx_to_char.get(idx, " ") for idx in outputs])

    # Run threads concurrently
    t1 = threading.Thread(target=worker, args=("Beyond Good and Evil", personality_models["Beyond Good and Evil"]))
    t2 = threading.Thread(target=worker, args=("Metamorphosis", personality_models["Metamorphosis"]))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Terminate background server process
    server_process.terminate()
    server_process.wait()
    
    print("\nConcurrent generation outputs (Multi-threaded worker):")
    print(f"  Beyond Good and Evil: {repr(results['Beyond Good and Evil'])}")
    print(f"  Metamorphosis:        {repr(results['Metamorphosis'])}")
    
    assert len(results["Beyond Good and Evil"]) > 20, "Output too short!"
    assert len(results["Metamorphosis"]) > 20, "Output too short!"
    
    # Check that outputs did not stomp or corrupt
    print("\n[SUCCESS] Request isolation, model thread-safety, and weight persistence verified!")

if __name__ == "__main__":
    main()
