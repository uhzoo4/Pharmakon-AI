import requests
import json

def test_generate_endpoint():
    url = "http://127.0.0.1:8000/api/generate"
    payload = {
        "personality": "the_assistant",
        "prompt": "Hello there!",
        "temperature": 0.8,
        "blacklist": ["z", "q"],
        "max_tokens": 50
    }
    
    print("Testing SSE stream from /api/generate...")
    
    with requests.post(url, json=payload, stream=True) as r:
        if r.status_code != 200:
            print(f"Error: HTTP {r.status_code}")
            print(r.text)
            return
            
        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    try:
                        data = json.loads(data_str)
                        if 'text' in data:
                            print(data['text'], end="", flush=True)
                        if 'done' in data and data['done']:
                            print("\n\n[Stream Finished successfully]")
                            break
                    except Exception as e:
                        print(f"\n[JSON parse error]: {e} on '{data_str}'")

if __name__ == "__main__":
    test_generate_endpoint()
