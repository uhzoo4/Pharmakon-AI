import sys
import numpy as np
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from transformer import PharmakonTransformer
from generate import EntmaxSampler

# 1. Configuration (must match modern_conversation.npz training)
VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
char_to_idx = {c: i for i, c in enumerate(VOCAB_CHARS)}
idx_to_char = {i: c for i, c in enumerate(VOCAB_CHARS)}
VOCAB_SIZE = len(VOCAB_CHARS)

def load_pinnacle_weights(path: str):
    """Loads weights and instantiates the Pharmakon architecture."""
    weights = np.load(path)
    # The modern_conversation.npz was trained with embed_dim=64, num_heads=4, ff_dim=128, num_layers=2
    # If using 'the_pinnacle.npz' it would be 128, 8, 256, 4
    model = PharmakonTransformer(
        vocab_size=VOCAB_SIZE, 
        embed_dim=64, 
        num_heads=4, 
        ff_dim=128, 
        num_layers=2, 
        max_seq_len=64
    )
    
    # Load parameters (numpy array dictionary -> model attributes)
    model.token_embedding = weights['token_embedding']
    model.W_out = weights['W_out']
    model.ln_final.gamma = weights['ln_final_gamma']
    model.ln_final.beta = weights['ln_final_beta']
    
    for i, block in enumerate(model.blocks):
        p = f'block_{i}_'
        block.Wq = weights[p+'Wq']
        block.Wk = weights[p+'Wk']
        block.Wv = weights[p+'Wv']
        block.Wo = weights[p+'Wo']
        block.ln1.gamma = weights[p+'ln1_gamma']
        block.ln1.beta = weights[p+'ln1_beta']
        block.ln2.gamma = weights[p+'ln2_gamma']
        block.ln2.beta = weights[p+'ln2_beta']
        block.W1 = weights[p+'W1']
        block.b1 = weights[p+'b1']
        block.W2 = weights[p+'W2']
        block.b2 = weights[p+'b2']
        
    return model

def encode(text: str) -> list[int]:
    return [char_to_idx.get(c, char_to_idx[' ']) for c in text]

def decode(tokens: list[int]) -> str:
    return "".join([idx_to_char.get(t, " ") for t in tokens])

def generate_response(model, prompt_text: str, max_length: int = 150):
    print(f"\n[Prompt]: {prompt_text}")
    print("[Response]: ", end="", flush=True)
    
    input_indices = encode(prompt_text)
    sampler = EntmaxSampler(temperature=0.8)
    
    # We maintain KV cache explicitly if supported, or just forward pass
    # Since PharmakonTransformer supports KV cache, let's use it for speed!
    kv_cache = None
    
    for _ in range(max_length):
        # We only pass the new token if using KV cache
        # Actually, for simplicity and guaranteed correctness (like your script requested), 
        # let's just pass the sliding window.
        window = input_indices[-64:] 
        x = np.array([window], dtype=np.int32)
        
        logits = model.forward(x)
        
        # Use our Entmax Engine to select the next character!
        next_idx = sampler.sample(logits[0])
        input_indices.append(next_idx)
        
        char = idx_to_char.get(next_idx, " ")
        print(char, end="", flush=True)
        
        if char == "\n" and len(input_indices) > len(prompt_text) + 2:
            break
            
    print("\n")
    return input_indices

if __name__ == "__main__":
    # Use the instruction-tuned assistant model which understands the User/Assistant prompt format
    weight_path = "backend/weights/the_assistant.npz"
    if not Path(weight_path).exists():
        print(f"Weights not found at {weight_path}")
        sys.exit(1)
        
    print(f"Loading {weight_path} into Pharmakon C/NumPy Architecture...")
    model = load_pinnacle_weights(weight_path)
    
    # Let's run a test!
    generate_response(model, "User: How are you?\nAssistant:", max_length=100)
