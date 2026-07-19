import sys
from pathlib import Path
import numpy as np

# Resolve imports for backend modules
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

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def test_exact_logits():
    print("Running exact logits match test (S <= 64)...")
    weights_dir = ROOT_DIR / "backend" / "weights"
    weight_manager = WeightManager(weights_dir, vocab_size=VOCAB_SIZE)
    personality = weight_manager.list_personalities()[0]
    
    model = PharmakonTransformer(vocab_size=VOCAB_SIZE)
    model.load_weights(weight_manager.get_weights(personality))
    
    prompt = "One morning, when Gregor Samsa woke from"
    prompt_indices = encode_prompt(prompt)
    
    input_nc = list(prompt_indices)
    input_c = list(prompt_indices)
    
    # Ingest prompt
    kv_caches = None
    logits_c = None
    for idx in prompt_indices:
        idx_arr = np.array([idx])
        logits_c, kv_caches = model.forward(idx_arr, use_cache=True, kv_caches=kv_caches)
        
    sampler = Sampler(temperature=0.8)
    next_idx = sampler.sample(logits_c)
    input_c.append(next_idx)
    input_nc.append(next_idx)
    
    for step in range(15):
        # 1. Non-cached
        logits_nc = model.forward(np.array(input_nc))
        # 2. Cached
        logits_c, kv_caches = model.forward(np.array([next_idx]), use_cache=True, kv_caches=kv_caches)
        
        diff = np.max(np.abs(logits_nc[-1] - logits_c))
        assert diff < 1e-10, f"Logits divergence at step {step}: max diff = {diff:.2e} exceeds threshold 1e-10"
        
        next_idx = sampler.sample(logits_c)
        input_c.append(next_idx)
        input_nc.append(next_idx)
        
    print("[OK] Exact logits test passed.")

def test_perplexity_drift():
    print("\nRunning perplexity drift test (S up to 500)...")
    weights_dir = ROOT_DIR / "backend" / "weights"
    weight_manager = WeightManager(weights_dir, vocab_size=VOCAB_SIZE)
    personality = weight_manager.list_personalities()[0]
    
    model = PharmakonTransformer(vocab_size=VOCAB_SIZE)
    model.load_weights(weight_manager.get_weights(personality))
    
    text = (
        "Someone must have slandered Josef K., for one morning, without having done anything wrong, he was arrested. "
        "The cook of Frau Grubach, his landlady, who each day brought him his breakfast at about eight in the morning, "
        "did not appear this time. That had never happened before. K. waited a little longer, looked from his pillow at "
        "the old woman who lived opposite and who was watching him with an inquisitiveness which was quite unusual for her, "
        "and then, both hungry and disturbed, he rang the bell. At once there was a knock at the door and a man he had never "
        "seen before in this house entered. He was slim but strongly built, he wore a close-fitting black suit, which, "
        "like travel-wear, had various pleats, pockets, buckles, buttons and a belt, and as a result appeared particularly practical."
    )
    
    # Try reading from files
    file_path = ROOT_DIR / "data" / "The Trial.txt"
    if file_path.exists():
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        
    import clean_corpus
    clean_text_str, _ = clean_corpus.clean_text(text)
    tokens = encode_prompt(clean_text_str[:500])
    
    # 1. Non-cached
    losses_nc = []
    for i in range(1, len(tokens)):
        prefix = tokens[:i]
        cropped = prefix[-64:]
        logits = model.forward(np.array(cropped))
        losses_nc.append(-np.log(softmax(logits[-1])[tokens[i]] + 1e-15))
        
    # 2. Cached
    losses_c = []
    kv_caches = None
    for i in range(len(tokens) - 1):
        idx = tokens[i]
        logits, kv_caches = model.forward(np.array([idx]), use_cache=True, kv_caches=kv_caches)
        losses_c.append(-np.log(softmax(logits[-1])[tokens[i+1]] + 1e-15))
        
    ppl_nc = np.exp(np.mean(losses_nc))
    ppl_c = np.exp(np.mean(losses_c))
    ppl_diff = abs(ppl_c - ppl_nc)
    
    print(f"  NC PPL: {ppl_nc:.4f}")
    print(f"  Cached PPL: {ppl_c:.4f}")
    print(f"  PPL Diff: {ppl_diff:.4f}")
    
    assert ppl_diff < 0.2, f"Perplexity drift {ppl_diff:.4f} exceeds threshold 0.2"
    print("[OK] Perplexity drift test passed.")

if __name__ == "__main__":
    test_exact_logits()
    test_perplexity_drift()
    print("\nALL REGRESSION TESTS PASSED.")
