import re
import json
from typing import Dict, List, Tuple

PRE_TOKEN_RE = re.compile(r'\w+|\s|[^\w\s]')

class BPETokenizer:
    def __init__(self):
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: List[str] = []
        self.vocab_to_id: Dict[str, int] = {}
        
    def _count_words(self, text: str) -> Dict[str, int]:
        counts = {}
        for word in PRE_TOKEN_RE.findall(text):
            counts[word] = counts.get(word, 0) + 1
        return counts

    def _merge_tokens(self, tokens: List[str], pair: Tuple[str, str], merged: str) -> List[str]:
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def train(self, text: str, vocab_size: int = 1000):
        print(f"[BPE] Training tokenizer with target vocab size: {vocab_size}")
        word_freqs = self._count_words(text)
        
        # Init word splits (characters)
        word_splits = {word: list(word) for word in word_freqs.keys()}
        
        # Track base characters
        base_chars = set()
        for word in word_splits.keys():
            for c in word:
                base_chars.add(c)
                
        # Ensure standard ASCII chars are always in vocab to prevent OOV on rare chars
        standard_chars = set(["\n", "\t"] + [chr(i) for i in range(32, 127)])
        base_chars.update(standard_chars)
        
        # Number of merges we can perform to hit target vocab size
        num_base = len(base_chars)
        max_merges = max(0, vocab_size - num_base)
        print(f"[BPE] Base character vocab: {num_base}. Will perform {max_merges} merges.")
        
        self.merges = {}
        merges_list = []
        
        for step in range(max_merges):
            pair_freq = {}
            for word, tokens in word_splits.items():
                weight = word_freqs[word]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + weight
                    
            if not pair_freq:
                break
                
            best_pair = max(pair_freq.keys(), key=lambda k: pair_freq[k])
            merged_token = best_pair[0] + best_pair[1]
            
            for word, tokens in word_splits.items():
                word_splits[word] = self._merge_tokens(tokens, best_pair, merged_token)
                
            self.merges[best_pair] = merged_token
            merges_list.append({"pair": best_pair, "merged": merged_token})
            
            if step % 200 == 0:
                print(f"[BPE] Step {step}/{max_merges}: Merged '{best_pair[0]}' + '{best_pair[1]}' -> '{merged_token}'")

        # Build final vocabulary mapping
        self.vocab = sorted(list(base_chars)) + [m["merged"] for m in merges_list]
        self.vocab_to_id = {t: i for i, t in enumerate(self.vocab)}
        print(f"[BPE] Training complete! Final Vocab Size: {len(self.vocab)}")
        
    def save(self, filepath: str):
        data = {
            "merges": [{"pair": list(pair), "merged": merged} for pair, merged in self.merges.items()],
            "vocab": self.vocab
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            
    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.merges = {tuple(m["pair"]): m["merged"] for m in data["merges"]}
        self.vocab = data["vocab"]
        self.vocab_to_id = {t: i for i, t in enumerate(self.vocab)}
        
    def encode(self, text: str) -> List[int]:
        if not self.vocab_to_id:
            raise ValueError("Tokenizer not trained or loaded.")
            
        pre_tokens = PRE_TOKEN_RE.findall(text)
        encoded_ids = []
        
        for pre_token in pre_tokens:
            tokens = list(pre_token)
            # Replay merges in exact order they were learned
            for pair, merged in self.merges.items():
                tokens = self._merge_tokens(tokens, pair, merged)
                
            for t in tokens:
                # Fallback to <UNK> token index 0 if something went wrong
                encoded_ids.append(self.vocab_to_id.get(t, 0))
                
        return encoded_ids
        
    def decode(self, ids: List[int]) -> str:
        if not self.vocab:
            raise ValueError("Tokenizer not trained or loaded.")
        return "".join([self.vocab[i] if i < len(self.vocab) else "" for i in ids])

# Quick testing
if __name__ == "__main__":
    t = BPETokenizer()
    sample = "the cat sat on the mat"
    t.train(sample, vocab_size=50) # Very small vocab for test
    ids = t.encode("the cat sat")
    print("Tokens:", ids)
    print("Decoded:", t.decode(ids))
