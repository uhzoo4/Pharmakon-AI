import numpy as np

class Sampler:
    def __init__(self, temperature=0.8, blacklist=None):
        self.temperature = temperature
        self.blacklist = blacklist or []

    def sample(self, logits):
        """Sample a token index from logits (last position).
        logits: shape (seq_len, vocab_size)
        """
        last_logits = logits[-1]
        
        # 1. Low temperature handling (Greedy argmax fallback)
        if self.temperature <= 0.05:
            idx = np.argmax(last_logits)
            # If the greedy index is blacklisted, we attempt to find the highest non-blacklisted index
            if idx in self.blacklist:
                sorted_indices = np.argsort(last_logits)[::-1]
                for alternative_idx in sorted_indices:
                    if alternative_idx not in self.blacklist:
                        return int(alternative_idx)
            return int(idx)

        # 2. Scale by temperature
        scaled_logits = last_logits / self.temperature
        
        # Numerically stable softmax probabilities
        probs = np.exp(scaled_logits - np.max(scaled_logits))
        probs /= np.sum(probs)

        # 3. Apply blacklist
        non_blacklisted_sum = 0.0
        for idx in self.blacklist:
            if idx < len(probs):
                probs[idx] = 0.0
        
        # 4. Blacklist vowel collision fallback
        probs_sum = np.sum(probs)
        if probs_sum <= 1e-7:
            # Vowel collision occurred (all probable tokens blacklisted).
            # Fallback: remove blacklist, do greedy sampling on original logits.
            return int(np.argmax(last_logits))

        # Re-normalize
        probs /= probs_sum

        # 5. Stochastic sampling
        return int(np.random.choice(len(probs), p=probs))
