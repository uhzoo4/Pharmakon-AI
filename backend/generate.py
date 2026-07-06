import numpy as np

class Sampler:
    """
    Statistical engine for next-character selection from logits.

    Applies temperature scaling, user-defined blacklisting, and handles
    collision fallbacks without any content-based intervention.
    """

    def __init__(self, temperature: float = 0.8, blacklist: list | None = None):
        # Enforce safe upper temperature bound (as per API spec)
        if temperature <= 0.0:
            temperature = 0.0
        self.temperature = min(temperature, 2.0)
        self.blacklist = blacklist or []

    def sample(self, logits: np.ndarray) -> int:
        """
        Sample a single token index from the last position of the logits array.

        Args:
            logits: shape (seq_len, vocab_size), raw model outputs.

        Returns:
            Integer index of the sampled token.
        """
        last_logits = logits[-1]

        # 1. Greedy argmax fallback for extremely low temperatures
        if self.temperature <= 0.05:
            # Attempt to respect blacklist by finding highest non‑blacklisted index
            sorted_indices = np.argsort(last_logits)[::-1]
            for idx in sorted_indices:
                if idx not in self.blacklist:
                    return int(idx)
            # All indices are blacklisted – fallback to pure argmax (no blacklist)
            return int(np.argmax(last_logits))

        # 2. Temperature scaling and numerically stable softmax
        scaled_logits = last_logits / self.temperature
        shifted = scaled_logits - np.max(scaled_logits)
        probs = np.exp(shifted)
        denom = np.sum(probs)
        if denom == 0:                              # edge case (should never happen)
            return int(np.argmax(last_logits))
        probs /= denom

        # 3. Apply user blacklist – set blacklisted indices to 0.0
        for idx in self.blacklist:
            if idx < len(probs):
                probs[idx] = 0.0

        total = np.sum(probs)

        # 4. Vowel collision recovery – fallback to greedy on original logits
        if total <= 1e-7:
            return int(np.argmax(last_logits))

        # 5. Renormalize and sample stochastically
        probs /= total
        return int(np.random.choice(len(probs), p=probs))
        