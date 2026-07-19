import numpy as np

class Sampler:
    """
    Statistical engine for next-character selection from logits.

    Implements Temperature scaling, Top-K, and Nucleus (Top-p) sampling.
    """

    def __init__(self, temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9, blacklist: list | None = None):
        self.temperature = max(0.01, min(temperature, 2.0))
        self.top_k = top_k
        self.top_p = top_p
        self.blacklist = blacklist or []

    def sample(self, logits: np.ndarray) -> int:
        """Sample a token index without returning probabilities."""
        next_idx, _ = self._sample_core(logits, return_probs=False)
        return next_idx

    def sample_with_probs(self, logits: np.ndarray) -> tuple[int, list[dict[str, float | int]]]:
        """Sample a token index and return top 3 alternative probabilities."""
        return self._sample_core(logits, return_probs=True) # type: ignore

    def _sample_core(self, logits: np.ndarray, return_probs: bool) -> tuple[int, list[dict[str, float | int]]]:
        """Core sampling logic."""
        last_logits = logits[-1].copy()

        # Apply blacklist
        for idx in self.blacklist:
            if idx < len(last_logits):
                last_logits[idx] = -float('inf')

        # Temperature scaling
        scaled_logits = last_logits / self.temperature
        
        # Numerically stable softmax
        shifted = scaled_logits - np.max(scaled_logits)
        probs = np.exp(shifted)
        probs /= np.sum(probs)

        # Top-K Filtering
        if self.top_k > 0:
            indices_to_remove = np.argsort(probs)[::-1][self.top_k:]
            probs[indices_to_remove] = 0.0
            probs /= np.sum(probs)

        # Top-p (Nucleus) Filtering
        if self.top_p > 0.0 and self.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Remove tokens with cumulative probability above the threshold
            cutoff_idx = int(np.searchsorted(cumulative_probs, self.top_p))
            if cutoff_idx < len(sorted_indices):
                indices_to_remove = sorted_indices[cutoff_idx + 1:]
                probs[indices_to_remove] = 0.0
                probs /= np.sum(probs)

        # Fallback to pure greedy if probs sum to 0
        total = np.sum(probs)
        if total <= 1e-7:
            next_idx = int(np.argmax(last_logits))
            return next_idx, []

        # Renormalize and sample
        probs /= total
        next_idx = int(np.random.choice(np.arange(len(probs)), p=probs))

        if not return_probs:
            return next_idx, []
            
        # Get top 3 alternatives for UI probabilistic visualization
        top_indices = np.argsort(probs)[::-1][:3]
        top_alts = [{"idx": int(i), "prob": float(probs[i])} for i in top_indices if i != next_idx]
        
        return next_idx, top_alts