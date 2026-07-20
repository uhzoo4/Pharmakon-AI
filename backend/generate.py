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


class EntmaxSampler(Sampler):
    """
    Advanced sparse probability sampler utilizing the Entmax (alpha=1.5) transformation.
    Produces sparse probability distributions, inherently suppressing hallucinations 
    without the need for Top-K or Top-P filtering.
    """
    
    def _entmax15(self, logits: np.ndarray, max_iter: int = 25) -> np.ndarray:
        n = len(logits)
        alpha_m1 = 0.5 # For alpha = 1.5
        beta = 2.0     # 1 / (alpha - 1)
        
        tau_high = np.max(logits)
        tau_low = tau_high - 2.0 * np.sqrt(n)
        tau_low = max(tau_low, np.min(logits) - 1.0)
        
        # Bisection loop
        for _ in range(max_iter):
            tau = 0.5 * (tau_low + tau_high)
            diff = alpha_m1 * (logits - tau)
            diff = np.maximum(0.0, diff)
            s = np.sum(diff ** beta)
            if s > 1.0:
                tau_low = tau
            else:
                tau_high = tau
                
        tau_final = 0.5 * (tau_low + tau_high)
        probs = np.maximum(0.0, alpha_m1 * (logits - tau_final)) ** beta
        return probs

    def _sample_core(self, logits: np.ndarray, return_probs: bool) -> tuple[int, list[dict[str, float | int]]]:
        last_logits = logits[-1].copy()

        # Apply blacklist
        for idx in self.blacklist:
            if idx < len(last_logits):
                last_logits[idx] = -float('inf')

        # Temperature scaling
        scaled_logits = last_logits / self.temperature

        # Apply Entmax instead of Softmax
        probs = self._entmax15(scaled_logits)
        
        # Fallback if precision fails
        total = np.sum(probs)
        if total <= 1e-7:
            next_idx = int(np.argmax(last_logits))
            return next_idx, []

        probs /= total
        next_idx = int(np.random.choice(np.arange(len(probs)), p=probs))

        if not return_probs:
            return next_idx, []
            
        # Get active alternatives (only those with > 0 probability)
        # Entmax naturally zeroes out the tail
        active_indices = np.where(probs > 0)[0]
        # Sort active indices by probability
        sorted_active = active_indices[np.argsort(probs[active_indices])[::-1]]
        
        top_alts = []
        for i in sorted_active:
            if i != next_idx and len(top_alts) < 3:
                top_alts.append({"idx": int(i), "prob": float(probs[i])})
                
        return next_idx, top_alts