"""
Pharmakon – In-Memory Weight Manager
=====================================
Handles loading, caching, and dynamic swapping of model weight arrays.
All weights are kept in RAM to enable instant personality switches (<1 ms).
On first run (empty weights/ folder), default personalities are automatically
generated using Xavier uniform initialization and saved as compressed .npz files.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List


class WeightManager:
    """
    Manages model weights for different literary personalities.

    On instantiation, the manager:
        1. Creates the weights directory if it does not exist.
        2. If the directory is empty, generates three default personalities
           ('kafkaesque', 'camus_stranger', 'dark_romance') with Xavier‑uniform
           initialised parameters and persists them as .npz files.
        3. Loads all .npz files into a memory‑resident dictionary.
    """

    def __init__(
        self,
        weights_dir: str | Path = "weights",
        vocab_size: int = 97,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_layers: int = 2,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        # Ensure the directory exists (idempotent)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: personality_name -> dict of numpy arrays
        self.personalities: Dict[str, Dict[str, np.ndarray]] = {}

        # Bootstrap default weights if no .npz files are present
        self._initialize_defaults_if_empty()

        # Load everything into RAM
        self._load_all()

    # ------------------------------------------------------------------
    # Default generation
    # ------------------------------------------------------------------
    def _initialize_defaults_if_empty(self) -> None:
        """If the weights directory has no .npz files, generate and save
        the three default literary personalities."""
        existing_npz = list(self.weights_dir.glob("*.npz"))
        if existing_npz:
            return

        print("No personality weights found. Creating default grimoire personalities...")
        default_personalities = ["kafkaesque", "camus_stranger", "dark_romance"]

        for name in default_personalities:
            weights = self._generate_xavier_weights()
            save_path = self.weights_dir / f"{name}.npz"
            np.savez_compressed(save_path, **weights)  # type: ignore[arg-type]
            print(f"  [OK] Initialized default weights: {save_path}")

    def _generate_xavier_weights(self) -> Dict[str, np.ndarray]:
        """Generate a full set of model parameters using Xavier uniform
        initialisation (Glorot uniform)."""
        weights: Dict[str, np.ndarray] = {}

        # --- Embedding & output head ---
        limit_embed = np.sqrt(6.0 / (self.vocab_size + self.embed_dim))
        weights["token_embedding"] = np.random.uniform(
            -limit_embed, limit_embed, (self.vocab_size, self.embed_dim)
        ).astype(np.float32)
        weights["W_out"] = np.random.uniform(
            -limit_embed, limit_embed, (self.embed_dim, self.vocab_size)
        ).astype(np.float32)

        # --- Final layer norm (output) ---
        weights["ln_final_gamma"] = np.ones(self.embed_dim, dtype=np.float32)
        weights["ln_final_beta"] = np.zeros(self.embed_dim, dtype=np.float32)

        # --- Attention & FFN blocks ---
        limit_attn = np.sqrt(6.0 / (2.0 * self.embed_dim))
        limit_ff1 = np.sqrt(6.0 / (self.embed_dim + self.ff_dim))
        limit_ff2 = np.sqrt(6.0 / (self.ff_dim + self.embed_dim))

        for i in range(self.num_layers):
            prefix = f"block_{i}_"

            # Attention weights
            for proj in ("Wq", "Wk", "Wv", "Wo"):
                weights[prefix + proj] = np.random.uniform(
                    -limit_attn, limit_attn, (self.embed_dim, self.embed_dim)
                ).astype(np.float32)

            # Layer norms (pre‑attention, pre‑FFN)
            weights[prefix + "ln1_gamma"] = np.ones(self.embed_dim, dtype=np.float32)
            weights[prefix + "ln1_beta"] = np.zeros(self.embed_dim, dtype=np.float32)
            weights[prefix + "ln2_gamma"] = np.ones(self.embed_dim, dtype=np.float32)
            weights[prefix + "ln2_beta"] = np.zeros(self.embed_dim, dtype=np.float32)

            # Feed‑forward network
            weights[prefix + "W1"] = np.random.uniform(
                -limit_ff1, limit_ff1, (self.embed_dim, self.ff_dim)
            ).astype(np.float32)
            weights[prefix + "b1"] = np.zeros(self.ff_dim, dtype=np.float32)
            weights[prefix + "W2"] = np.random.uniform(
                -limit_ff2, limit_ff2, (self.ff_dim, self.embed_dim)
            ).astype(np.float32)
            weights[prefix + "b2"] = np.zeros(self.embed_dim, dtype=np.float32)

        return weights

    # ------------------------------------------------------------------
    # Loading & retrieval
    # ------------------------------------------------------------------
    def _load_all(self) -> None:
        """Load every .npz file into the in‑memory personalities dictionary."""
        for npz_path in self.weights_dir.glob("*.npz"):
            name = npz_path.stem
            try:
                # np.load returns a lazy NpzFile; dict() forces immediate loading.
                self.personalities[name] = dict(np.load(npz_path))
                print(f"  Loaded personality '{name}' ({len(self.personalities[name])} tensors).")
            except Exception as exc:
                print(f"  [WARNING] Failed to load {npz_path}: {exc}")

    def get_weights(self, personality: str) -> Dict[str, np.ndarray]:
        """Return the weight dictionary for the requested personality.
        This is an O(1) in‑memory lookup; the caller can use the reference
        to instantly swap the model's active parameters."""
        if personality not in self.personalities:
            available = list(self.personalities.keys())
            raise KeyError(
                f"Unknown personality: '{personality}'. "
                f"Available: {available}"
            )
        return self.personalities[personality]

    def list_personalities(self) -> List[str]:
        """Return a sorted list of all loaded personality names."""
        return sorted(self.personalities.keys())