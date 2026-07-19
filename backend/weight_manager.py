"""
Pharmakon – In-Memory Weight Manager
=====================================
Handles loading, caching, and dynamic swapping of model weight arrays.
All weights are kept in RAM to enable instant personality switches (<1 ms).
On first run (empty weights/ folder), default personalities are automatically
generated using Xavier uniform initialization and saved as compressed .npz files.
"""
import json
import os
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
        mapped_personality = personality
        if personality not in self.personalities:
            mapping = {
                "kafkaesque": "The Trial",
                "camus_stranger": "Beyond Good and Evil",
                "dark_romance": "Wuthering Heights"
            }
            mapped_personality = mapping.get(personality, personality)

        if mapped_personality not in self.personalities:
            available = list(self.personalities.keys())
            raise KeyError(
                f"Unknown personality: '{personality}' (mapped to '{mapped_personality}'). "
                f"Available: {available}"
            )
        return self.personalities[mapped_personality]

    def list_personalities(self) -> List[str]:
        """Return a sorted list of all loaded personality names."""
        return sorted(self.personalities.keys())

    def save_weights(self, name: str, weights: Dict[str, np.ndarray], metadata: Dict | None = None) -> None:
        """Validate and atomically save personality weight dictionary to disk and update cache."""
        import tempfile
        import datetime

        # 1. Structural Verification: define expected shape for all model parameters
        shape_specs = {
            "token_embedding": (self.vocab_size, self.embed_dim),
            "W_out": (self.embed_dim, self.vocab_size),
            "ln_final_gamma": (self.embed_dim,),
            "ln_final_beta": (self.embed_dim,),
        }
        for i in range(self.num_layers):
            prefix = f"block_{i}_"
            shape_specs.update({
                prefix + "Wq": (self.embed_dim, self.embed_dim),
                prefix + "Wk": (self.embed_dim, self.embed_dim),
                prefix + "Wv": (self.embed_dim, self.embed_dim),
                prefix + "Wo": (self.embed_dim, self.embed_dim),
                prefix + "ln1_gamma": (self.embed_dim,),
                prefix + "ln1_beta": (self.embed_dim,),
                prefix + "ln2_gamma": (self.embed_dim,),
                prefix + "ln2_beta": (self.embed_dim,),
                prefix + "W1": (self.embed_dim, self.ff_dim),
                prefix + "b1": (self.ff_dim,),
                prefix + "W2": (self.ff_dim, self.embed_dim),
                prefix + "b2": (self.embed_dim,),
            })

        # 2. Numerical validation
        for key, expected_shape in shape_specs.items():
            if key not in weights:
                raise ValueError(f"Missing required parameter: {key}")
            val = weights[key]
            if val.shape != expected_shape:
                raise ValueError(f"Parameter {key} has invalid shape {val.shape}, expected {expected_shape}")
            if not np.isfinite(val).all():
                raise ValueError(f"Parameter {key} contains non-finite values (NaN or Inf)")

        # 3. Create JSON-serializable config state
        config = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "d_model": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.ff_dim,
            "sequence_length": 64
        }
        
        # 4. Create JSON-serializable metadata state
        meta = {
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "epochs": metadata.get("epochs", 0) if metadata else 0,
            "token_count": metadata.get("token_count", 0) if metadata else 0
        }

        # Filter out existing config/metadata arrays and convert to float32
        float_weights = {
            k: v.astype(np.float32) for k, v in weights.items() 
            if k not in ("config", "metadata")
        }

        # 5. Atomic check-point save via NamedTemporaryFile
        with tempfile.NamedTemporaryFile(
            mode="wb", 
            dir=self.weights_dir, 
            delete=False, 
            suffix=".tmp.npz"
        ) as tmp_f:
            np.savez_compressed(
                tmp_f,
                config=np.array(json.dumps(config)),
                metadata=np.array(json.dumps(meta)),
                **float_weights  # type: ignore[arg-type]
            )
            tmp_path = Path(tmp_f.name)

        final_path = self.weights_dir / f"{name}.npz"
        try:
            # Atomic swap using os.replace
            os.replace(tmp_path, final_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise e

        # 6. In-memory cache refresh (directly load dictionary from final file)
        self.personalities[name] = dict(np.load(final_path))