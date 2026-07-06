import numpy as np
from pathlib import Path

class WeightManager:
    def __init__(self, weights_dir: str, vocab_size=97, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2):
        self.weights_dir = Path(weights_dir)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        
        # Ensure the directory exists
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        self.personalities = {}  # name -> dict of arrays
        self._initialize_defaults_if_empty()
        self._load_all()

    def _initialize_defaults_if_empty(self):
        """If weights directory has no .npz files, generate and save randomized default files."""
        existing_npz = list(self.weights_dir.glob("*.npz"))
        if not existing_npz:
            print("No personality weights found. Creating default grimoire personalities...")
            default_personalities = ["kafkaesque", "camus_stranger", "dark_romance"]
            for name in default_personalities:
                weights = self._generate_xavier_weights()
                save_path = self.weights_dir / f"{name}.npz"
                np.savez_compressed(save_path, **weights)
                print(f"Initialized default weights: {save_path}")

    def _generate_xavier_weights(self) -> dict:
        """Helper to generate Xavier uniform initialized weights."""
        weights = {}
        
        # Vocab embedding weights
        limit_embed = np.sqrt(6.0 / (self.vocab_size + self.embed_dim))
        weights['token_embedding'] = np.random.uniform(-limit_embed, limit_embed, (self.vocab_size, self.embed_dim)).astype(np.float32)
        weights['W_out'] = np.random.uniform(-limit_embed, limit_embed, (self.embed_dim, self.vocab_size)).astype(np.float32)
        
        # Final Norm parameters
        weights['ln_final_gamma'] = np.ones(self.embed_dim, dtype=np.float32)
        weights['ln_final_beta'] = np.zeros(self.embed_dim, dtype=np.float32)
        
        # Block-level parameters
        limit_attn = np.sqrt(6.0 / (2.0 * self.embed_dim))
        limit_ff1 = np.sqrt(6.0 / (self.embed_dim + self.ff_dim))
        limit_ff2 = np.sqrt(6.0 / (self.ff_dim + self.embed_dim))
        
        for i in range(self.num_layers):
            prefix = f'block_{i}_'
            weights[prefix+'Wq'] = np.random.uniform(-limit_attn, limit_attn, (self.embed_dim, self.embed_dim)).astype(np.float32)
            weights[prefix+'Wk'] = np.random.uniform(-limit_attn, limit_attn, (self.embed_dim, self.embed_dim)).astype(np.float32)
            weights[prefix+'Wv'] = np.random.uniform(-limit_attn, limit_attn, (self.embed_dim, self.embed_dim)).astype(np.float32)
            weights[prefix+'Wo'] = np.random.uniform(-limit_attn, limit_attn, (self.embed_dim, self.embed_dim)).astype(np.float32)
            weights[prefix+'ln1_gamma'] = np.ones(self.embed_dim, dtype=np.float32)
            weights[prefix+'ln1_beta'] = np.zeros(self.embed_dim, dtype=np.float32)
            weights[prefix+'ln2_gamma'] = np.ones(self.embed_dim, dtype=np.float32)
            weights[prefix+'ln2_beta'] = np.zeros(self.embed_dim, dtype=np.float32)
            weights[prefix+'W1'] = np.random.uniform(-limit_ff1, limit_ff1, (self.embed_dim, self.ff_dim)).astype(np.float32)
            weights[prefix+'b1'] = np.zeros(self.ff_dim, dtype=np.float32)
            weights[prefix+'W2'] = np.random.uniform(-limit_ff2, limit_ff2, (self.ff_dim, self.embed_dim)).astype(np.float32)
            weights[prefix+'b2'] = np.zeros(self.embed_dim, dtype=np.float32)
            
        return weights

    def _load_all(self):
        """Pre-load all .npz files into memory."""
        for npz_path in self.weights_dir.glob("*.npz"):
            name = npz_path.stem
            self.personalities[name] = dict(np.load(npz_path))

    def get_weights(self, personality: str) -> dict:
        if personality not in self.personalities:
            raise KeyError(f"Unknown personality: {personality}. Available: {list(self.personalities.keys())}")
        return self.personalities[personality]

    def list_personalities(self):
        return list(self.personalities.keys())
