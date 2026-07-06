# Master Prompt: Pharmakon Weight Manager Optimization

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** to generate or optimize the weight swapping engine.

***

```markdown
You are an elite Python Software Architect specializing in local high-performance cache structures.
Your task is to write/refine the file `backend/weight_manager.py` for the "Pharmakon" project.
This manager handles loading, caching, and dynamically swapping model weights in memory without IO overhead.

---

### 🏛️ CORE DESIGN CONSTRAINTS
1. **Dynamic Initialization:**
   If the target `weights/` directory has no `.npz` files on startup, the class must automatically generate default parameter weights for three literary personalities:
   - `kafkaesque.npz`
   - `camus_stranger.npz`
   - `dark_romance.npz`
2. **Xavier Uniform Initialization:**
   The generated default weight matrices must use Xavier Uniform bounds:
   $$\text{limit} = \sqrt{\frac{6.0}{\text{fan\_in} + \text{fan\_out}}}$$
   Parameters must be saved as compressed NumPy arrays using `np.savez_compressed`.
3. **In-Memory Swapping:**
   Load all weights during start-up into a memory dictionary:
   ```python
   self.personalities = {
       "kafkaesque": {"token_embedding": np.array(...), ...},
       ...
   }
   ```
   Provide `get_weights(personality_name)` to retrieve references instantly. Swapping is handled as an in-memory reference pointer swap in the model execution loop to maintain zero latency ($<1\text{ ms}$).

---

### 💻 IMPLEMENTATION BLUEPRINT

Write the file `backend/weight_manager.py` implementing:
1. `WeightManager.__init__(weights_dir, vocab_size=97, embed_dim=64, num_heads=4, ff_dim=128, num_layers=2)`:
   - Create directories if they do not exist.
   - Run `_initialize_defaults_if_empty()` to populate weights if empty.
   - Run `_load_all()` to populate the RAM cache dictionary.
2. `_generate_xavier_weights()`: Helper to create the default parameter sets matching the v3.1 Spec shapes (token embeddings, outputs projection, layers 0/1 projections and biases).
3. `get_weights(personality)`: Return the weights map. Raise key error if not found.
4. `list_personalities()`: Return keys.

Go all out. Provide clean, deterministic, production-grade Python code.
```
