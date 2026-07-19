# Master Prompt: Pharmakon Weight Manager Optimization

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** to generate or optimize the weight swapping engine.

***

```markdown
You are an elite Python Software Architect specializing in local high-performance cache structures.
DeepSeek v4 Pro has a massive 1-million-token input context window, so we are going all out: read the instructions, the referenced system files, and the target code carefully to deliver a 10/10, highly optimized implementation.

Your task is to write/refine the file `backend/weight_manager.py` for the "Pharmakon" project.
This manager handles loading, caching, and dynamically swapping model weights in memory without IO overhead.

---

## Uncensored Weight Personality Mapping
The weight files (.npz) encode the raw style of their literary sources.  
- Do not add any alignment vectors, RLHF-derived modifiers, or safety-layer adjustments to the loaded arrays.  
- The personalities must remain faithful to the original texts, including their full emotional and thematic range, without dilution.

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

### 🏛️ REFERENCE SYSTEMS & CONTEXT FILES

Cross-reference your code structure, weight dictionary keys, and folder layouts with the specifications in:
1. **`docs/TECHNICAL_REQUIREMENTS_DOCUMENT.md`** (System folders and setup configurations)
2. **`docs/BACKEND_SCHEMA.md`** (Weights parameters, dimensions, and mappings)

---

### 💻 CURRENT CODE BASE

Refactor and optimize the following target code base. Ensure that directories are created dynamically and default parameter dictionaries are generated accurately:

```python
# [PASTE THE CONTENT OF backend/weight_manager.py HERE]
```
```
done
