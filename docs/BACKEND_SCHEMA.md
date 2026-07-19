# Pharmakon – Backend Schema Specification

This document defines the structural layouts of backend memory, weights storage formats, vocabulary encoding tables, and JSON communication schemas.

---

## 1. Weights Storage Schema (`.npz` Layout)

Each literary personality is stored as a compressed `.npz` file containing serialized NumPy arrays. The keys and array dimensions are strictly defined below:

| Key Name | Target Layer Component | Array Shape | Data Type | Initialization Range (Xavier Uniform) |
| --- | --- | --- | --- | --- |
| `token_embedding` | Input Embedding Table | `(vocab_size, embed_dim)` | `float32` | $\pm \sqrt{6 / (\text{vocab} + \text{embed})}$ |
| `block_i_Wq` | Layer `i` Query Weights | `(embed_dim, embed_dim)` | `float32` | $\pm \sqrt{6 / (2 \cdot \text{embed})}$ |
| `block_i_Wk` | Layer `i` Key Weights | `(embed_dim, embed_dim)` | `float32` | $\pm \sqrt{6 / (2 \cdot \text{embed})}$ |
| `block_i_Wv` | Layer `i` Value Weights | `(embed_dim, embed_dim)` | `float32` | $\pm \sqrt{6 / (2 \cdot \text{embed})}$ |
| `block_i_Wo` | Layer `i` Output Project | `(embed_dim, embed_dim)` | `float32` | $\pm \sqrt{6 / (2 \cdot \text{embed})}$ |
| `block_i_ln1_gamma` | Layer `i` Attention LN Scale | `(embed_dim,)` | `float32` | Initialize to `1.0` |
| `block_i_ln1_beta` | Layer `i` Attention LN Offset | `(embed_dim,)` | `float32` | Initialize to `0.0` |
| `block_i_ln2_gamma` | Layer `i` FFN LN Scale | `(embed_dim,)` | `float32` | Initialize to `1.0` |
| `block_i_ln2_beta` | Layer `i` FFN LN Offset | `(embed_dim,)` | `float32` | Initialize to `0.0` |
| `block_i_W1` | Layer `i` FFN Expansion | `(embed_dim, ff_dim)` | `float32` | $\pm \sqrt{6 / (\text{embed} + \text{ff\_dim})}$ |
| `block_i_b1` | Layer `i` FFN Bias 1 | `(ff_dim,)` | `float32` | Initialize to `0.0` |
| `block_i_W2` | Layer `i` FFN Contraction | `(ff_dim, embed_dim)` | `float32` | $\pm \sqrt{6 / (\text{ff\_dim} + \text{embed})}$ |
| `block_i_b2` | Layer `i` FFN Bias 2 | `(embed_dim,)` | `float32` | Initialize to `0.0` |
| `ln_final_gamma` | Output LN Scale | `(embed_dim,)` | `float32` | Initialize to `1.0` |
| `ln_final_beta` | Output LN Offset | `(embed_dim,)` | `float32` | Initialize to `0.0` |
| `W_out` | Output Projection Head | `(embed_dim, vocab_size)` | `float32` | $\pm \sqrt{6 / (\text{embed} + \text{vocab})}$ |

---

## 2. Vocabulary & Metadata Schema

Each personality package requires a local dictionary file alongside it, or a stored structure inside the `.npz` mapping character-to-integer relationships:

### 2.1 Character Mappings
* **`char_to_idx`**: Map string characters to index tokens.
  ```json
  {
    "\n": 0,
    " ": 1,
    "!": 2,
    "a": 3,
    "b": 4,
    ...
  }
  ```
* **`idx_to_char`**: Map index tokens back to string characters for UI presentation.
  ```json
  {
    "0": "\n",
    "1": " ",
    "2": "!",
    "3": "a",
    "4": "b",
    ...
  }
  ```

---

## 3. Memory & Cache Schema

```
┌────────────────────────────────────────────────────────┐
│                   In-Memory Weight Cache               │
│                                                        │
│   "kafkaesque" ──► { W_embed: np.array, Wq: ... }      │
│   "camus"      ──► { W_embed: np.array, Wq: ... }      │
│   "gothic"     ──► { W_embed: np.array, Wq: ... }      │
└──────────────────────────┬─────────────────────────────┘
                           │ Active Swap
                           ▼
┌────────────────────────────────────────────────────────┐
│             Active Model Execution Scope               │
│                                                        │
│   self.token_embedding = active["W_embed"]            │
│   self.blocks[0].Wq    = active["block_0_Wq"]          │
│   ...                                                  │
└────────────────────────────────────────────────────────┘
```
- **Active State Pointer:** A single global reference stores which dictionary is currently bound. Switching personalities swaps the dictionary reference pointer, avoiding duplicate array allocation or system paging.
