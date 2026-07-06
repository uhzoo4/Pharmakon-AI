# Master Prompt: Pharmakon Last Resort / Next Steps

This document details the advanced, post-MVP features that can be added to the Pharmakon engine to optimize execution speed, memory usage, and optimization convergence. It includes a copy-pasteable master prompt to implement these changes.

---

## 1. Post-MVP Feature Roadmap

### 1.1 In-Memory KV Caching (Inference Speedups)
- **Problem:** Currently, at each autoregressive step, the model re-evaluates queries, keys, and values for the entire history window ($S$ tokens). This leads to an $O(S^2)$ complexity curve, slowing down generation speed as the context window grows.
- **Remedy:** Implement in-memory KV caches. During inference, store keys ($K$) and values ($V$) from previous tokens. At each step, only evaluate the new token ($S=1$), compute its Query, append its Key and Value to the cache, and calculate attention against the accumulated cache. This collapses complexity to $O(S)$, ensuring consistent generation speed ($>20\text{ tokens/second}$).

### 1.2 NumPy FlashAttention (Memory Bandwidth Reductions)
- **Problem:** The intermediate attention matrix $Q K^T \in \mathbb{R}^{B \times H \times S \times S}$ is written to main RAM and read back during softmax, triggering massive memory bandwidth overhead on CPUs.
- **Remedy:** Implement a block-by-block FlashAttention approximation in NumPy. By partitioning the Query, Key, and Value arrays into blocks that fit within the CPU's L2/L3 cache, we compute the softmax scaling updates incrementally, avoiding the creation of the full $S \times S$ matrix in main memory.

### 1.3 Gradient Checkpointing (Training Memory Capping)
- **Problem:** Storing all activation caches for backpropagation across multiple layers scales memory linearly with the number of blocks ($O(L)$). This causes out-of-memory (OOM) faults on systems with limited RAM.
- **Remedy:** Implement gradient checkpointing. Only store the inputs to each block. During the backward pass, recompute the block's internal activations on the fly, reducing memory scaling from $O(L)$ to $O(\sqrt{L})$.

### 1.4 AdamW Optimizer with Weight Decay & Cosine Schedulers
- **Problem:** Standard Adam does not decouple weight decay from gradient updates, causing weights to grow uncontrollably on long text corpora.
- **Remedy:** Implement AdamW with L2 regularization directly on the parameter update step, coupled with a cosine learning rate decay schedule to guarantee training convergence.

---

## 2. Copy-Paste Master Prompt for Advanced Features

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** when you are ready to implement these optimizations.

***

```markdown
You are an elite AI Compiler Engineer specializing in high-performance Tensor architectures and memory-efficient deep learning optimizers.
DeepSeek v4 Pro has a massive 1-million-token input context window, so we are going all out: read the instructions, the referenced system files, and the target code carefully to deliver a 10/10, highly optimized implementation.

Your task is to refactor the "Pharmakon" project codebase to incorporate advanced training and inference speedups.

---

### 🏛️ REFACTORING SPECIFICATIONS

#### 1. Implement KV Cache in `backend/transformer.py`
- Modify the attention layer to support a stateful Key-Value cache.
- During forward pass, if `use_cache=True` is passed:
  - Retrieve the stored historical $K$ and $V$ matrices for this sequence.
  - Compute $Q, K, V$ only for the incoming single token.
  - Concatenate the new $K$ and $V$ to the cached history.
  - Perform attention using the single Query against the consolidated Key and Value cache.
  - Store the updated cache in the session context.

#### 2. Implement block-level FlashAttention in NumPy
- Implement the tiled attention forward pass.
- Compute the softmax normalization factors incrementally (online softmax) to avoid allocating the massive $S \times S$ scores matrix in RAM.

#### 3. Refactor `backend/transformer.py` for Gradient Checkpointing
- Add `checkpoint=True` parameter to the training loop.
- Instead of caching intermediate FFN and Attention activations for all layers during forward, only cache the input vector to each block.
- Recompute intermediate activations during the block's backward step.

#### 4. Create `backend/train.py` with AdamW & Cosine Decay
- Implement an AdamW optimization step:
  $$\theta_t \leftarrow \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
  Where $\lambda$ represents the decoupled weight decay factor.
- Configure a learning rate scheduler with warm-up steps followed by cosine learning rate decay.

Ensure all modifications are written in pure NumPy (version 2.5.1) and use 64-bit precision (`np.float64`).

---

### 🏛️ REFERENCE SYSTEMS & CONTEXT FILES

Cross-reference your optimization architectures and equations with the specifications in:
1. **`docs/MODEL_TRAINING_PLAN.md`** (DDP barrier structures and optimization formulas)

---

### 💻 CURRENT CODE BASE

Paste your current backend code arrays here to run refactoring passes:

```python
# [PASTE CURRENT backend/transformer.py CONTENT HERE]
```
```
