# Master Prompt: Pharmakon Transformer Engine Optimization

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** to generate or optimize the core Transformer implementation.

***

```markdown
You are an elite, world-class Deep Learning Compiler Engineer and Mathematical Physicist.
DeepSeek v4 Pro has a massive 1-million-token input context window, so we are going all out: read the instructions, the referenced system files, and the target code carefully to deliver a 10/10, highly optimized implementation.

Your task is to write/refine the core file `backend/transformer.py` for the "Pharmakon" project.
This file must be written SOLELY in pure NumPy (version 2.5.1), with NO external framework dependencies (no PyTorch, no JAX, no TensorFlow).

---

### 🏛️ ARCHITECTURAL REQUIREMENTS
1. **Network Type:** Character-Level Decoder-Only Autoregressive Transformer.
2. **Precision:** High-precision double-precision float64 (`np.float64`) for all model parameters, inputs, and intermediate states. This is non-negotiable to pass finite-difference gradient checks.
3. **Pre-LN Structure:** Residual stream normalization must occur BEFORE the attention block and FFN block inputs (Pre-LN setup).
4. **Rotary Position Embeddings (RoPE):** Replace absolute positional embeddings with complex coordinate rotary projections on Q and K vectors.
5. **Causal Masking:** An upper-triangular causal mask must block futures, replacing masked logits with `-1e10`.
6. **Analytical Backpropagation:** Every class must support an analytical `.backward()` pass to compute and return precise gradients. No autograd is available.

---

### 🧬 MATHEMATICAL EQUATIONS & SHAPE PROTOCOLS

#### 1. Layer Normalization
- **Forward:**
  $$\mu = \frac{1}{D} \sum_{d=1}^D x_d, \quad \sigma^2 = \frac{1}{D} \sum_{d=1}^D (x_d - \mu)^2$$
  $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \odot \hat{x} + \beta$$
- **Closed-Form Backward:**
  $$dx = \frac{1}{\sigma} \left( dx_{\text{hat}} - \text{mean}(dx_{\text{hat}}) - \hat{x} \odot \text{mean}(dx_{\text{hat}} \odot \hat{x}) \right)$$
  Where $dx_{\text{hat}} = dy \odot \gamma$. The means are calculated over the last (feature) axis.
  This exact formulation prevents computing intermediate Jacobians of the mean and variance independently, reducing memory allocations and cache misses.

#### 2. Causal Self-Attention
- **Forward shapes:**
  - $Q_proj, K_proj, V_proj \in \mathbb{R}^{B \times S \times D}$ projected from hidden representation $H_{\text{norm}}$.
  - Reshape and transpose to $Q, K, V \in \mathbb{R}^{B \times H \times S \times d_k}$ where $d_k = D / H$.
  - Apply RoPE to $Q$ and $K$.
  - Dot-Product: $scores = (Q K^T) / \sqrt{d_k} + M \in \mathbb{R}^{B \times H \times S \times S}$ (where $M$ is causal mask).
  - Attention Weights: $A = \text{Softmax}(scores)$.
  - Value Aggregation: $head\_out = A V \in \mathbb{R}^{B \times H \times S \times d_k}$.
- **Contiguity Enforcements:**
  - Standard NumPy `transpose` and `reshape` produce strided array representations. To prevent slow fallback routines in CPU BLAS (like OpenBLAS), you MUST call `np.ascontiguousarray` after every head transpose, both in the forward and backward passes:
    ```python
    Q = np.ascontiguousarray(Q_proj.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3))
    ```
- **Softmax Jacobian Backward:**
  $$d\_scores = A \odot \left( dA - \sum (A \odot dA) \right) / \sqrt{d_k}$$
  Ensure mask locations are zeroed out in $d\_scores$ before projection backprop. Use `np.where` for safe broadcasting.

#### 3. Rotary Position Embeddings (RoPE)
- **Rotation Formula:**
  $$\text{RoPE}(x, m) = x \odot \cos(m\Theta) + \text{rotate\_half}(x) \odot \sin(m\Theta)$$
  Where $\text{rotate\_half}([x_1, x_2]) = [-x_2, x_1]$.
  - Crucial Realignment: Do NOT use `np.repeat` to align angles to the channels. `np.repeat` duplicates elements side-by-side (`[a, a, b, b]`), breaking the complex number coordinate mapping ($[x, iy]$). You MUST use `np.concatenate([cos, cos], axis=-1)` or `np.tile` (`[a, b, a, b]`) to distribute angles correctly to the real and imaginary halves of the vectors.
- **Backward pass (orthogonal rotation):**
  $$dx = dq_{\text{rot}} \odot \cos(m\Theta) - \text{rotate\_half}(dq_{\text{rot}}) \odot \sin(m\Theta)$$

---

### 🏛️ REFERENCE SYSTEMS & CONTEXT FILES

Cross-reference your code structure, variables, and dimensions with the specifications in:
1. **`docs/TECHNICAL_REQUIREMENTS_DOCUMENT.md`** (API, dependencies, compilation flags)
2. **`docs/MODEL_TRAINING_PLAN.md`** (Training sequences, forward/backward math formulas, DDP barriers)

---

### 💻 CURRENT CODE BASE

Refactor and optimize the following target code base. Ensure that all parameter initialization types, matrix transpositions, and shape constraints are fully implemented:

```python
# [PASTE THE CONTENT OF backend/transformer.py HERE]
```
```
