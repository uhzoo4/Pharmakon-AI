# Pharmakon – System DNA & Architectural Blueprint

**Version:** 1.1  
**Status:** Verified & Locked  
**Language Constraints:** Pure NumPy (`numpy==2.5.1`), strict no-framework code.

---

## 1. Mathematical Rules & Primitives

All layers, attention models, and utility functions must be written from scratch using NumPy. 

### Primitives & Hidden Shapes

| Layer / Operation | Input Shape | Weights / Parameters | Output Shape | Equation / Description |
| --- | --- | --- | --- | --- |
| **Token Embedding** | `(seq_len,)` | `W_embed` of shape `(V, embed_dim)` | `(seq_len, embed_dim)` | Indexing: `x = W_embed[idx]` |
| **LayerNorm** | `(seq_len, dim)` | `gamma` of shape `(dim,)`, `beta` of shape `(dim,)` | `(seq_len, dim)` | `(x - mean) / sqrt(var + eps) * gamma + beta` |
| **Rotary Embedding (RoPE)** | `(n_heads, seq_len, head_dim)` | Cosine and Sine frequencies (precomputed) | `(n_heads, seq_len, head_dim)` | Element-wise rotation of complex pairs |
| **Causal Mask** | `(seq_len, seq_len)` | None | `(seq_len, seq_len)` | `scores + Mask` where upper triangle is `-1e10` |
| **Softmax** | `(..., V)` | None | `(..., V)` | `exp(x - max) / sum(exp(x - max))` |
| **Dense (Linear)** | `(seq_len, in_dim)` | `W` of shape `(in_dim, out_dim)`, optionally bias `b` | `(seq_len, out_dim)` | `x @ W + b` |

---

## 2. Attention Block Design

The attention mechanism implements **Multi-Head Self-Attention with Causal Masking and RoPE**:

1. **Pre-Norm:** LayerNorm is applied to incoming hidden state $x$ before any projection.
2. **Q, K, V Projections:** Linear projection of normalized hidden state into Query, Key, and Value arrays.
3. **Split Heads:** Reshape and transpose to `(num_heads, seq_len, head_dim)`.
4. **RoPE Injection:** Query and Key heads are rotated using precomputed sinusoid tables.
5. **Causal Attention Scores:** Scaled dot-product query-key multiplication, causal mask summation, and stable softmax.
6. **Value Aggregation & Out Projection:** Softmax weights multiplied with Value vectors, heads combined and projected via output weight $W_o$.
7. **Residual Merge:** Sum output back into the un-normalized input hidden state (Pre-Norm architecture).

---

## 3. Sparse Attention / Sampling: 1.5-Entmax

To enable sparse probability distributions during sampling, Pharmakon implements the **1.5-Entmax** operator:

### 3.1 Forward Pass (Projection onto Simplex)
$$\text{entmax}_{1.5}(z) = [z/2 - \tau]_+^2$$
Where $\tau$ is a threshold scalar solved numerically (e.g. using bisection or Newton's method) such that:
$$\sum_{i=1}^V [z_i/2 - \tau]_+^2 = 1.0$$
Here, $[x]_+ = \max(x, 0)$ is the thresholding operator that enforces sparsity by mapping low-scoring logits exactly to $0.0$.

### 3.2 Backward Pass (Jacobian-Vector Product)
Given upstream gradient vector $d$ and output probability vector $p$:
$$\frac{\partial L}{\partial z_i} = p_i \cdot \left( d_i - \frac{\sum_{j \in \mathcal{A}} p_j \cdot d_j}{\sum_{j \in \mathcal{A}} p_j} \right)$$
Where $\mathcal{A} = \{j \mid p_j > 0\}$ is the active support set containing coordinates that received non-zero probability in the forward pass.

---

## 4. Training Backpropagation (Reference)

For backpropagation (when training personalities), the following analytical gradients are implemented:

### 4.1 Softmax Cross-Entropy Loss
$$\frac{\partial L}{\partial z_{i}} = p_{i} - y_{i}$$
Where $p$ is the softmax probabilities vector, and $y$ is the target index one-hot representation.

### 4.2 Full Softmax Jacobian Product
Given upstream gradient $d_{probs}$:
$$d_{scores} = \text{probs} \odot \left( d_{probs} - \sum \left( \text{probs} \odot d_{probs} \right) \right)$$

### 4.3 Layer Normalization Gradient
Let $D$ be the normalization dimension (`embed_dim`):
$$dx_{\text{hat}} = dout \odot \gamma$$
$$dx = \frac{1}{D \cdot \sqrt{\text{var} + \epsilon}} \cdot \left( D \cdot dx_{\text{hat}} - \sum dx_{\text{hat}} - x_{\text{hat}} \odot \sum \left( dx_{\text{hat}} \odot x_{\text{hat}} \right) \right)$$
*(Summations are over the feature dimension axis=-1)*

---

## 5. Optimizer & Learning Rate Schedule

### 5.1 AdamW Optimizer with Weight Decay
Parameters $\theta$ are updated using gradient moments $m_t$ (first moment) and $v_t$ (second moment) with decoupled weight decay $\lambda$:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

### 5.2 Cosine Decay with Learning Rate Warmup
The step learning rate $\eta_t$ alternates based on training phases:
- **Warmup Phase** ($t < T_{\text{warmup}}$):
  $$\eta_t = \eta_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}}$$
- **Decay Phase** ($T_{\text{warmup}} \le t < T_{\text{total}}$):
  $$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left( 1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \pi\right) \right)$$

---

## 6. Operation Integrity Checks
- **Initialization:** Xavier uniform distributions with bounds calculated using input/output sizes.
- **Precision:** Float32 for weight matrices, Float64 for accumulation where numerical stability is a bottleneck.
- **Verification:** Backpropagation logic must pass a finite-difference gradient check before any corpus training is executed.
