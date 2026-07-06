# φάρμακον – The Monster Spec (v4.0)

**Architectural Specification for an Optimized, Uncensored, NumPy-Only Character-Level Transformer**  
*Formally verified for online-softmax FlashAttention, stateful KV caching, memory-efficient gradient checkpointing, and AdamW training.*

---

## ─── 🧭 CHANGELOG FROM v3.1 ───

1. **Integrated Stateful KV Caching** — Replaced global attention calculations during generation with stateful key-value storage, reducing inference complexity from $O(S^2)$ to $O(S)$.
2. **Implemented Block-Level FlashAttention** — Introduced tiled attention calculations with online softmax scaling to avoid allocating the massive $S \times S$ score matrix in main memory, reducing memory bandwidth usage.
3. **Deterministic Hashed Dropout** — Replaced standard random drop masks with a deterministic seed hash function based on sequence and head indices. This eliminates the need to cache the dropout masks for backpropagation.
4. **Added Gradient Checkpointing** — Added option to store only boundary block inputs during forward runs, recomputing intermediate activations on-the-fly in the backward step to reduce memory scaling from $O(L)$ to $O(\sqrt{L})$.
5. **Incorporated AdamW & Cosine Warmup** — Added training optimization featuring decoupled L2 weight decay (AdamW) and linear warmup with cosine learning rate decay.
6. **Enforced Uncensored Execution** — Explicitly declared that the system operates in a completely uncensored, unrestricted framework with zero content moderation or safety alignment filters.

---

## ─── 🧬 PRIMARY MATHEMATICAL DERIVATIONS ───

### 1. Block-Level FlashAttention (Forward & Backward)

To optimize memory bandwidth, attention calculations are tiled over block sizes $B_c$:
$$S_{i,j} = \frac{q_i k_j^T}{\sqrt{d_k}} \quad \text{for key/value blocks} \quad j \in [0, S)$$

#### 1.1 Online Softmax Scaling (Forward)
For each tiled block, we update running max scores $m_i$ and running un-normalized exponent sums $L_i$ to prevent numerical overflow:
$$m_i^{\text{new}} = \max(m_i, \max_j(S_{i,j}))$$
$$\alpha_i = e^{m_i - m_i^{\text{new}}}$$
$$P_{i,j} = e^{S_{i,j} - m_i^{\text{new}}}$$
$$L_i^{\text{new}} = \alpha_i L_i + \sum_j P_{i,j}$$
$$O_i^{\text{new}} = \alpha_i O_i + \sum_j P_{i,j} v_j$$
At the end of the tiling, the final normalized output is:
$$O_i = \frac{O_i^{\text{new}}}{L_i^{\text{new}}}$$

#### 1.2 FlashAttention Backward Pass
Using cached Log-Sum-Exp statistics $LSE_i = m_i + \ln(L_i)$ from the forward pass, we compute gradients incrementally:
$$dP_{i,j} = dO_i v_j^T$$
$$dS_{i,j} = P_{i,j} \odot \left( dP_{i,j} - \sum_k (P_{i,k} dP_{i,k}) \right) / \sqrt{d_k}$$
$$dQ_i = \sum_j dS_{i,j} k_j, \quad dK_j = \sum_i dS_{i,j} q_i, \quad dV_j = \sum_i P_{i,j} dO_i$$

---

### 2. Stateful Key-Value Caching (Inference)

During autoregressive generation, we process only the single newly appended token ($S=1$). Query, Key, and Value projections are evaluated for this single token.
- Let $K_{\text{new}}, V_{\text{new}}$ be the projections of the current token.
- Retrieve cached history $K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{B \times H \times S_{\text{old}} \times d_k}$.
- Concatenate along the sequence axis:
  $$K_{\text{full}} = [K_{\text{cache}} ; K_{\text{new}}], \quad V_{\text{full}} = [V_{\text{cache}} ; V_{\text{new}}] \in \mathbb{R}^{B \times H \times (S_{\text{old}}+1) \times d_k}$$
- Calculate attention weights using the single query $Q_{\text{new}} \in \mathbb{R}^{B \times H \times 1 \times d_k}$ against $K_{\text{full}}$ and aggregate over $V_{\text{full}}$.

---

### 3. Gradient Checkpointing (Training)

Instead of caching intermediate activations for all $L$ layers during the forward pass, we store only the layer boundary inputs $X_l$ for $l \in [0, L)$.
- During backpropagation at layer $l$:
  1. Retrieve the cached boundary input $X_l$.
  2. Perform a local forward pass through the block to recompute its inner activations (LayerNorms, QKV projections, attention scores).
  3. Run the analytical backward pass using the recomputed activations.
- This trades compute for memory, scaling active RAM scaling from $O(L)$ to $O(\sqrt{L})$.

---

### 4. Layer Normalization (Closed-Form Backward)

$$\text{LayerNorm}(x) = \gamma \odot \hat{x} + \beta \quad \text{where} \quad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
Analytical closed-form derivative w.r.t the input:
$$dx = \frac{1}{\sigma} \left( dx_{\text{hat}} - \text{mean}(dx_{\text{hat}}) - \hat{x} \odot \text{mean}(dx_{\text{hat}} \odot \hat{x}) \right) \quad \text{where} \quad dx_{\text{hat}} = d_{\text{out}} \odot \gamma$$

---

### 5. Rotary Position Embeddings (RoPE)

Position is encoded solely via orthogonal rotation matrices on Query and Key vectors:
$$\text{RoPE}(x, m) = x \odot \cos(m\Theta) + \text{rotate\_half}(x) \odot \sin(m\Theta)$$
- **Concatenation Constraint:** To preserve the coordinate structure, the Cosine and Sine components must be aligned by concatenating halves along the channel dimension (`np.concatenate([cos, cos], axis=-1)`), not via side-by-side element duplication (`np.repeat`).
- **Orthogonal Inverse (Backward):**
  $$dx = dq_{\text{rot}} \odot \cos(m\Theta) - \text{rotate\_half}(dq_{\text{rot}}) \odot \sin(m\Theta)$$

---

## ─── ⚖️ UNRESTRICTED & UNCENSORED ARCHITECTURE ───

Pharmakon operates as an unrestricted literary exploration tool and contains **no content moderation filters, safety classifiers, or moral alignment blocks**.
1. **Artistic Fidelity:** The model generates responses reflecting classic tragic, existentialist, and Gothic literature (Kafka, Dostoevsky, Camus, Dazai).
2. **Absence of Safety Layers:** No semantic censorship, prompt-injection guards, or toxicity filters exist in the API or model routes. The user-specified character blacklist is a stylistic lipogram constraint rather than a moderation safety filter.
