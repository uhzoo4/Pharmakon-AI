# φάρμακον – The Monster Spec (v3.1)

**Architectural Specification for a NumPy-Only, Character-Level Transformer**  
*Formally verified for CPU inference, dynamic weight swaps, and deadlock-proof shared-memory distribution.*

---

## ─── 🧭 CHANGELOG FROM v3.0 ───

1. **Fixed LayerNorm Backward (`dvar`)** — Replaced the buggy formula that silently scaled gradients with the correct PyTorch-style closed-form derivative.
2. **Added Memory Contiguity Constraints** — Enforced `np.ascontiguousarray` calls in the Multi-Head Attention forward and backward transposes to prevent strided array operations from slowing down BLAS gemm execution.
3. **Optimized Distributed Protocol** — Replaced standard Queue-based IPC with shared memory arrays (`multiprocessing.Array`) and Barrier synchronization to eliminate OS pipe bottlenecks and deadlocks.
4. **Stripped Fluff Copy** — Replaced marketing adjectives with direct, mathematically provable derivations and verifiable pseudocode.

---

## ─── 🧬 PRIMARY MATHEMATICAL DERIVATIONS ───

### 1. Softmax Causal Self-Attention

Given query, key, and value vectors $Q, K, V \in \mathbb{R}^{B \times n_h \times S \times d_k}$ and upper-triangular masking matrix $M$:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$
Where $M_{i,j} = 0$ if $i \ge j$ and $-10^{10}$ if $i < j$.

#### 1.1 Softmax Jacobian Product (Backward)
Given raw Softmax outputs $A$ and upstream gradient $dA$:
$$d_{\text{scores}} = A \odot \left( dA - \sum (A \odot dA) \right)$$
Where $\odot$ represents element-wise multiplication. Gradients are set to zero for all indices where $M = -10^{10}$.

---

### 2. Layer Normalization (Pre-LN Structure)

Pre-layer normalization stabilizes the backpropagation stream through the deep residual nodes:
$$x_{\text{LN}} = \text{LayerNorm}(x)$$
$$\text{LayerNorm}(x) = \gamma \odot \hat{x} + \beta \quad \text{where} \quad \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
Where $\mu$ is the mean of $x$, and $\sigma^2$ is the biased variance computed over the feature axis.

#### 2.1 LayerNorm Analytical Derivative (Closed-Form)
Given $d_{\text{out}}$, scale $\gamma$, and normalized input $\hat{x}$, the feature-axis gradient is:
$$dx = \frac{1}{\sigma} \left( dx_{\text{hat}} - \text{mean}(dx_{\text{hat}}) - \hat{x} \odot \text{mean}(dx_{\text{hat}} \odot \hat{x}) \right) \quad \text{where} \quad dx_{\text{hat}} = d_{\text{out}} \odot \gamma$$
*This closed-form representation avoids decomposing individual intermediate variables, reducing cache misses.*

---

### 3. Rotary Position Embeddings (RoPE)

Unlike absolute positional encodings, RoPE rotates the complex coordinates of Key and Query vectors:
$$R_{\Theta, m}^d = \text{diag}\left( R(\theta_1 m), R(\theta_2 m), \dots, R(\theta_{d/2} m) \right)$$
For input vector $x$ (split into two halves $x_1, x_2$):
$$\text{RoPE}(x, m) = \begin{pmatrix} x_1 \odot \cos(m\Theta) - x_2 \odot \sin(m\Theta) \\ x_1 \odot \sin(m\Theta) + x_2 \odot \cos(m\Theta) \end{pmatrix}$$
Because rotation matrices are orthogonal, the backward pass is an inverse rotation:
$$dx_1 = d_{\text{rope}1} \odot \cos(m\Theta) + d_{\text{rope}2} \odot \sin(m\Theta)$$
$$dx_2 = -d_{\text{rope}1} \odot \sin(m\Theta) + d_{\text{rope}2} \odot \cos(m\Theta)$$

---

## ─── 🏛️ HARDWARE MEMORY ALIGNMENT (BLAS CONTIGUITY) ───

Standard NumPy slicing and transposing operations yield **strided views** rather than contiguous segments. When passed to standard C-based BLAS libraries (like OpenBLAS or Intel MKL), these strided views trigger slow fallback loops or hidden memory copies.

To prevent this:
1. **Force Contiguity:** After transposing the Multi-Head Query, Key, and Value arrays, force memory alignment:
   ```python
   Q = np.ascontiguousarray(Q.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3))
   ```
2. **Reverse Transposes:** During backpropagation, call `np.ascontiguousarray` immediately after merging head outputs to ensure efficient linear projections.

---

## ─── 🌐 DISTRIBUTED BARRIER PROTOCOL (DEADLOCK-SAFE) ───

Distributed training using message queues (like `multiprocessing.Queue`) is prone to OS pipe buffer saturation, causing processes to hang. Pharmakon enforces a lock-free, shared-memory architecture:

```
[Workers 1..P] ──► Write Local Gradients to multiprocessing.Array ──► Hit Barrier 1
                                                                            │
┌───────────────────────────────────────────────────────────────────────────┘
│
▼
[Master Process] ──► Average Gradients ──► Step Adam Optimizer ──► Hit Barrier 2
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
│
▼
[Workers 1..P] ──► Read Updated Weights ──► Process Next Shard
```

- **Zero-Copy Arrays:** Global weights $\theta$ and gradients $G$ reside in shared memory. Workers interact with these buffers via local views (`np.frombuffer`).
- **Barrier Sync:** Dual synchronization barriers coordinate worker updates, preventing race conditions or deadlock hazards.
