# Pharmakon – Model Training Plan

This document specifies the training pipeline, optimization math, backpropagation gradients, and distributed multi-process training protocols for the Pharmakon Character-Level Transformer.

---

## 1. Preprocessing & Dataset Curation

### 1.1 Text Corpus Ingestion
- **Source Files:** Plain-text files (`.txt`) loaded via command-line arguments.
- **Deduplication & Sanitization:** Text is decoded as UTF-8, line endings normalized (`\r\n` to `\n`), and duplicate whitespaces cleaned.
- **Vocabulary Extraction:** 
  - Scan the entire text to build a sorted set of all unique characters.
  - Create bi-directional mappings:
    - `char_to_idx`: `Dict[str, int]` mapping characters to index integers.
    - `idx_to_char`: `Dict[int, str]` mapping index integers back to characters.
  - Let vocabulary size be $V$.

### 1.2 Sequence Dataset Formatting
- For training, we partition the dataset into sequence blocks of length `seq_len`.
- For any index $i$ in the tokenized list:
  - **Input sequence ($X$):** `data[i : i + seq_len]`
  - **Target sequence ($Y$):** `data[i + 1 : i + seq_len + 1]` (next-character prediction target).
- **Batching:** Randomly sample $B$ start indices to pack inputs and targets into shape `(B, seq_len)`.

---

## 2. Training Hyperparameters

| Hyperparameter | Value | Description |
| --- | --- | --- |
| `embed_dim` | 64 | Latent space vector size |
| `num_heads` | 4 | Multi-head attention branches |
| `head_dim` | 16 | Hidden size per head (`embed_dim // num_heads`) |
| `num_layers` | 2 | Number of stacked Transformer blocks |
| `ff_dim` | 128 | Expansion layer size in Feed-Forward network |
| `dropout` | 0.1 | Drop probability for attention and FFN outputs |
| `seq_len` | 64 | Length of input sequences |
| `batch_size` | 32 | Batch dimension size |
| `lr` | 3e-4 | Peak learning rate for Adam optimizer |
| `beta1` | 0.9 | Adam first moment decay rate |
| `beta2` | 0.999 | Adam second moment decay rate |
| `eps` | 1e-8 | Stability constant |
| `max_grad_norm` | 1.0 | Global gradient norm threshold |
| `epochs` | 50 | Total training iterations |

---

## 3. Mathematical Optimization & Backpropagation

The training loop requires manual evaluation of analytical gradients. The backward pass must flow through every layer in reverse order.

### 3.1 Softmax Cross-Entropy Loss
Given predicted logits $Z \in \mathbb{R}^{B \times S \times V}$ and integer targets $Y \in \mathbb{R}^{B \times S}$:
- **Softmax probabilities ($p$):**
  $$p_{b,s,v} = \frac{\exp(z_{b,s,v} - \max(z_{b,s}))}{\sum_{v'=1}^V \exp(z_{b,s,v'} - \max(z_{b,s}))}$$
- **Cross-Entropy Loss ($L$):**
  $$L = -\frac{1}{B \cdot S} \sum_{b=1}^B \sum_{s=1}^S \log(p_{b,s,y_{b,s}})$$
- **Logits Gradient ($dZ$):**
  $$dZ_{b,s,v} = \frac{p_{b,s,v} - \mathbb{I}(v = y_{b,s})}{B \cdot S}$$
  *(Subtract $1$ from the target indices, then divide by the total tokens $B \times S$)*

### 3.2 Output Projection
- **Forward:** $logits = H_{\text{norm}} W_{\text{proj}}$, where $H_{\text{norm}} \in \mathbb{R}^{(B \cdot S) \times d_{\text{model}}}$, $W_{\text{proj}} \in \mathbb{R}^{d_{\text{model}} \times V}$.
- **Gradients:**
  $$dW_{\text{proj}} = H_{\text{norm}}^T dZ_{\text{flat}}$$
  $$dH_{\text{norm}} = dZ_{\text{flat}} W_{\text{proj}}^T$$

### 3.3 Layer Normalization Backward (Closed Form)
Given incoming gradient $d_{\text{out}}$, normalized states $\hat{x}$, and standard deviation $\sigma$:
$$dx_{\text{hat}} = d_{\text{out}} \odot \gamma$$
$$dx = \frac{1}{\sigma} \left( dx_{\text{hat}} - \text{mean}(dx_{\text{hat}}) - \hat{x} \odot \text{mean}(dx_{\text{hat}} \odot \hat{x}) \right)$$
$$d\gamma = \sum (d_{\text{out}} \odot \hat{x}), \quad d\beta = \sum d_{\text{out}}$$
*(All means and sums are taken over the feature dimension axis=-1)*

### 3.4 Feed-Forward Network Backward
- **Forward:** $ff_1 = \max(0, H_{\text{norm}} W_1 + b_1)$, $ff_2 = ff_1 W_2 + b_2$.
- **Gradients:**
  $$dW_2 = ff_1^T dff_2, \quad db_2 = \sum dff_2 \text{ (axis=0)}$$
  $$dff_1 = dff_2 W_2^T \odot \mathbb{I}(ff_1 > 0) \quad \text{(ReLU backward)}$$
  $$dW_1 = H_{\text{norm}}^T dff_1, \quad db_1 = \sum dff_1 \text{ (axis=0)}$$
  $$dH_{\text{norm}} = dff_1 W_1^T$$

### 3.5 Multi-Head Attention (MHA) Backward
To maintain cache contiguity during backpropagation, memory layout must be managed explicitly:
1. **Output Projection:**
   $$dW_o = C_{\text{flat}}^T dH_{\text{att\_out\_flat}}$$
   $$dC = dH_{\text{att\_out}} W_o^T \quad \text{where } dC \in \mathbb{R}^{B \times S \times d_{\text{model}}}$$
2. **Reshape & Split:**
   Convert $dC$ back to multi-head shape:
   $$dH_{\text{head\_out}} = dC.\text{reshape}(B, S, n_{\text{heads}}, d_k).\text{transpose}(0, 2, 1, 3)$$
   Ensure array memory is contiguous:
   $$dH_{\text{head\_out}} = \text{np.ascontiguousarray}(dH_{\text{head\_out}})$$
3. **Value & Weight Projections:**
   $$dV = A^T dH_{\text{head\_out}} \quad \text{(Attention weights } A)$$
   $$dA = dH_{\text{head\_out}} V^T$$
4. **Softmax & Scale Backward:**
   Apply dropout mask backward, then flow through stable softmax Jacobian:
   $$d_{\text{scores}} = A \odot \left( dA - \sum (A \odot dA) \right)$$
   $$d_{\text{scores}}[\text{masked\_positions}] = 0$$
   $$d_{\text{scores}} /= \sqrt{d_k}$$
5. **Query & Key Gradients:**
   $$dQ_{\text{rope}} = d_{\text{scores}} K_{\text{rope}}$$
   $$dK_{\text{rope}} = d_{\text{scores}}^T Q_{\text{rope}}$$
6. **Reverse RoPE Rotation:**
   Gradients are spun in reverse to recover un-rotated projections:
   $$dQ_1 = dQ_{\text{rope}1} \odot \cos + dQ_{\text{rope}2} \odot \sin$$
   $$dQ_2 = -dQ_{\text{rope}1} \odot \sin + dQ_{\text{rope}2} \odot \cos$$
   $$dQ = \text{concatenate}(dQ_1, dQ_2, \text{axis}=-1)$$
   *(Same operation applies to $dK$)*
7. **Input Projections:**
   Merge heads and compute projection matrix updates:
   $$dW_q = H_{\text{norm}}^T dQ_{\text{flat}}, \quad dW_k = H_{\text{norm}}^T dK_{\text{flat}}, \quad dW_v = H_{\text{norm}}^T dV_{\text{flat}}$$
   $$dH_{\text{norm}} = dQ W_q^T + dK W_k^T + dV W_v^T$$

### 3.6 Global Gradient Clipping
To prevent gradient explosions during batch training:
1. Gather all calculated parameter gradients $\{G_i\}$ into a flat vector.
2. Calculate the global $L_2$ norm:
   $$\|G\|_2 = \sqrt{\sum_i \sum_j (G_{i,j})^2}$$
3. If $\|G\|_2 > \text{max\_grad\_norm}$ ($1.0$), scale all gradients:
   $$G_i \leftarrow G_i \cdot \frac{\text{max\_grad\_norm}}{\|G\|_2}$$

### 3.7 Adam Update Rules
For each parameter $\theta$ and its gradient $g$:
$$m \leftarrow \beta_1 m + (1 - \beta_1) g, \quad v \leftarrow \beta_2 v + (1 - \beta_2) g^2$$
$$\hat{m} = \frac{m}{1 - \beta_1^t}, \quad \hat{v} = \frac{v}{1 - \beta_2^t}$$
$$\theta \leftarrow \theta - \frac{\text{lr} \cdot \hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

---

## 4. Distributed Data-Parallel (DDP) Summoning

To train personalities faster without system network overhead, we use a custom, deadlock-proof data-parallel routine using only standard Python `multiprocessing`.

```
               ┌───────────────────────────────┐
               │    Shared Memory Segment      │
               │  - multiprocessing.Array      │
               │  - Global Model Weights       │
               │  - Global Gradients Buffer    │
               └──────────────┬────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│  Worker 1    │         │  Worker 2    │         │  Worker P    │
│  - Dataset 1 │         │  - Dataset 2 │         │  - Dataset P │
│  - Local grad│         │  - Local grad│         │  - Local grad│
└──────┬───────┘         └──────┬───────┘         └──────┬───────┘
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                ▼
                       [Barrier Synchronize]
                                │
                                ▼
                    Master averages gradients,
                    Adam step, update shared array.
                                │
                                ▼
                       [Barrier Release]
```

### 4.1 Shared Memory Segment & Buffers
- The global parameter arrays $\theta$ and their corresponding gradients $g$ are allocated in a shared memory region using `multiprocessing.Array(ctypes.c_float)`.
- Workers access these memory segments as standard NumPy views (`np.frombuffer`). This avoids data copying.

### 4.2 Synchronization Protocol
To prevent processes from deadlocking (a common failure mode with queues or pipes under high workloads):
1. **Sharded Data Processing:** $P$ worker processes are spawned. Each worker is assigned a disjoint shard of the text corpus.
2. **Lock-Free Local Accumulation:** Workers compute local forward and backward passes independently, writing local gradients to their private buffers.
3. **Array Consolidation & Barrier 1:** Once a step completes, workers copy their local gradients to their designated slots in the shared memory array. All processes hit `Barrier(P + 1)`.
4. **Master Aggregate & Step:** The master process wakes up, computes the mean gradient across the $P$ worker slots, clips the mean gradient, executes the Adam update step on the shared weights, and clears the shared gradient slot.
5. **Barrier 2 & Release:** The master and worker processes synchronize at `Barrier(P + 1)` releasing workers to copy the updated weights and proceed to the next batch.
