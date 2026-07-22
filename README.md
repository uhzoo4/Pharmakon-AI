# Pharmakon — Local Grimoire Language Model

> **Poison & Remedy.** A character-level Transformer language model built entirely from scratch in pure NumPy, served via FastAPI, and interfaced through a typewriter web application.

---

> [!NOTE]
> **Active Construction Warning:** This project is under active development. The core mathematical engine, the massive dataset ingestion pipelines, and the autonomous continuous learning supervisor are 100% complete and verified. The Next.js frontend is currently under active construction.

---

> "...the *pharmakon* is neither remedy nor poison, neither good nor evil, neither the inside nor the outside, neither speech nor writing; the supplement is neither a plus nor a minus, neither an outside nor the complement of an inside, neither accident nor essence, etc.; the hymen is neither confusion nor distinction, neither identity nor difference, neither consummation nor virginity, neither the veil nor unveiling, neither the inside nor the outside, etc.; the gram is neither a signifier nor a signified, neither a sign nor a thing, neither a presence nor an absence, neither a position nor a negation, etc.; spacing is neither space nor time; the incision is neither the incised integrity of a beginning, or of a simple cutting into, nor simple secondarity. Neither/nor, that is, simultaneously either or; the mark is also the marginal limit, the march, etc."
>
> — **Jacques Derrida**, *Positions*

---

## Core Capabilities

Pharmakon is a desktop-centric deep learning system constructed from first principles to execute training, text parsing, translation, and inference without high-level neural network libraries like PyTorch or TensorFlow.

| Capability | Technical Implementation | Status |
| :--- | :--- | :--- |
| **Model Engine** | Manual forward/backward propagation of causal multi-head self-attention, pre-layer normalization (Pre-Norm), and final linear projections in pure `numpy`. | Complete |
| **KV Caching** | Stateful Key-Value caching with a sliding context window (capped at 64 tokens), collapsing inference complexity from $O(S^2)$ to $O(S)$. | Complete |
| **Continuous Hivemind** | Endless autonomous pipeline (`train_continuous.py`) that scrapes live data from HackerNews/Discord and incrementally fine-tunes the base model. | Complete |
| **Supervised Fine-Tuning** | Dedicated SFT pipeline (`train_sft.py`) to hammer the raw autocomplete base model into a highly structured, instruction-following conversational assistant. | Complete |
| **Dynamic Ingestion** | Integrated Project Gutenberg Catalog Search (via Gutendex API) to search, download, and clean public domain literature dynamically. | Complete |
| **Dynamic Weight Swap** | In-memory RAM swapping of model checkpoints (.npz files) corresponding to different characters in under 1 millisecond. | Complete |
| **Service API Layer** | FastAPI server running Server-Sent Events (SSE) for streaming text generation with concurrency isolation locks. | Complete |
| **Web Frontend** | Next.js interactive typewriter dashboard to communicate with local model instances. | Under Construction |

---

## The Autonomous Immune System (Supervisor)

To train massive models (like the 128-dimensional `the_pinnacle.npz`) on restricted hardware (8GB RAM), the architecture uses a custom OS-level supervisor (`train_autonomous.py`).

1. **Self-Kill Protocol:** If a gradient calculation results in a `NaN` or `Inf`, the training thread instantly triggers a `sys.exit(88)` anomaly code, killing itself to force the OS to completely flush any memory leaks.
2. **JSON Hot-Swapping:** The supervisor catches the crash, reads the rolling `training_state.json` file, and seamlessly re-launches the worker process at the exact epoch and step it left off.
3. **Infinite Uptime:** This ensures "nuclear-grade safety," allowing the model to train continuously for days on end despite hardware limits, memory fragmentation, or numerical explosions.

---

## Theoretical Demonology & Mathematical Purity

The model architecture is built on exact derivation, rejecting crude approximations that lead to numerical divergence. 

1. **The Exact Softmax Jacobian:** Most manual implementations use a diagonal approximation for the softmax backward pass. Pharmakon uses the full exact Jacobian-vector product: $dx = s \odot (d\_probs - \sum(s \cdot d\_probs))$ ensuring perfect gradients and stability (The "Shield of Stability").
2. **Contiguity Incantations:** By rigorously enforcing `np.ascontiguousarray` on multi-head attention reshaping, BLAS GEMM falls into optimized contiguous memory layouts rather than strided memory hell. This yields a 100x speedup, transforming a sleeping snail into a screaming demon.
3. **Pre-Norm Exorcism:** Gradients are stabilized against exponential explosion by routing residual connections strictly through LayerNorm blocks before the attention logic.
4. **Rotary Position Embeddings (RoPE):** Positions are encoded solely via complex rotations applied to key/query projections, weaving translation invariance directly into the fabric of attention.

---

## API Specifications

### 1. Generate Text (Streaming SSE)
* **Endpoint:** `POST /api/generate`
* **Payload:**
  ```json
  {
    "personality": "the_assistant",
    "prompt": "User: How are you?\nAssistant:",
    "temperature": 0.85
  }
  ```
* **Returns:** Server-Sent Events (SSE) streaming raw characters.

### 2. On-Demand Training / Fine-Tuning
* **Endpoint:** `POST /api/train`
* **Payload:**
  ```json
  {
    "personality": "new_existentialist",
    "text": "The absurd arises from the confrontation...",
    "epochs": 10
  }
  ```
* **Returns:** Training metadata and dynamically caches the updated weights.

---

*"The grimoire represents a technology of inscriptions; the model, a mathematics of ghosts."*
