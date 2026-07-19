# 🔮 φάρμακον (Pharmakon) — Local Grimoire Language Model

> **Poison & Remedy.** A character-level Transformer language model built entirely from scratch in pure NumPy, served via FastAPI, and interfaced through a gorgeous Dark Academia typewriter web application.

---

> “...the *pharmakon* is neither remedy nor poison, neither good nor evil, neither the inside nor the outside, neither speech nor writing; the supplement is neither a plus nor a minus, neither an outside nor the complement of an inside, neither accident nor essence, etc.; the hymen is neither confusion nor distinction, neither identity nor difference, neither consummation nor virginity, neither the veil nor unveiling, neither the inside nor the outside, etc.; the gram is neither a signifier nor a signified, neither a sign nor a thing, neither a presence nor an absence, neither a position nor a negation, etc.; spacing is neither space nor time; the incision is neither the incised integrity of a beginning, or of a simple cutting into, nor simple secondarity. Neither/nor, that is, simultaneously either or; the mark is also the marginal limit, the march, etc.”
>
> — **Jacques Derrida**, *Positions*

---

## ─── 🔮 Core Architecture ───

Pharmakon is an offline-capable, desktop-centric AI project designed to showcase deep learning implementation from absolute first principles.

1. **NumPy Model Engine:** Complete manual forward and backward implementation of causal multi-head self-attention, Rotary Positional Embeddings (RoPE), pre-layer normalization (Pre-Norm), and final linear projections—built entirely in pure `numpy==2.5.1` with no PyTorch or external dependencies. Features stateful KV Caching and an optimized inference pipeline with sliding context truncation (capped at 64 tokens), delivering a ~6.1x generation speedup.
2. **Dynamic Weight Swap:** High-speed in-memory swapping of `.npz` parameter files corresponding to distinct literary personalities (e.g., Kafkaesque existential dread, Camus Absurdist, or Gothic Dark Romance) in under 1 millisecond.
3. **Live Training API (`POST /api/train`):** On-demand, serialized, and thread-safe backpropagation engine. Users can send text payloads to fine-tune existing models or initialize and bootstrap entirely new personalities dynamically.
4. **Mathematical Corpus Cleaner (`clean_corpus.py`):** Normalizes raw text sequences using Unicode Decomposition (NFKD), strips combining diacritical marks (accents), and normalizes smart punctuation to ensure a strict 97-character printable ASCII vocabulary.

---

## ─── ⚡ Performance & Isolation Engineering ───

1. **Stateful KV Caching:** Inference complexity is reduced from $O(S^2)$ to $O(S)$ by feeding prompt tokens sequentially to construct stateful key-value cache tables, and then doing single-query cached lookups during generation.
2. **Sliding Context Window:** The Key/Value caches are automatically truncated to the last 63 tokens before new tokens are appended. This strictly limits the attention context length to the 64-token sequence training window, preventing memory bloat and preserving correct relative position alignments.
3. **Request Isolation:** Each FastAPI SSE request instantiates a local copy of the model instance dynamically (taking microseconds) rather than using a global singleton. This guarantees thread-safety and prevents concurrent requests from corrupting weight parameters mid-stream.
4. **Permanent Regression Tests:** A test suite under `tests/test_kv_cache_correctness.py` enforces logits equivalence ($\le 10^{-10}$ diff) and perplexity constraints ($<0.2$ deviation) up to 500 tokens.

---

## ─── 📁 Directory Layout ───

```
pharmakon/
├── backend/                    # CPU Inference & Training Engine
│   ├── requirements.txt        # Backend dependencies (FastAPI, NumPy, Pydantic)
│   ├── transformer.py          # Custom Transformer class with RoPE & Causal Masking
│   ├── train.py                # AdamW optimizer, Cosine Decay, and backprop loop
│   ├── generate.py             # Logits Sampler with temperature & blacklists
│   ├── weight_manager.py       # In-memory RAM cache & atomic checkpoint persistence
│   ├── main.py                 # FastAPI uvicorn application server
│   └── weights/                # Compressed .npz weights for trained personalities
├── data/                       # Training Corpora Directory
│   ├── README.md               # Dataset formatting and cleaning guidelines
│   └── *.txt                   # Raw and cleaned text files (ignored by Git)
├── docs/                       # Project Documentation & Technical Specs
│   ├── CORPUS_CLEANING_SPECIFICATION.md  # Math specs for Unicode normalization
│   ├── SYSTEM_DNA.md           # Mathematical rules, shapes, and backward equations
│   ├── TECHNICAL_REQUIREMENTS_DOCUMENT.md
│   └── PRODUCT_REQUIREMENTS_DOCUMENT.md
├── clean_corpus.py             # Standalone Python dataset cleaning utility
├── train_personalities.py      # Batch training pipeline for all corpora
├── pyrightconfig.json          # Pyright/Pylance editor type-checker configuration
└── README.md                   # [This file] Setup & Startup guide
```

---

## ─── ⚡ Quick Start ───

### 1. Prerequisites
* **Python:** 3.10+ (Tested on Python 3.13)
* **Node.js:** 18+ (for upcoming frontend web application)

### 2. Environment Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   cd "pharmakon AI"
   ```
2. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

### 3. Clean and Train Personalities
Before running the server, clean your raw text data and train the personalities. We have bootstrapped the project with three sample datasets in `data/`:
1. **Clean raw datasets:**
   ```bash
   python clean_corpus.py
   ```
2. **Train model checkpoints:**
   ```bash
   python train_personalities.py
   ```
   *This will train the custom transformer on your CPU and write compressed `.npz` parameter checkpoints to `backend/weights/`.*

### 4. Start the FastAPI API Server
1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Start the Uvicorn application server:
   ```bash
   python -m uvicorn main:app --port 8000 --reload
   ```
3. Open your browser and navigate to:
   * **Swagger API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Test endpoints interactively)
   * **Personalities Endpoint:** [http://127.0.0.1:8000/api/personalities](http://127.0.0.1:8000/api/personalities) (List loaded models)

---

## ─── 🔮 API Specifications ───

### 1. Generate Text (Streaming SSE)
* **Endpoint:** `POST /api/generate`
* **Content-Type:** `application/json`
* **Payload:**
  ```json
  {
    "personality": "kafkaesque",
    "prompt": "I stood before the large wooden gate...",
    "temperature": 0.85,
    "max_tokens": 150,
    "blacklist": []
  }
  ```
* **Returns:** Server-Sent Events (SSE) streaming raw characters: `data: {"text": "h"}` ... `data: {"done": true}`

### 2. On-Demand Training / Fine-Tuning
* **Endpoint:** `POST /api/train`
* **Content-Type:** `application/json`
* **Payload:**
  ```json
  {
    "personality": "new_existentialist",
    "text": "The absurd arises from the confrontation between the human need for meaning and the cold, silent universe...",
    "epochs": 10,
    "batch_size": 16,
    "lr": 0.0003
  }
  ```
* **Returns:**
  ```json
  {
    "success": true,
    "training_id": "8b525f24-2c4d-44a6-9ea3-4cf15de11f0a",
    "personality": "new_existentialist",
    "created": true,
    "epochs": 10,
    "batch_size": 16,
    "learning_rate": 0.0003,
    "training_tokens": 112,
    "training_time_seconds": 0.85
  }
  ```
  *Note: A global lock ensures training runs are safely serialized. The freshly trained weights are atomically written to disk and cached in RAM for immediate text generation.*

---

## ─── 🔮 Mathematical Foundations ───

### 1. Causal Attention Output
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
Where $M$ is the causal mask (upper triangular elements set to $-\infty$).

### 2. Rotary Position Embeddings (RoPE)
Positions are encoded solely via complex rotations applied to key/query projections:
$$R_{\Theta, m}^d = \text{diag}\left(R_{\theta_1, m}, R_{\theta_2, m}, \dots, R_{\theta_{d/2}, m}\right)$$
This embeds relative positioning directly into the dot-product attention calculation.

### 3. Shannon Entropy of the Corpus
Used by the cleaning engine to profile data density:
$$H(C) = - \sum_{v \in \mathcal{V}} P(v) \log_2 P(v)$$
   
---

*“The grimoire represents a technology of inscriptions; the model, a mathematics of ghosts.”*
