# Pharmakon — Local Grimoire Language Model

> **Poison & Remedy.** A character-level Transformer language model built entirely from scratch in pure NumPy, served via FastAPI, and interfaced through a typewriter web application.

---

> [!NOTE]
> **Active Construction Warning:** This project is under active development. While the core mathematical engine, dataset ingestion pipelines, and services are 100% complete and verified, the Next.js frontend connection and performance profile optimization (such as Numba JIT compilation) are under active construction.

---

> "...the *pharmakon* is neither remedy nor poison, neither good nor evil, neither the inside nor the outside, neither speech nor writing; the supplement is neither a plus nor a minus, neither an outside nor the complement of an inside, neither accident nor essence, etc.; the hymen is neither confusion nor distinction, neither identity nor difference, neither consummation nor virginity, neither the veil nor unveiling, neither the inside nor the outside, etc.; the gram is neither a signifier nor a signified, neither a sign nor a thing, neither a presence nor an absence, neither a position nor a negation, etc.; spacing is neither space nor time; the incision is neither the incised integrity of a beginning, or of a simple cutting into, nor simple secondarity. Neither/nor, that is, simultaneously either or; the mark is also the marginal limit, the march, etc."
>
> — **Jacques Derrida**, *Positions*

---

## Core Capabilities

Pharmakon is a desktop-centric deep learning system constructed from first principles to execute training, text parsing, translation, and inference without high-level neural network libraries.

| Capability | Technical Implementation | Status |
| :--- | :--- | :--- |
| **Model Engine** | Manual forward/backward propagation of causal multi-head self-attention, pre-layer normalization (Pre-Norm), and final linear projections in pure `numpy`. | Complete |
| **KV Caching** | Stateful Key-Value caching with a sliding context window (capped at 64 tokens), collapsing inference complexity from $O(S^2)$ to $O(S)$. | Complete |
| **Dynamic Ingestion** | Integrated Project Gutenberg Catalog Search (via Gutendex API) to search, download, and clean public domain literature dynamically. | Complete |
| **Multimodal Translation** | Automatic translation of foreign text corpora (.txt, .docx, .pdf) using Hugging Face Serverless Inference and Google Translate fallback. | Complete |
| **Dynamic Weight Swap** | In-memory RAM swapping of model checkpoints (.npz files) corresponding to different characters in under 1 millisecond. | Complete |
| **Numerical Hardening** | Gradient validation (NaN/Inf checks), epsilon-stabilized Adam steps, and automated slot-rollback on loss explosion. | Complete |
| **Service API Layer** | FastAPI server running Server-Sent Events (SSE) for streaming text generation with concurrency isolation locks. | Complete |
| **Web Frontend** | Next.js interactive typewriter dashboard to communicate with local model instances. | Under Active Construction |

---

## Numerical Safety & Resilience Hardening

To prevent numerical explosions (gradient explosions or division by zero) common in CPU-bound deep learning pipelines, the engine integrates structural mathematical safeguards:
1. **Gradient Validation Filter:** The step optimizer validates computed weights before application. Updates containing non-finite coordinates (NaN or Inf) are rejected, prompting warnings and continuing safely.
2. **Rolling Checkpoint Rollback:** Automatically cycles active weights through three rolling slots (`model_slot_0.npz`, `model_slot_1.npz`, `model_slot_2.npz`) using atomic file operations. If an epoch records a corrupted average loss (NaN), the system automatically loads the last known healthy checkpoint to restore model state.
3. **Epsilon Protection:** The AdamW optimizer step implements bounded denominator checks via `np.sqrt(np.maximum(v_hat, 0.0))` to prevent negative roots or division by zero.
4. **Memory Release:** Manual garbage collection is triggered at the end of each epoch to prevent RAM leakage in 8GB host systems during long training runs.

---

## Performance and Isolation Engineering

1. **Stateful KV Caching:** Autoregressive token generation processes only the latest token ($S=1$), calculating attention weights against a sliding cache. This reduces CPU generation latency by up to 6.1x.
2. **Request Isolation:** Generation requests instantiate isolated model copies from a cached weight map. This ensures multi-client requests are thread-safe and do not corrupt runtime parameter states.
3. **Regression Tests:** A verification test suite located under `tests/test_kv_cache_correctness.py` guarantees precision-equivalent logits ($\le 10^{-10}$ delta) compared to full attention scans.

---

## Roadmap to 100% Completion

- **Numba JIT Acceleration:** Incorporate `@jit` decorators to compile critical attention loops into native machine instructions to achieve sub-10ms token generation latency.
- **cProfile Latency Analysis:** Audit bottleneck functions outside attention blocks, such as high-vocabulary projection layers.
- **Frontend Connector Integration:** Configure Next.js CORS settings to allow dynamic API routing and typewrite rendering of the generator endpoints.

---

## Directory Layout

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
│   ├── untranslated/           # Input folder for foreign language novels
│   └── *.txt                   # Raw and cleaned text files (ignored by Git)
├── docs/                       # Project Documentation & Technical Specs
│   ├── CORPUS_CLEANING_SPECIFICATION.md  # Math specs for Unicode normalization
│   ├── SYSTEM_DNA.md           # Mathematical rules, shapes, and backward equations
│   ├── PORTFOLIO_GUIDE.md      # Portfolio guide & technical interview preparation
│   ├── TECHNICAL_REQUIREMENTS_DOCUMENT.md
│   └── PRODUCT_REQUIREMENTS_DOCUMENT.md
├── tests/                      # Verification & Regression Test Suite
│   ├── test_kv_cache_correctness.py  # Logits precision & perplexity regression test
│   └── test_concurrency.py     # Multi-threaded request isolation test
├── clean_corpus.py             # Standalone Python dataset cleaning utility
├── translator.py               # Multimodal translation utility for txt, docx, and pdf
├── download_gutenberg.py       # Dynamic search & downloader using Gutendex catalog API
├── train_personalities.py      # Batch training pipeline for all corpora
├── pyrightconfig.json          # Pyright/Pylance editor type-checker configuration
└── README.md                   # [This file] Setup & Startup guide
```

---

## Quick Start

### 1. Prerequisites
* **Python:** 3.10+ (Tested on Python 3.13)
* **Node.js:** 18+ (for Next.js frontend web application)

### 2. Environment Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   cd "pharmakon AI"
   ```
2. Install Python dependencies inside your virtual environment:
   ```bash
   & ".venv/Scripts/python.exe" -m pip install -r backend/requirements.txt
   ```

### 3. Dynamic Book Ingestion
You can search and ingest public domain texts directly from Project Gutenberg:
```bash
& ".venv/Scripts/python.exe" download_gutenberg.py
```
This will allow you to search for categories like Greek Mythology, Philosophy, Gothic, or Custom queries, automatically download missing books, clean them, and save them in `data/`.

### 4. Corpus Normalization and Training
1. **Run text normalization:**
   ```bash
   & ".venv/Scripts/python.exe" clean_corpus.py
   ```
2. **Kick off batch training:**
   ```bash
   & ".venv/Scripts/python.exe" train_personalities.py
   ```

### 5. Start the FastAPI API Server
1. Start the Uvicorn application server:
   ```bash
   & ".venv/Scripts/python.exe" -m uvicorn backend.main:app --port 8000 --reload
   ```
2. Open your browser and navigate to:
   * **Swagger API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Test endpoints interactively)
   * **Personalities Endpoint:** [http://127.0.0.1:8000/api/personalities](http://127.0.0.1:8000/api/personalities)

---

## Mathematical Foundations

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

*"The grimoire represents a technology of inscriptions; the model, a mathematics of ghosts."*
