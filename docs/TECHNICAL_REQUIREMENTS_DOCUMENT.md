# Pharmakon – Technical Requirements Document (TRD)

**Version:** 1.2  
**Date:** 2026-07-05  
**Status:** Approved & Locked  
**Target Environment:** Localhost execution (Windows or Linux Mint Xfce)

---

## 1. System Architecture & Components

```
┌─────────────────────────────────┐
│     Frontend (Next.js 14+)      │
│  - Port 3000 (React / TS)       │
│  - Framer Motion UI Animations   │
│  - SSE Listener (EventSource)   │
└────────────────┬────────────────┘
                 │
                 │ HTTP POST & GET requests
                 ▼
┌─────────────────────────────────┐
│        Backend (FastAPI)        │
│  - Port 8000 (Python 3.11+)     │
│  - In-Memory weight caches      │
│  - SSE stream loop (uvicorn)    │
└────────────────┬────────────────┘
                 │
                 │ In-Memory Array References
                 ▼
┌─────────────────────────────────┐
│   NumPy Model (transformer.py)  │
│  - Pure NumPy (version 2.5.1)   │
│  - 41k Parameter Decoder        │
│  - RoPE Embeddings & Attention  │
└─────────────────────────────────┘
```

---

## 2. Technical Stack Specifications

### 2.1 Dependencies (`backend/requirements.txt`)
- **`numpy==2.5.1`**: Core tensor execution engine. Must be compiled against OpenBLAS.
- **`fastapi==0.136.1`**: API routing, validation, and SSE streaming.
- **`uvicorn[standard]==0.50.0`**: High-performance ASGI server.
- **`pydantic==2.13.0`**: JSON data parsing and schema validation.

### 2.2 Compilation Guidelines (CPU Speedups)
To achieve $\ge 5$ tokens/second on an 8GB machine, stock NumPy binaries must be replaced with custom binaries compiled against optimized Basic Linear Algebra Subprograms (BLAS).

#### Compilation steps on Linux:
```bash
# 1. Install OpenBLAS headers
sudo apt install build-essential libopenblas-dev

# 2. Force pip to build NumPy from source
pip install numpy==2.5.1 --no-binary numpy
```
This forces NumPy's setup script to detect `libopenblas` and compile C extensions with native SIMD vectorized instructions (AVX2/AVX-512), boosting matrix multiplications ($Q K^T$) by up to $10\text{x}$ to $50\text{x}$.

---

## 3. Core Engine Mechanics

### 3.1 In-Memory Weight Swapping
Model parameters (~41k float32 parameters, occupying ~166KB of storage) are saved as compressed `.npz` files in `backend/weights/`.
* On backend initialization, `WeightManager` reads all `.npz` files and expands them into a memory-resident dictionary:
  ```python
  self.personalities = {
      "kafkaesque": {"Wq": np.array(...), "Wk": np.array(...), ...},
      "camus_stranger": {"Wq": np.array(...), "Wk": np.array(...), ...},
      ...
  }
  ```
* When a user changes the personality in the UI, the backend **does not reload files**. It performs an in-memory pointer swap on the model's weight reference. This ensures zero latency ($<1\text{ ms}$) during swaps.

### 3.2 SSE Generation Loop
On receiving a generation request, the backend:
1. Spawns an asynchronous generator using FastAPI's `StreamingResponse`.
2. Converts the prompt text into character index integers using the loaded personality's `char_to_idx` map.
3. Feeds the indices through the model's forward pass to calculate the next-token logits.
4. Samples a token index via the `Sampler` (applying temperature and blacklist).
5. Appends the new index to the prompt window, decodes it back to a character, and yields it as a JSON payload over SSE.
6. The connection is kept alive until `max_tokens` are yielded or the model emits an end-of-text marker.

---

## 4. API Endpoints Schema

### 4.1 `GET /api/personalities`
Returns a list of all currently loaded personality keys.
* **Response Content-Type:** `application/json`
* **Response Body:**
```json
{
  "personalities": ["kafkaesque", "camus_stranger", "dark_romance"]
}
```

### 4.2 `POST /api/generate`
Initiates a text generation stream.
* **Request Content-Type:** `application/json`
* **Request Body Schema:**
```json
{
  "personality": "kafkaesque",
  "prompt": "One morning, when Gregor Samsa woke ",
  "temperature": 0.8,
  "max_tokens": 200,
  "blacklist": [5, 12, 19]
}
```
* **Response Content-Type:** `text/event-stream`
* **SSE Event Payload Schema:**
```json
data: {"text": "u"}

data: {"text": "n"}

data: {"text": "e"}
```
* **Connection Closing Event:**
```json
data: {"done": true}
```

---

## 5. Security & Stability Constraints
- **CORS Protection:** FastAPI must configure CORS rules to restrict connections to the localhost domains (`http://localhost:3000` and `http://127.0.0.1:3000`).
- **Input Boundaries:**
  - Prompt length is capped at $500$ characters.
  - Temperature is bound between $0.1$ and $2.0$.
  - Maximum tokens per request is capped at $500$ to prevent CPU exhaustion.
- **In-Memory Rate Limiter:** An IP-based sliding window rate limiter limits clients to $1$ active generation request at any given time.
- **Process Memory Cap:** A memory checker routine monitors the Python heap. If the backend process exceeds $300\text{ MB}$, garbage collection is forced, and inactive weight references are purged.
