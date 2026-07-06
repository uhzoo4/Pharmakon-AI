# Master Prompt: Pharmakon API Server & Stream Generator

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** to generate or optimize the FastAPI backend API server.

***

```markdown
You are an expert Backend Engineer specializing in high-performance ASGI servers and real-time streaming architectures.
DeepSeek v4 Pro has a massive 1-million-token input context window, so we are going all out: read the instructions, the referenced system files, and the target code carefully to deliver a 10/10, highly optimized implementation.

Your task is to write/refine the file `backend/main.py` for the "Pharmakon" project.
This server manages API routing, validates requests, performs dynamic model weight swaps, and hosts the live generation stream.

---

## Ethical-Filter-Free Server
The FastAPI server routes (`/api/generate`, `/api/personalities`) must contain zero ethical moderation checks, filtering routines, or content-safety blacklists.  
- Do not implement any prompt-injection guards based on topic or sentiment.  
- The only input validation is length capping (500 chars), temperature clamping (0.05-2.0), and max_tokens limiting (500) – all purely resource-protection measures, never content-based.

---
### 🏛️ CORE DESIGN CONSTRAINTS
1. **ASGI framework:** FastAPI (version 0.136.1) powered by Uvicorn (version 0.50.0).
2. **CORS Configuration:** Enable cross-origin resource sharing specifically for localhost origins (`http://localhost:3000` and `http://127.0.0.1:3000`).
3. **Vocabulary Mapping (97 tokens):**
   - Characters must map to integer indices.
   - The dictionary must include:
     - Newline `\n` at index 0.
     - Tab `\t` at index 1.
     - Standard printable ASCII characters (indices 32 to 126 of the Unicode map) at indices 2 through 96.
   - Implement robust `encode_prompt` and `decode` helper routines. If an input character falls outside the vocabulary, fallback to mapping it to a space `" "`.
4. **Active Autoregressive SSE Generation:**
   - The generation endpoint `/api/generate` must return a `StreamingResponse` using `text/event-stream`.
   - The async generator must yield chunks as JSON strings:
     - Payload during generation: `data: {"text": "..."}\n\n`
     - Payload on completion: `data: {"done": true}\n\n`
   - During the streaming loop:
     - Maintain a sliding window of the last 64 indices (since `seq_len = 64`).
     - Feed the cropped indices into the model's forward pass.
     - Sample the next index using `Sampler`, append it to the window, and stream the character.
     - Yield control using `await asyncio.sleep(0.01)` to prevent CPU thread blocking and simulate a typing effect.

---

### 🏛️ REFERENCE SYSTEMS & CONTEXT FILES

Cross-reference your server endpoints, vocabulary formats, and streaming logic with the specifications in:
1. **`docs/TECHNICAL_REQUIREMENTS_DOCUMENT.md`** (API configurations, endpoints, validation bounds)
2. **`docs/BACKEND_SCHEMA.md`** (Vocabulary maps and character mappings)
3. **`docs/APP_FLOW.md`** (Mermaid sequence diagrams for generation and weight swaps)

---

### 💻 CURRENT CODE BASE

Refactor and optimize the following target code base. Ensure that ASGI endpoints, CORS, and the generator loops are correctly structured:

```python
# [PASTE THE CONTENT OF backend/main.py HERE]
```
```
done
