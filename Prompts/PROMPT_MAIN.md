# Master Prompt: Pharmakon API Server & Stream Generator

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** to generate or optimize the FastAPI backend API server.

***

```markdown
You are an expert Backend Engineer specializing in high-performance ASGI servers and real-time streaming architectures.
Your task is to write/refine the file `backend/main.py` for the "Pharmakon" project.
This server manages API routing, validates requests, performs dynamic model weight swaps, and hosts the live generation stream.

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

### 💻 IMPLEMENTATION BLUEPRINT

Write the file `backend/main.py` containing:
1. Pydantic request model: `GenerateRequest` checking limits for temperature (0.1 to 2.0) and max tokens (1 to 500).
2. Root setup and CORS middleware configurations.
3. `GET /api/personalities`: Return a list of loaded names from the weight manager.
4. `POST /api/generate`: The streaming route. Inside:
   - Perform an in-memory pointer swap on the global model instance to bind the requested personality weights.
   - Tokenize prompt, run the autoregressive generation loop, and stream character indices via SSE.
   - Capture inference crashes and stream them back to the client as error messages before completing.

Ensure all imports, libraries, and types are cleanly integrated.
```
