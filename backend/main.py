from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import numpy as np
import json


from weight_manager import WeightManager
from transformer import PharmakonTransformer
from generate import Sampler

app = FastAPI(title="φάρμακον (Pharmakon) Backend")

# -------------------------------------------------------------------
# CORS – restrict origins to localhost development frontends
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Vocabulary – 97 tokens: \n, \t, printable ASCII 32-126
# -------------------------------------------------------------------
VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
VOCAB_SIZE = len(VOCAB_CHARS)
char_to_idx = {char: idx for idx, char in enumerate(VOCAB_CHARS)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB_CHARS)}


def encode_prompt(prompt: str) -> list[int]:
    """Convert prompt string to character index integers.
    Unknown characters fall back to space (' ') index.
    """
    space_idx = char_to_idx[" "]
    return [char_to_idx.get(c, space_idx) for c in prompt]


# -------------------------------------------------------------------
# Weight management & model singleton
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
weights_dir = BASE_DIR / "weights"
weight_manager = WeightManager(weights_dir, vocab_size=VOCAB_SIZE)


model = PharmakonTransformer(vocab_size=VOCAB_SIZE)

# -------------------------------------------------------------------
# Simple IP‑based concurrency limiter (1 active generation per IP)
# -------------------------------------------------------------------
active_clients: set[str] = set()
active_clients_lock = asyncio.Lock()

# -------------------------------------------------------------------
# Pydantic schema – strict resource‑protection bounds only
# -------------------------------------------------------------------
class GenerateRequest(BaseModel):
    personality: str
    prompt: str = Field(..., max_length=500)
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    max_tokens: int = Field(default=200, ge=1, le=500)
    blacklist: list[int] = Field(default_factory=list)


@app.get("/api/personalities")
async def get_personalities():
    """Return a list of all loaded personality keys."""
    return {"personalities": weight_manager.list_personalities()}


@app.post("/api/generate")
async def generate_text(req: GenerateRequest, request: Request):
    """
    Autoregressive text generation with SSE streaming.
    Enforces an IP‑based concurrent request limit (1 per IP).
    """
    client_ip = request.client.host if request.client else "unknown"

    # --- Rate limiter: check + acquire slot ---
    async with active_clients_lock:
        if client_ip in active_clients:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Only one active generation per IP is allowed. Please wait."
                },
            )
        active_clients.add(client_ip)

    async def event_stream():
        try:
            # 1. Load active weight pointers into global model instance
            try:
                params_dict = weight_manager.get_weights(req.personality)
                model.load_weights(params_dict)
            except Exception as e:
                yield f"data: {json.dumps({'error': f'Failed to swap weight matrix: {str(e)}'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            # 2. Tokenize prompt
            input_indices = encode_prompt(req.prompt)
            if not input_indices:
                # Fallback priming token for completely empty prompt
                input_indices = [char_to_idx["\n"]]

            # 3. Instantiate sampler (temperature, blacklist)
            sampler = Sampler(temperature=req.temperature, blacklist=req.blacklist)

            # 4. Autoregressive streaming loop
            for _ in range(req.max_tokens):
                # Crop to sequence length 64 (model's training window)
                cropped_indices = input_indices[-64:]
                idx_arr = np.array(cropped_indices)

                try:
                    logits = model.forward(idx_arr)
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'Inference crash: {str(e)}'})}\n\n"
                    break

                # Sample next token index
                next_idx = sampler.sample(logits)
                input_indices.append(next_idx)

                # Decode character and stream to client
                next_char = idx_to_char.get(next_idx, " ")
                yield f"data: {json.dumps({'text': next_char})}\n\n"

                # Tiny async sleep to prevent CPU blocking + typewriter feel
                await asyncio.sleep(0.01)

            # 5. Signal end of stream
            yield f"data: {json.dumps({'done': True})}\n\n"

        finally:
            # Release rate‑limiter slot
            async with active_clients_lock:
                active_clients.discard(client_ip)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
