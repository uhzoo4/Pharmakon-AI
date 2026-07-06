from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import asyncio
import numpy as np
import json
import string

from weight_manager import WeightManager
from transformer import PharmakonTransformer
from generate import Sampler

app = FastAPI(title="φάρμακον (Pharmakon) Backend")

# Setup CORS for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predefined Vocabulary (97 characters: newlines, tabs, and standard printable ASCII)
VOCAB_CHARS = ["\n", "\t"] + [chr(i) for i in range(32, 127)]
VOCAB_SIZE = len(VOCAB_CHARS)
char_to_idx = {char: idx for idx, char in enumerate(VOCAB_CHARS)}
idx_to_char = {idx: char for idx, char in enumerate(VOCAB_CHARS)}

def encode_prompt(prompt: str) -> list[int]:
    """Convert prompt string to character index integers."""
    return [char_to_idx.get(c, char_to_idx[" "]) for c in prompt]

# Load weight manager and pre-load/auto-initialize personalities
BASE_DIR = Path(__file__).parent
weights_dir = BASE_DIR / "weights"
weight_manager = WeightManager(weights_dir, vocab_size=VOCAB_SIZE)

# Global model instance
model = PharmakonTransformer(vocab_size=VOCAB_SIZE)

class GenerateRequest(BaseModel):
    personality: str
    prompt: str
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    max_tokens: int = Field(default=200, ge=1, le=500)
    blacklist: list[int] = Field(default_factory=list)

@app.get("/api/personalities")
async def get_personalities():
    """Retrieve list of available loaded personalities."""
    return {"personalities": weight_manager.list_personalities()}

@app.post("/api/generate")
async def generate_text(req: GenerateRequest):
    """Autoregressive text generation streaming via Server-Sent Events (SSE)."""
    
    async def event_stream():
        try:
            # 1. Load active weight pointers into global model instance
            params_dict = weight_manager.get_weights(req.personality)
            model.load_weights(params_dict)
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Failed to swap weight matrix: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # 2. Tokenize prompt
        input_indices = encode_prompt(req.prompt)
        if not input_indices:
            input_indices = [char_to_idx["\n"]]  # primimg token fallback for empty prompt

        # 3. Instantiate sampler
        sampler = Sampler(temperature=req.temperature, blacklist=req.blacklist)

        # 4. Stream loop
        for _ in range(req.max_tokens):
            # Crop context window to training sequence length (seq_len = 64)
            cropped_indices = input_indices[-64:]
            idx_arr = np.array(cropped_indices)  # shape (seq_len,)
            
            # Forward pass: returns (seq_len, vocab_size)
            try:
                logits = model.forward(idx_arr)
            except Exception as e:
                yield f"data: {json.dumps({'error': f'Inference crash: {str(e)}'})}\n\n"
                break

            # Sample next token
            next_idx = sampler.sample(logits)
            input_indices.append(next_idx)

            # Decode and yield to SSE connection
            next_char = idx_to_char.get(next_idx, " ")
            yield f"data: {json.dumps({'text': next_char})}\n\n"

            # Tiny async delay to simulate typewriter tick and yield control
            await asyncio.sleep(0.01)

        # Yield completion packet
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
