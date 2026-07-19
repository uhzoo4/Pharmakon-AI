from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import asyncio
import numpy as np
import json
import sys
import time
import uuid

# Adjust sys.path to resolve root-level imports
sys.path.append(str(Path(__file__).parent.parent))
from clean_corpus import clean_text
import train

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


personality_models: dict[str, PharmakonTransformer] = {}
personality_models_lock = asyncio.Lock()

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


class TrainRequest(BaseModel):
    personality: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9_-]+$")
    text: str = Field(..., min_length=100, max_length=5_242_880)
    epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=16, ge=1, le=128)
    lr: float = Field(default=3e-4, ge=1e-5, le=1e-2)

    @field_validator("personality")
    @classmethod
    def clean_personality(cls, v: str) -> str:
        return v.strip().lower()


training_lock = asyncio.Lock()


def extract_weights(model: PharmakonTransformer) -> dict[str, np.ndarray]:
    """Extract model parameters cast to float32 for storage."""
    params_dict = {}
    params_dict["token_embedding"] = model.token_embedding.astype(np.float32)
    params_dict["W_out"] = model.W_out.astype(np.float32)
    params_dict["ln_final_gamma"] = model.ln_final.gamma.astype(np.float32)
    params_dict["ln_final_beta"] = model.ln_final.beta.astype(np.float32)

    for i, block in enumerate(model.blocks):
        prefix = f"block_{i}_"
        params_dict[prefix + "Wq"] = block.Wq.astype(np.float32)
        params_dict[prefix + "Wk"] = block.Wk.astype(np.float32)
        params_dict[prefix + "Wv"] = block.Wv.astype(np.float32)
        params_dict[prefix + "Wo"] = block.Wo.astype(np.float32)
        params_dict[prefix + "ln1_gamma"] = block.ln1.gamma.astype(np.float32)
        params_dict[prefix + "ln1_beta"] = block.ln1.beta.astype(np.float32)
        params_dict[prefix + "ln2_gamma"] = block.ln2.gamma.astype(np.float32)
        params_dict[prefix + "ln2_beta"] = block.ln2.beta.astype(np.float32)
        params_dict[prefix + "W1"] = block.W1.astype(np.float32)
        params_dict[prefix + "b1"] = block.b1.astype(np.float32)
        params_dict[prefix + "W2"] = block.W2.astype(np.float32)
        params_dict[prefix + "b2"] = block.b2.astype(np.float32)
    return params_dict


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
            # 1. Retrieve or instantiate a request-isolated model copy for this personality
            try:
                # Fast path: check cache first (no lock needed for read)
                if req.personality in personality_models:
                    local_model = personality_models[req.personality]
                else:
                    async with personality_models_lock:
                        # Double-check inside lock
                        if req.personality not in personality_models:
                            m = PharmakonTransformer(vocab_size=VOCAB_SIZE)
                            params_dict = weight_manager.get_weights(req.personality)
                            m.load_weights(params_dict)
                            personality_models[req.personality] = m
                        local_model = personality_models[req.personality]
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

            # 4. Ingest prompt sequentially using KV Cache
            # Crop to sequence length 64 (model's training window)
            prompt_indices = input_indices[-64:]
            kv_caches = None
            logits = None

            try:
                for idx in prompt_indices:
                    idx_arr = np.array([idx])
                    logits, kv_caches = local_model.forward(idx_arr, use_cache=True, kv_caches=kv_caches)
            except Exception as e:
                yield f"data: {json.dumps({'error': f'KV Cache initialization crash: {str(e)}'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            # Sample first token index
            if logits is None:
                yield f"data: {json.dumps({'error': 'Failed to compute initial logits'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return
            try:
                next_idx = sampler.sample(logits)
                input_indices.append(next_idx)
            except Exception as e:
                yield f"data: {json.dumps({'error': f'Sampling crash: {str(e)}'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            # Decode character and stream to client
            next_char = idx_to_char.get(next_idx, " ")
            yield f"data: {json.dumps({'text': next_char})}\n\n"
            await asyncio.sleep(0.01)

            # 5. Autoregressive streaming loop
            for _ in range(req.max_tokens - 1):
                idx_arr = np.array([next_idx])

                try:
                    logits, kv_caches = local_model.forward(idx_arr, use_cache=True, kv_caches=kv_caches)
                    assert isinstance(logits, np.ndarray)
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

            # 6. Signal end of stream
            yield f"data: {json.dumps({'done': True})}\n\n"

        finally:
            # Release rate‑limiter slot
            async with active_clients_lock:
                active_clients.discard(client_ip)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/train")
async def train_personality(req: TrainRequest):
    """
    On-demand training and fine-tuning endpoint.
    Accepts raw text, cleans it, trains a local model copy,
    validates the final parameters, and atomically updates the cache/disk.
    """
    training_id = str(uuid.uuid4())
    start_time = time.time()
    
    print(f"[{training_id}] Starting training run for personality: '{req.personality}'")
    
    async with training_lock:
        try:
            # 1. Clean Corpus
            cleaned_text, dropped_counts = clean_text(req.text)
            
            # 2. Validate clean text length (must fit sequence window)
            if len(cleaned_text) < 66:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Cleaned text is too short ({len(cleaned_text)} chars after cleaning). "
                                 f"Must be at least 66 characters to train."
                    }
                )
            
            # 3. Load or Initialize weights
            created_new = False
            if req.personality in weight_manager.personalities:
                orig_weights = weight_manager.get_weights(req.personality)
                weights_dict = {k: v.copy() for k, v in orig_weights.items() if k not in ("config", "metadata")}
            else:
                weights_dict = weight_manager._generate_xavier_weights()
                created_new = True
                
            # Cast parameters to float64 for backprop precision
            weights_double = {k: v.astype(np.float64) for k, v in weights_dict.items()}
            
            # 4. Instantiate local model instance
            local_model = PharmakonTransformer(
                vocab_size=VOCAB_SIZE,
                embed_dim=weight_manager.embed_dim,
                num_heads=weight_manager.num_heads,
                ff_dim=weight_manager.ff_dim,
                num_layers=weight_manager.num_layers,
                max_seq_len=64,
                dropout=0.0
            )
            local_model.load_weights(weights_double)
            
            # Calculate training hyperparameters
            warmup_steps = min(100, len(cleaned_text) // (req.batch_size * 64))
            
            # 5. Train in-memory (updates local_model in place)
            train.train(
                model=local_model,
                data=cleaned_text,
                char_to_idx=char_to_idx,
                epochs=req.epochs,
                batch_size=req.batch_size,
                seq_len=64,
                lr=req.lr,
                weight_decay=0.01,
                warmup_steps=warmup_steps,
                use_checkpoint=True
            )
            
            # 6. Extract updated parameters and validate
            updated_weights = extract_weights(local_model)
            
            # 7. Atomic Save & Cache Refresh
            meta_info = {
                "epochs": req.epochs,
                "token_count": len(cleaned_text)
            }
            weight_manager.save_weights(req.personality, updated_weights, metadata=meta_info)
            
            # Invalidate cached model to ensure the next generation request reloads new weights
            async with personality_models_lock:
                personality_models.pop(req.personality, None)
            
            duration = time.time() - start_time
            print(f"[{training_id}] Completed training successfully in {duration:.2f}s.")
            
            return {
                "success": True,
                "training_id": training_id,
                "personality": req.personality,
                "created": created_new,
                "epochs": req.epochs,
                "batch_size": req.batch_size,
                "learning_rate": req.lr,
                "training_tokens": len(cleaned_text),
                "training_time_seconds": round(duration, 2)
            }
            
        except Exception as e:
            import traceback
            print(f"[{training_id}] Training failed: {str(e)}")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Training failed: {str(e)}",
                    "training_id": training_id
                }
            )
