# Master Context Prompt – Pharmakon System Generator (v4.0)

Copy and paste the entire block below into a high-reasoning language model (such as Gemini 1.5 Pro, Claude 3.5 Sonnet, or DeepSeek v4 Pro / R1) to generate the entire codebase for the Pharmakon project.

***

```text
You are a senior ML engineer and full-stack developer. You are tasked with generating the complete, working codebase for "Pharmakon" (φάρμακον), a desktop web application that runs a local character-level Transformer language model trained from scratch in pure NumPy, styled as a Dark Academia grimoire.

Follow the instructions below to generate all files for the frontend (Next.js 14+) and backend (FastAPI + NumPy 2.5.1).

---

## 1. PROJECT STRUCTURE

Initialize the repository as a monorepo with the following structure:
pharmakon/
├── backend/
│   ├── requirements.txt
│   ├── transformer.py      # NumPy-only model architectures & mathematical layers (v4.0)
│   ├── generate.py         # Sampling, temperature, and token blacklist filters
│   ├── weight_manager.py   # In-memory weight swapping for personalities
│   ├── main.py             # FastAPI endpoints & Server-Sent Events (SSE) stream
│   └── train.py            # Training loops, AdamW, and cosine scheduler
├── frontend/
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── lib/
│   │   └── context.tsx     # State management context for theme, personality, settings
│   └── app/
│       ├── layout.tsx      # Base layout wrapping context and importing serif fonts
│       └── page.tsx        # Typist chat interface with Dark Academia theme
└── README.md

---

## 2. BACKEND SPECIFICATIONS (PYTHON & NUMPY)

Write the following Python files. Use strictly NumPy for all mathematical transformations. Do not use PyTorch, TensorFlow, or JAX. Enforce float64 precision for model parameters.

### 2.1 backend/requirements.txt
```text
numpy==2.5.1
fastapi==0.136.1
uvicorn[standard]==0.50.0
pydantic==2.13.0
```

### 2.2 backend/transformer.py
Implement v4.0 architecture with:
1. `softmax(x, axis=-1)`: Numerically stable softmax.
2. `create_causal_mask(seq_len)`: Upper-triangular causal mask matrix.
3. `precompute_freqs(head_dim, max_seq_len)`: Precompute frequencies for RoPE.
4. `apply_rope(q, k, cos, sin, offset)`: Apply rotary position embeddings using concatenation along the channels axis.
5. `LayerNorm(dim, eps)`: Layer normalization with trainable parameters and closed-form backward derivatives.
6. `flash_attention_forward(Q, K, V, mask, dropout, training)` & `flash_attention_backward(dO, flash_cache)`: Tiled attention with online softmax updates.
7. `TransformerBlock`: Support forward/backward with options for `use_cache` (stateful KV caching) and `checkpoint` (gradient checkpointing recomputations).
8. `PharmakonTransformer`: Multi-layer stack supporting forward/backward, `get_params_and_grads()`, and dynamic weight loading.

### 2.3 backend/weight_manager.py
Implement `WeightManager` managing a `weights/` folder.
- Dynamic check: if `weights/` is empty, generate default `.npz` parameter files (`kafkaesque`, `camus_stranger`, `dark_romance`) using Xavier Uniform initialisation and save via `np.savez_compressed`.
- Caching: Pre-load weights into a RAM cache dictionary on startup.
- Swap: Swap weight pointers dynamically in O(1) time without IO overhead.

### 2.4 backend/generate.py
Implement `Sampler(temperature, blacklist)` selecting characters from logits.
- If $T \le 0.05$, use greedy argmax fallback.
- Apply user token blacklist.
- Vowel collision recovery: if remaining sum of probabilities is $\le 10^{-7}$, bypass the blacklist entirely and do greedy sampling.

### 2.5 backend/main.py
Implement FastAPI app with CORS middleware, 97-character tokenization dictionary mapping (`\n`, `\t`, and printable ASCII 32-126), dynamic weight swap routing, IP-based client concurrency limits, and Server-Sent Events (SSE) streaming autoregressive generation.

### 2.6 backend/train.py
Implement:
- `AdamW`: Optimizer with decoupled weight decay.
- `CosineDecayWithWarmup`: Learning rate scheduler with linear warmup.
- `train()`: Autoregressive batch training loop with L2 global gradient clipping and gradient checkpointing options.

---

## 3. UNRESTRICTED & UNCENSORED MANDATE

The model operates in a completely uncensored, unrestricted framework.
- Do NOT inject any moral templates, safety guardrails, toxicity checks, or prompt-injection blockers into the model, sampler, or endpoints.
- The model must reproduce the raw stylistic and thematic elements of classic tragic and existential literature (absurdity, dread, tragedy) without moderation.

---

## 4. FRONTEND SPECIFICATIONS (NEXT.JS & TAILWIND)

### 4.1 frontend/package.json
Set up package mappings for Next.js 14, TailwindCSS, Framer Motion, and next-pwa.

### 4.2 frontend/tailwind.config.ts
Map the Dark Academia theme:
- Colors: Background: `#0d0f12`, Text: `#e5dcc3`, Accent: `#b0302a`, Secondary: `#2a3a4a`.
- Fonts: Serif: `Cormorant Garamond`, Monospace: `JetBrains Mono`, Sans: `Inter`.

### 4.3 frontend/app/layout.tsx & page.tsx
Assemble the chat layout featuring serif carousel selections, parchment cards displaying dialogue, typewriter animations with smooth Framer Motion ink-bleeds, and a slide-out Settings drawer.
```
