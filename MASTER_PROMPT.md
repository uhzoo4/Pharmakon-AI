# Master Context Prompt – Pharmakon System Generator

Copy and paste the entire block below into a high-reasoning language model (such as Gemini 1.5 Pro, Claude 3.5 Sonnet, or Gemini 3.5 Flash) to generate the entire codebase for the Pharmakon project.

```text
You are a senior ML engineer and full-stack developer. You are tasked with generating the complete, working codebase for "Pharmakon" (φάρμακον), a desktop web application that runs a local character-level Transformer language model trained from scratch in pure NumPy, styled as a Dark Academia grimoire.

Follow the instructions below to generate all files for the frontend (Next.js 14+) and backend (FastAPI + NumPy 2.5.1).

---

## 1. PROJECT STRUCTURE

Initialize the repository as a monorepo with the following structure:
pharmakon/
├── backend/
│   ├── requirements.txt
│   ├── transformer.py      # NumPy-only model architectures & mathematical layers
│   ├── generate.py         # Sampling, temperature, and token blacklist filters
│   ├── weight_manager.py   # In-memory weight swapping for personalities
│   └── main.py             # FastAPI REST endpoints & Server-Sent Events (SSE) stream
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

Write the following Python files. Use strictly NumPy for all mathematical transformations. Do not use PyTorch, TensorFlow, or JAX.

### 2.1 backend/requirements.txt
```text
numpy==2.5.1
fastapi==0.136.1
uvicorn[standard]==0.50.0
pydantic==2.13.0
```

### 2.2 backend/transformer.py
Implement:
1. `softmax(x, axis=-1)`: Numerically stable softmax using maximum subtraction.
2. `create_causal_mask(seq_len)`: Upper-triangular matrix summing -1e10 for future steps.
3. `precompute_freqs(head_dim, max_seq_len)`: Rotary Position Embedding cos/sin calculations.
4. `apply_rope(q, k, cos, sin)`: Apply rotational coordinates to Q and K.
5. `LayerNorm(dim, eps=1e-8)`: Pre-norm layers with trainable gamma and beta scaling.
6. `TransformerBlock`: Self-attention, causal mask summation, Softmax weight distribution, projection, Feed-Forward layers (Linear + ReLU + Linear), and Pre-Norm residual bindings.
7. `PharmakonTransformer`: Multi-layer stacking of transformer blocks, character vocabulary lookups, and load_weights utility mapping weight names from .npz dictionaries.

### 2.3 backend/weight_manager.py
Implement `WeightManager` mapping a `weights/` directory. On startup, scan for `*.npz` files (e.g., `kafkaesque.npz`, `camus_stranger.npz`, `dark_romance.npz`), load the NumPy array matrices into a memory-resident dictionary, and swap the reference pointer when a user changes their active personality.

### 2.4 backend/generate.py
Implement `Sampler(temperature, blacklist)` scaling output logits using temperature division and zeroing probability of blacklist token indices prior to final probability normalization and random token sampling.

### 2.5 backend/main.py
Implement:
- `GET /api/personalities`: Return a list of loaded weight file keys.
- `POST /api/generate`: Recieve prompt, personality, temperature, blacklist parameters, encode prompt, run the generator loop, and stream character-level responses using EventSource text/event-stream packets.

---

## 3. FRONTEND SPECIFICATIONS (NEXT.JS & TAILWIND)

### 3.1 frontend/package.json
Generate package dependencies mapping Next.js 14 (App Router), TailwindCSS, Framer Motion, and next-pwa.

### 3.2 frontend/tailwind.config.ts
Map the Dark Academia theme:
- Colors: Background: `#0d0f12`, Text: `#e5dcc3` (Parchment), Accent: `#b0302a` (Blood/Crimson), Secondary: `#2a3a4a`.
- Typography: Headings: `Cormorant Garamond` (serif), Typewriter: `JetBrains Mono` (monospace), Interface: `Inter`.

### 3.3 frontend/lib/context.tsx
Manage global state inside a `PharmakonProvider` tracking `activePersonality`, `temperature`, `blacklist`, and the active `messages` history.

### 3.4 frontend/app/layout.tsx & page.tsx
Assemble the desktop layout:
- **Header:**φάρμακον title and a serif personality carousel.
- **Main Chat:** Parchment cards dynamically rendering user and model dialogs. Generated text should utilize a typewriter effect with smooth Framer Motion ink-bleeds.
- **Settings Drawer:** Slide-out panel adjusting sampling temperature and blacklisted tokens.
- **Integration:** Connect to `POST /api/generate` and stream the responses directly into the message stack.
```
