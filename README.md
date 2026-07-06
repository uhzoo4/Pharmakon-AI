# φάρμακον (Pharmakon) — Local Grimoire Language Model

> **Poison & Remedy.** A character-level Transformer language model built entirely from scratch in pure NumPy, served via FastAPI, and interfaced through a gorgeous Dark Academia Next.js web application.

---

## ─── 🔮 Project Overview ───

Pharmakon is an offline-capable, desktop-centric AI project designed to showcase deep learning implementation from absolute first principles. It contains:
1. **NumPy Model Engine:** Complete manual forward implementation of causal attention, Rotary Positional Embeddings (RoPE), and layer normalization inside `backend/transformer.py`.
2. **Dynamic Weight Swap:** High-speed in-memory swap of `.npz` parameter files corresponding to distinct literary personalities (e.g., Kafkaesque existential dread, Camus Absurdist, or Gothic Dark Romance).
3. **SSE Streaming Server:** FastAPI backend delivering character-by-character tokens in real-time.
4. **Dark Academia Interface:** A rich, gothic typewriter UI built with Next.js, styled with custom parchment colors, serif headers, and typewriter animations.

---

## ─── 📁 Directory Layout ───

```
pharmakon/
├── backend/                  # CPU Inference Engine
│   ├── requirements.txt      # Fixed package dependencies (NumPy 2.5.1, FastAPI)
│   ├── transformer.py        # Custom Transformer classes & RoPE calculations
│   ├── generate.py           # Logits Sampler with temperature & blacklists
│   ├── weight_manager.py     # In-memory dictionary weights swapper
│   ├── main.py               # FastAPI server setting up GET/POST streams
│   └── weights/              # Directory holding trained personality .npz arrays
├── docs/                     # Comprehensive Architecture & Requirements Docs
│   ├── PRODUCT_REQUIREMENTS_DOCUMENT.md
│   ├── TECHNICAL_REQUIREMENTS_DOCUMENT.md
│   ├── SYSTEM_DNA.md
│   └── TASK_MANIFEST.md
├── MASTER_PROMPT.md          # Multi-file generation context instruction
├── master_blueprint.md       # Master architectural blueprint & mathematical keys
└── README.md                 # [This file] Setup & Startup guide
```

---

## ─── ⚡ Quick Start ───

### 1. Prerequisites
- **Python:** 3.11+
- **Node.js:** 18+ (for frontend)

### 2. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   The backend endpoints will be available at `http://127.0.0.1:8000`.

### 3. Frontend Setup (Next Phase)
1. Initialize the frontend directory with your choice of Next.js setup.
2. Adjust the `tailwind.config.ts` to include the Dark Academia theme:
   - Primary BG: `#0d0f12`
   - Parchment Text: `#e5dcc3`
   - Crimson Accent: `#b0302a`
   - Steel Secondary: `#2a3a4a`
3. Connect the message form to stream from `http://127.0.0.1:8000/api/generate`.

---

## ─── 📜 Documentation Reference ───

For deeper specifications, review the markdown files in the `docs/` folder:
- [Product Requirements](file:///d:/WebProjects/pharmakon%20AI/docs/PRODUCT_REQUIREMENTS_DOCUMENT.md)
- [Technical Requirements](file:///d:/WebProjects/pharmakon%20AI/docs/TECHNICAL_REQUIREMENTS_DOCUMENT.md)
- [System DNA & Primitives](file:///d:/WebProjects/pharmakon%20AI/docs/SYSTEM_DNA.md)
- [Task Manifest & Sprints](file:///d:/WebProjects/pharmakon%20AI/docs/TASK_MANIFEST.md)
- [Master Blueprint](file:///d:/WebProjects/pharmakon%20AI/master_blueprint.md)
- [Master Context Prompt](file:///d:/WebProjects/pharmakon%20AI/MASTER_PROMPT.md)
