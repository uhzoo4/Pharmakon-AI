# Pharmakon – Task Manifest & Cross-Reference Table

This document maps feature requirements to implementation specifics and tracks Sprints 0 & 1 deliverables.

---

## 1. Feature Map & Gap Analysis

| Feature (PRD) | Module / Script (TRD) | Verification Strategy | Gaps & Mitigations |
| --- | --- | --- | --- |
| **Personality Switcher** | `weight_manager.py` (swapping weight pointers in dictionary). | Verify dynamic switching endpoint (`GET /api/personalities`). | Gaps: PDF export and background audio not documented in backend. Mitigation: Defer audio to Post-MVP; add frontend PDF conversion using `jspdf`. |
| **Text Generation** | `generate.py` (logits sampling with temp + blacklist). | API testing using Postman/cURL to check response variation. | Verify temperature bounds and blacklist formatting. |
| **Typewriter Streaming** | `main.py` (FastAPI `StreamingResponse` yielding SSE). | Validate MIME type `text/event-stream` and check client connection stability. | Standardize JSON envelopes for event messages. |
| **Offline PWA Shell** | Next.js layout & `next-pwa` integration. | Audit using Chrome DevTools Lighthouse in offline emulation. | Gaps: Backend must be running locally. Mitigation: Add clear "Local Server Disconnected" warning in frontend. |

---

## 2. Sprint 0 Deliverable Checklist (Foundations)

- [x] **Pin Dependencies:** Create `requirements.txt` pinning exact versions of `numpy==2.5.1`, `fastapi==0.136.1`, `uvicorn[standard]==0.50.0`, `pydantic==2.13.0`.
- [x] **Rotary Positional Embeddings (RoPE):** Implement `precompute_freqs`, `rotate_half`, and `apply_rope` in `transformer.py`.
- [x] **Layer Normalization:** Implement `LayerNorm.forward` pass in `transformer.py`.
- [x] **Causal Mask & Softmax:** Implement numerically stable `softmax` and `create_causal_mask` in `transformer.py`.
- [x] **In-Memory weight management:** Implement `WeightManager` class in `weight_manager.py` to scan the `weights/` directory for `.npz` files and buffer them.

---

## 3. Sprint 1 Deliverable Checklist (API & Inference Loop)

- [x] **Transformer Assembly:** Assemble `TransformerBlock` and `PharmakonTransformer` classes in `transformer.py`.
- [x] **Sampling System:** Complete `Sampler` class containing temperature scaling and index blacklist filtering in `generate.py`.
- [x] **FastAPI Routes:** Implement `/api/personalities` to list available `.npz` files and the `/api/generate` SSE streaming placeholder in `main.py`.
- [ ] **Frontend Scaffold:** Initialize Next.js layout, setup React Context for active personality and configuration states.
- [ ] **CSS Setup:** Build custom theme in TailwindCSS configuration referencing the Dark Academia colors (`#0d0f12`, `#e5dcc3`, `#b0302a`, `#2a3a4a`).
