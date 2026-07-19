# Pharmakon – Product Requirements Document (PRD)

**Version:** 1.2  
**Date:** 2026-07-05  
**Status:** Approved & Locked  
**Project Codename:** φάρμακον (Pharmakon)

---

## 1. Product Vision & Duality

Pharmakon is an offline-capable, highly stylized desktop web application that allows users to interact with a character-level Transformer language model written from scratch using only NumPy. 

The name **Pharmakon (φάρμακον)** represents the dualistic philosophy of the system:
* **The Poison:** It exposes raw language generation, mimicking human existential dread, absurdity, and doom. It runs locally, consuming CPU cycles, and demands manual mathematical execution (raw backpropagation) from the developer.
* **The Remedy:** It acts as an artistic grimoire for writers, students, and thinkers, offering creative prompts, stylistic inspiration, and complete transparency into AI operations with zero cloud dependencies or telemetry tracking.

## 1.1 Full Artistic Fidelity – No Censorship
Pharmakon is a fully unrestricted, uncensored text generation system.  
- It contains no keyword filters, no toxicity classifiers, and no ethical-alignment post-processing.  
- The model reproduces the full stylistic and thematic spectrum of its training corpus, including the dark, violent, or psychologically intense content present in classic existentialist and Gothic literature.  
- The user-facing character blacklist is a poetic constraint (lipogram), not a safety mechanism. All semantic filtering is the responsibility of the human operator, never the software.

---

## 2. Literary Personality Profiles

The model holds multiple discrete personality configurations. Swapping a personality dynamically re-binds the backend weight parameters and changes the aesthetic skin of the user interface.

### 2.1 The Kafkaesque Bureaucrat
* **Corpus Source:** Franz Kafka’s *Metamorphosis*, *The Trial*, *The Castle*, and private diaries.
* **Aesthetic Skin:** Slate gray visual accents (`#2a3a4a`), typewriter sound overlays, and a faint soundscape resembling bureaucratic whispering.
* **Prose Style:** Surreal dread, bureaucratic absurdity, endless clinical justifications, and clinical but deeply unsettling sentences.

### 2.2 The Camus Absurdist
* **Corpus Source:** Albert Camus’ *The Stranger*, *The Myth of Sisyphus*, and private notebooks.
* **Aesthetic Skin:** Parchment colors with faint desert-dusk accents, visual simplicity, and a silent background.
* **Prose Style:** Detached clarity, existential indifference, sparse and beautiful sentences, and acceptance of the meaningless void.

### 2.3 The Gothic Dark Novelist
* **Corpus Source:** Emily Brontë’s *Wuthering Heights*, Mary Shelley’s *Frankenstein*, and selected Gothic poetry.
* **Aesthetic Skin:** Crimson accents (`#b0302a`), visual shadows, and a faint background audio overlay of rain and thunder.
* **Prose Style:** Passionate, doomed, obsessive, highly descriptive, and emotionally intense.

---

## 3. Core User Journeys

### 3.1 Session Initiation
1. The user launches the local application.
2. The landing screen displays the title "φάρμακον" in large Greek letters, alongside a rotating carousel of antique book covers representing the available personalities.
3. The user hovers over a cover, triggering an ink-bleed hover transition, and clicks to enter the chat interface.

### 3.2 Prompting and streaming
1. The user types a prompt (e.g., *"The sun rises, yet..."*) in the monospace input box at the bottom.
2. Upon pressing Enter or clicking the arrow icon, the input is disabled, and the prompt card slides into the conversation area.
3. The model begins generating characters. Each character is streamed live with a typing animation and matching typewriter ticks.
4. The generation can be paused or terminated by clicking a "Sever Connection" button.

### 3.3 Fine-Tuning & Constraints
1. The user clicks a small cog icon to slide open the Settings Drawer.
2. Adjusts the **Temperature** slider (range: $0.1$ to $2.0$). High temperatures trigger more chaotic prose; low temperatures trigger repetitive, structured sentences.
3. Enters comma-separated characters into the **Blacklist** input box (e.g., `e,t,s`). The model is blocked from generating these characters, forcing it to find alternative phrasing.

### 3.4 Conversation Preservation
1. The user clicks the "Extract Dialogue" button in the header.
2. The conversation is compiled and exported as a styled PDF that mimics an old manuscript with aged paper coloring, serif typography, and a crimson seal.

---

## 4. Feature Manifest & Edge Cases

### 4.1 Detailed Feature Grid
- **Carousel Swapper:** Dynamic Cover Flow rendering of the three active personalities.
- **SSE Character Streamer:** Real-time token streaming using chunked Server-Sent Events.
- **Settings Drawer:** Slides out from the right, holding local context values.
- **Manuscript PDF Exporter:** Local PDF renderer converting HTML markup into a printable document.
- **PWA Installer:** Offline capability allowing local loading of the UI shell.

### 4.2 Edge Cases and Robustness
* **Blacklist Collisions:** If the user blacklists common vowels (like `e` or `a`), the model's logits for those indices are zeroed out. If all valid next characters are blacklisted, the sampler automatically falls back to selecting the highest remaining non-zero index to prevent infinite loops or crashes.
* **Extreme Temperatures:** 
  * At $T \to 0.1$, the probabilities collapse, causing the model to repeat characters indefinitely (e.g., `the the the`). The UI should display a warning: *"Low temperature may cause repetitive stuttering."*
  * At $T \to 2.0$, logits lose meaning, causing the model to generate pure gibberish (e.g., `x&9!a_f`). The UI warning: *"High temperature may cause linguistic dissolution."*
* **Empty Prompts:** If the user submits an empty prompt, the system automatically inserts a default starting character (typically `\n` or `I`) to prime the decoder.

---

## 5. Offline & System Boundaries
- The application is a **Progressive Web App (PWA)**, meaning the frontend assets (HTML, CSS, JS) are cached in the browser's Cache Storage.
- Because model execution requires a custom NumPy environment, **inference cannot run in-browser** without a local backend.
- If the frontend cannot establish a connection to `127.0.0.1:8000`, the UI displays a clean alert: *"The local engine is sleeping. Please execute uvicorn main:app to awaken the grimoire."*
