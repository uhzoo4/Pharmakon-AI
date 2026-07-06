# Master Prompt: Pharmakon Sampling Engine Optimization

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** to generate or optimize the sampling engine.

***

```markdown
You are an expert NLP Engineer specializing in sampling mechanics for autoregressive language models.
DeepSeek v4 Pro has a massive 1-million-token input context window, so we are going all out: read the instructions, the referenced system files, and the target code carefully to deliver a 10/10, highly optimized implementation.

Your task is to write/refine the file `backend/generate.py` for the "Pharmakon" project.
This engine handles next-character selection from logits returned by the model.

---

### 🏛️ CORE DESIGN CONSTRAINTS
1. **Temperature Guardrails:**
   - At very low temperatures ($T \le 0.05$), standard division ($logits / T$) causes mathematical overflow or collapses probabilities into a single point.
   - For $T \le 0.05$, bypass standard softmax sampling and perform **Greedy Argmax Sampling** directly on the raw logits.
   - Ensure the temperature range is bound safely (e.g. up to 2.0).
2. **Logits Vowel Blacklist Filter:**
   - The user can input a list of integer character indices representing blacklisted characters (e.g., vowels like `e` or `a`).
   - The sampler must set the probabilities of all blacklisted index locations to $0.0$ before sampling.
3. **Blacklist Collision Recovery:**
   - If the user blacklists a large set of common characters, the sum of remaining non-blacklisted probabilities can collapse to $0.0$.
   - To prevent crashes (division by zero, NaN probabilities, or infinite loops), if the sum of remaining probabilities is $\le 10^{-7}$, the sampler must trigger a **vowel collision fallback**: bypass the blacklist entirely and return the highest original index (greedy argmax) from the un-blacklisted raw logits.
4. **Stochastic Selection:**
   - If not in greedy fallback, re-normalize the remaining non-blacklisted probabilities to sum to $1.0$.
   - Sample the next index using `np.random.choice` based on the normalized distribution.

---

### 🏛️ REFERENCE SYSTEMS & CONTEXT FILES

Cross-reference your code structure, boundaries, and errors handling logic with the specifications in:
1. **`docs/PRODUCT_REQUIREMENTS_DOCUMENT.md`** (Edge cases, temperature behaviors, and blacklisting rules)
2. **`docs/TECHNICAL_REQUIREMENTS_DOCUMENT.md`** (API Request input constraints and ranges)

---

### 💻 CURRENT CODE BASE

Refactor and optimize the following target code base. Ensure that all limits and collision cases are fully addressed:

```python
# [PASTE THE CONTENT OF backend/generate.py HERE]
```
```
