# Master Prompt: Pharmakon Sampling Engine Optimization

Copy and paste the prompt below into **DeepSeek (V4 Pro / R1)** to generate or optimize the sampling engine.

***

```markdown
You are an expert NLP Engineer specializing in sampling mechanics for autoregressive language models.
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

### 💻 IMPLEMENTATION BLUEPRINT

Write the file `backend/generate.py` implementing:
1. `Sampler` class containing:
   - `__init__(temperature=0.8, blacklist=None)`
   - `sample(logits)`: process raw logits of shape `(seq_len, vocab_size)` and return the single sampled integer index.

Provide clean, robust, mathematical Python code.
```
