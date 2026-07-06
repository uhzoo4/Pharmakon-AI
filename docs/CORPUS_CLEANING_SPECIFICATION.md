# Pharmakon – Corpus Cleaning & Mathematical Validation Specification

**Version:** 1.0  
**Domain:** Text Normalization, Information Theory, & Vocabulary Alignment  
**Target Engine:** Character-level Causal Transformer ($V = 97$, sequence window $S = 64$)

---

## 1. Mathematical Formulation

To train a character-level transformer without runtime dictionary errors, the training text corpus must be mapped from a broad unicode space to a strictly defined, closed set of characters.

### 1.1 Vocabulary Spaces

Let the target vocabulary be $\mathcal{V}$, consisting of $V = 97$ discrete tokens:
$$\mathcal{V} = \{ \text{'\n'}, \text{'\t'} \} \cup \{ c \in \text{Unicode} \mid 32 \le \text{ord}(c) \le 126 \}$$

Let the source text corpus be a sequence $C = (c_1, c_2, \dots, c_N)$ of length $N$, drawn from a large, open character alphabet $\Sigma$.

The objective is to construct a projection function $\pi: \Sigma \to \mathcal{V} \cup \{ \emptyset \}$ that maps every character in the text sequence to a character in the target vocabulary, or deletes it (mapping to empty set $\emptyset$), producing the cleaned sequence:
$$C_{\text{clean}} = \left( \pi(c_i) \mid \pi(c_i) \neq \emptyset \text{ for } i = 1, \dots, N \right)$$

---

### 1.2 Normalization and Unicode Decomposition

To maximize information retention, the mapping function $\pi$ must not simply drop non-ASCII characters, but must normalize them using a sequence of rules:

1. **Smart Punctuation Substitution:**
   Non-ASCII characters denoting punctuation are explicitly mapped to their nearest ASCII counterparts:
   $$\text{Norm}(c) = \begin{cases} 
     \text{"'"} & \text{if } c \in \{ \text{’}, \text{‘} \} \\
     \text{'"'} & \text{if } c \in \{ \text{”}, \text{“} \} \\
     \text{"-"} & \text{if } c \in \{ \text{—}, \text{–} \} \\
     \text{" "} & \text{if } c = \text{\xa0 (non-breaking space)} \\
     \text{"\n"} & \text{if } c = \text{\r\n} \\
     c & \text{otherwise}
   \end{cases}$$

2. **Canonical Unicode Decomposition (NFKD):**
   To handle accented letters (e.g. `é`, `ü`), we apply Unicode Normalization Form KD (NFKD). This splits a character into its base character and any combining diacritical marks:
   $$\text{NFKD}(c) = (c_{\text{base}}, m_1, m_2, \dots)$$
   Where:
   * $c_{\text{base}} \in \text{ASCII}$ is the base alphabet character.
   * $m_j \in \text{Unicode Category 'Mn'}$ represent combining non-spacing marks (accents).
   
   The projection strips the diacritics, retaining only the base character:
   $$\pi_{\text{decomp}}(c) = \begin{cases}
     c_{\text{base}} & \text{if } c_{\text{base}} \in \mathcal{V} \\
     \emptyset & \text{otherwise}
   \end{cases}$$

---

### 1.3 Statistical Profiling Metrics

Before and after processing, the corpus must be evaluated on three criteria:

1. **Vocabulary Coverage Fraction ($\eta$):**
   $$\eta = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(c_i \in \mathcal{V})$$
   Where $\mathbb{I}(x)$ is the indicator function:
   $$\mathbb{I}(x) = \begin{cases} 1 & \text{if } x \text{ is true} \\ 0 & \text{otherwise} \end{cases}$$
   The clean corpus must strictly guarantee $\eta = 1.0$.

2. **Character Probability Distribution ($P(v)$):**
   The probability of occurrence for any character $v \in \mathcal{V}$ in a sequence of length $M$ is:
   $$P(v) = \frac{1}{M} \sum_{i=1}^M \mathbb{I}(c_i = v)$$

3. **Shannon Information Entropy ($H$):**
   Measures the average information density of the text in bits/character:
   $$H(C) = - \sum_{v \in \mathcal{V}} P(v) \log_2 P(v)$$

---

## 2. LLM Code Generation Prompt

Use this prompt with any Google AI (or another LLM) to generate the clean corpus script:

```markdown
You are an expert systems programmer specialized in compiler design and NLP tokenization. 

Your task is to write a standalone Python script `clean_corpus.py` that processes raw text files (.txt) inside a `data/` folder and saves them back with a mathematical guarantee of 100% vocabulary compliance.

### Target Specifications:
1. **Target Vocabulary (97 tokens):**
   - Newline `\n` and Tab `\t`
   - All printable ASCII characters: space (32) through tilde (126).
   
2. **Mathematical Normalization Pipeline:**
   - **Smart replacement map:** Transform curly single/double quotes, em/en dashes, non-breaking spaces (`\xa0`), and carriage returns (`\r\n`) to their standard ASCII equivalents.
   - **Unicode Decomposition (NFKD):** Decompose characters (using standard `unicodedata.normalize('NFKD', ...)`). Strip out any combining diacritical marks (Unicode category `'Mn'`).
   - **Filtering:** Drop any remaining characters not in the 97-token vocabulary. Track dropped counts for logging.
   
3. **Statistical Metrics Output:**
   Compute and log:
   - File length pre and post-cleaning.
   - Vocabulary Coverage Fraction (\eta) for raw vs. cleaned states.
   - Shannon Entropy (H = - \sum P(c) \log_2 P(c)) in bits/character for raw vs. cleaned states.
   
4. **Safety & Verification:**
   - Create a `.txt.orig` backup of the raw file before writing changes.
   - Assert that the final vocabulary coverage is exactly 1.0 before writing to disk.

Write pure Python code using only standard libraries (except NumPy if needed for vector math, but standard library `math` is preferred for portability). Use high-quality type hinting, and log the top 10 most common dropped characters for each file.
```
