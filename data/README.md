# 📚 Pharmakon — Training Data Directory & Corpus Pipeline

This directory acts as the local datastore for raw text corpora (`.txt`). These datasets are parsed, mathematically cleaned, and used to train the corresponding model checkpoints.

---

## ─── 🔮 Core Requirements ───

To train successfully, each raw file must follow these specifications:
* **Format:** Plain text files ending with the `.txt` extension.
* **Encoding:** Strictly **UTF-8** encoded.
* **Naming Convention:** The file stem must match the personality identifier exactly (e.g. `greek_mythology.txt` trains `greek_mythology.npz`).
* **Volume:** Recommended $>10,000$ characters of prose to capture style, syntax, and punctuation.

---

## ─── 🔮 Target Vocabulary ($V = 97$) ───

The custom NumPy transformer restricts its token space to **97 characters** (printable ASCII + basic control characters). Any out-of-vocabulary character will cause dictionary-lookup crashes during training.

The vocabulary $\mathcal{V}$ is defined as:
$$\mathcal{V} = \{ \text{'\n'}, \text{'\t'} \} \cup \{ c \in \text{Unicode} \mid 32 \le \text{ord}(c) \le 126 \}$$

### Supported Characters:
* Control: Newline (`\n`), Tab (`\t`)
* Numbers: `0-9`
* Lowercase & Uppercase Letters: `a-z`, `A-Z`
* Standard Punctuation & Symbols: Space, `!`, `"`, `#`, `$`, `%`, `&`, `'`, `(`, `)`, `*`, `+`, `,`, `-`, `.`, `/`, `:`, `;`, `<`, `=`, `>`, `?`, `@`, `[`, `\`, `]`, `^`, `_`, `` ` ``, `{`, `|`, `}`, `~`

---

## ─── 🔮 The Cleaning Pipeline (`clean_corpus.py`) ───

To prepare your raw text files without manual editing, run the following command from the project root:
```bash
python clean_corpus.py
```

This utility applies three sequential mathematical operations:

### 1. Smart Punctuation Mapping (Norm)
Replaces common non-ASCII punctuation symbols with compatible ASCII equivalents:
* Left/Right Curly Quotation Marks (`“`, `”`) $\to$ Standard Double Quote (`"`)
* Left/Right Curly Apostrophes (`‘`, `’`) $\to$ Standard Single Quote (`'`)
* Em/En Dashes (`—`, `–`) $\to$ Standard Hyphen (`-`)
* Non-breaking Spaces (`\xa0`) $\to$ Standard Space (` `)
* Carriage Returns (`\r\n` or `\r`) $\to$ Unix Newline (`\n`)

### 2. Canonical Unicode Decomposition (NFKD)
Splits letters from their combining diacritical marks:
$$\text{NFKD}(c) = (c_{\text{base}}, m_1, m_2, \dots)$$
* Example: `é` is decomposed into the base letter `e` and the combining acute accent mark `\u0301`.
* Accents and umlauts (combining marks) are stripped, and the base letters (`e`, `o`, `u`, `c`, etc.) are retained.

### 3. Indicator Filtering (Projection $\pi$)
Any remaining character not belonging to the 97-token vocabulary is dropped from the sequence.

---

## ─── 🔮 Validation & Analytics ───

After running `clean_corpus.py`, the script outputs a validation report:
* **Vocabulary Coverage Fraction ($\eta$):** Asserts that $100\%$ of characters belong to the vocabulary ($\eta = 1.0$).
* **Shannon Information Entropy ($H$):** Displays the statistical information density in bits per character.
* **Atomic Backup:** Automatically creates a `.txt.orig` backup of your original file before saving the sanitized version, ensuring your raw data is never lost.
