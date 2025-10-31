# NLP Language Modeling & Tokenization Project

This project explores **text preprocessing**, **tokenization**, and **statistical language modeling** techniques from scratch — step by step — in **Python**.  
It was developed as part of an NLP coursework focused on practical understanding of `Regex`, `Edit Distance`, `Tokenization`, and `N-gram Language Models` with smoothing.

---

## Project Overview

We gradually build up an NLP pipeline starting from raw text to fully functional probabilistic language models.  
The implementation avoids external NLP libraries (like Hugging Face tokenizers or NLTK) to demonstrate deep understanding of each component.

---

## Project Structure

### **1️⃣ Regular Expression & Auto-Correction**
- Detect valid email addresses using **Regex**.
- Implement **Minimum Edit Distance (Levenshtein Distance)** for auto-correction.
  - Calculates insertion, deletion, and substitution costs.
  - Suggests closest valid word candidates.

---

### **2️⃣ Tokenization**
We implement **three tokenizers** from scratch and explore their differences:

| Tokenizer | Description | Highlights |
|------------|--------------|-------------|
| **Rule-Based Tokenizer** | Uses handcrafted rules and regex to split Persian text into words and punctuation | Handles Persian spacing and normalization |
| **Byte Pair Encoding (BPE)** | Learns merges based on most frequent symbol pairs | Keeps `</w>` word-boundary markers for clean subword reconstruction |
| **WordPiece Tokenizer** | Similar to BPE but merges by maximizing conditional probability | Implements scoring based on unigram likelihood |

#### Visualization
Each tokenizer’s segmentation can be visualized to compare how subwords differ for the same Persian text.

---

### **3️⃣ N-gram Language Modeling**
We implement a fully functional **N-gram language model** supporting any `n` (2-gram, 4-gram, 8-gram, etc.), trained on tokenized corpora.

#### Features
- Dynamic vocabulary building with `<bos>`, `<eos>`, and `<unk>` tokens  
- Support for **Laplace**, **Backoff**, and **Interpolation** smoothing  
- Text **generation** with temperature-based sampling  
- **Perplexity** evaluation for model quality

#### Smoothing Methods
| Method | Formula / Idea | Description |
|---------|----------------|-------------|
| **Laplace (Add-1)** | `(count + 1) / (context + |V|)` | Prevents zero probabilities |
| **Backoff** | Use lower-order models scaled by λ (0.4ⁿ) | Steps down to smaller n-grams when unseen |
| **Interpolation** | Weighted sum of all n-gram orders | Smooth combination for stability |

---

### **4️⃣ Text Generation**
Using trained models:
- **2-gram** → simple, repetitive structure  
- **4-gram** → smoother and more grammatical output  
- **8-gram** → overfits to training text (copies patterns)

Example Persian generation (100 tokens) is compared for **fluency**, **coherence**, and **variety**.

---

### **5️⃣ Perplexity Evaluation**
We compute **sentence-level** and **average perplexity** for validation data:
```python
for sentence in df_val_tokenized:
    pp = model.perplexity([sentence])
