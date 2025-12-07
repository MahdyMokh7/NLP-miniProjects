# Word2Vec (CBOW & Skip-Gram) + MLP News Classification

### **From-Scratch Implementation + FastText-Based Embeddings**

This project explores core NLP representation learning through **manual Word2Vec implementations** and a downstream **MLP-based news classifier** using pretrained FastText embeddings. It demonstrates strong skills in:

* Neural network implementation from scratch
* Embedding learning (CBOW & Skip-Gram)
* Text preprocessing for Persian datasets
* Classification using MLP models
* Efficient training loops and evaluation

---

## 🚀 Project Overview

### **1️⃣ Word2Vec (CBOW & Skip-Gram) — Fully Implemented From Scratch**

A complete bottom-up implementation of Word2Vec without using ML libraries.

#### Preprocessing

* Tokenization, normalization, vocabulary construction
* Generating context–target pairs
* Windowing and negative sampling

#### Model Architecture

* Learnable input/output embedding matrices
* Forward pass for CBOW and Skip-Gram
* Softmax probability computation
* Manual backpropagation and embedding updates
* Custom batching and loss tracking

#### Comparison

* Training loss trends
* Qualitative embedding behavior
* Differences in learning dynamics between CBOW and Skip-Gram

---

## **2️⃣ News Classification Using MLP + FastText Embeddings**

A neural classifier built on top of pretrained **FastText wiki-news-300d-1M-subword** embeddings.

#### Preprocessing

* Cleaning and token-level normalization
* Converting samples to vector form via embedding aggregation
* Robust handling of OOV words (subword embeddings)

#### MLP Classifier

* Dense multi-layer architecture
* Activation functions, dropout/regularization
* Training loop with cross-entropy loss
* Performance evaluation on held-out data

#### Analysis

* Accuracy, precision, recall, F1-score
* Comparison between Word2Vec-derived and FastText embeddings

---

## 🛠️ Technical Skills Demonstrated

### **NLP**

✔ Manual Word2Vec (CBOW & Skip-Gram)
✔ Context-window data generation
✔ Embedding engineering (FastText, custom vectors)

### **Machine Learning**

✔ Implementing forward/backward propagation
✔ Training optimization loops
✔ Neural classifier design (MLP)
✔ Model performance evaluation

### **Engineering**

✔ Clean modular Python code
✔ Reproducible NLP/ML pipeline
✔ Efficient batching and memory management

---

## 📁 Repository Structure (Suggested)

```
├── data/
├── data/datasets/
│   ├── q1/
│   ├── q2/
├── models/
├── Reference-Glove-Embedding/
├── HW3.ipynb
└── README.md
```
