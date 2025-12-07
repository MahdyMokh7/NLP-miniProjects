# 📝 Text Classification with Logistic Regression & Naive Bayes

### **From-Scratch Implementation + Feature Engineering with Scikit-Learn**

This project is an end-to-end **Natural Language Processing (NLP)** exercise focused on building and evaluating text classification models using both **from-scratch implementations** and **standard machine learning libraries**.

It demonstrates strong proficiency in **Python**, **machine learning fundamentals**, **mathematical modeling**, and **NLP preprocessing pipelines**, while showcasing the ability to implement core algorithms manually without relying on external ML libraries.

---

## 🚀 Project Overview

The assignment consists of two major parts:

---

## **1️⃣ Logistic Regression & Naive Bayes — Implemented From Scratch (60 points)**

This section demonstrates a full low-level implementation of two classical machine learning algorithms.
I manually implemented:

### 🔧 **Data Preprocessing**

* Text normalization (lowercasing, trimming whitespace)
* Tokenization
* **Bag-of-Words (BoW)** vectorization built from scratch
* Train/test splitting without external utilities

### 📊 **Evaluation Metrics (Manual Implementation)**

* Accuracy
* Precision
* Recall
* F1-Score

### 🤖 **Model Implementation**

* **Logistic Regression from scratch**, including:

  * Sigmoid function
  * Cross-entropy loss
  * Gradient descent optimization
  * Weight updates and convergence
* **Multinomial Naive Bayes from scratch**, including:

  * Likelihood estimation
  * Laplace smoothing
  * Log-probability formulation

### 📈 **Result Analysis**

* Performance comparison between Logistic Regression & Naive Bayes
* Discussion of advantages, limitations, and behavior of each model

---

## **2️⃣ Logistic Regression & Naive Bayes using Sklearn**

The second part focuses on **advanced feature engineering** and evaluates classical ML models using the scikit-learn ecosystem.

### 🧩 **Feature Extraction**

Structured, statistical, and content-based features were engineered, including:

#### 🔹 Structural Features

* Message length
* Word count
* Average token length
* Special character ratios

#### 🔹 Statistical & Content Features

* N-gram features
* Term Frequency–Inverse Document Frequency (TF-IDF)
* Stop-word frequency
* Uppercase/lowercase ratios

#### 🔹 Custom Features

* Hand-crafted domain-specific indicators
* Binary and count-based features relevant to the text classification task

### 🧪 **Model Training**

* Logistic Regression (sklearn)
* MultinomialNB (sklearn)
* Comparison with from-scratch models

---

## 🛠️ **Technologies & Skills Demonstrated**

### **NLP & Data Processing**

✔ Manual text normalization and feature extraction
✔ Bag-of-Words vectorizer implementation
✔ Experience with TF-IDF, n-grams, and statistical feature engineering

### **Machine Learning**

✔ Mathematical implementation of Logistic Regression
✔ Mathematical implementation of Multinomial Naive Bayes
✔ Understanding of loss functions, gradients, and optimization
✔ Evaluation metrics implemented manually
✔ Use of scikit-learn models and utilities

