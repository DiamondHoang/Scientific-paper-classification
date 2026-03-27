# Scientific Paper Classification using Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive system for classifying scientific papers using various text encoding techniques and machine learning algorithms. This project evaluates traditional methods like **Bag-of-Words** and **TF-IDF** against modern **Sentence Embeddings** to achieve high-accuracy categorization across diverse scientific domains.

---

## Features

- **Multiple Text Representations:** BoW, TF-IDF, and state-of-the-art Sentence Embeddings.
- **Diverse ML Models:** Naive Bayes, K-Nearest Neighbors (KNN), Decision Trees, and K-Means clustering.
- **Comprehensive Evaluation:** Automatic generation of precision, recall, F1-score, and confusion matrices.
- **Reproducible Workflow:** Automated directory creation, data loading, and report saving in `.txt` and `.json` formats.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Diamond-Hoang/Scientific-paper-classification.git
   cd Scientific-paper-classification
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the complete pipeline—from data preparation to model evaluation and visualization—simply execute:

```bash
python main.py
```

The script will:
1. Initialize necessary output directories.
2. Load and preprocess the **arXiv-abstracts-large** dataset.
3. Train all model/vectorizer combinations.
4. Save detailed performance reports in `outputs/reports/`.
5. Generate confusion matrices in `outputs/figures/`.

---

## 📈 Results Summary

| Model          | BoW Accuracy | TF-IDF Accuracy | Embeddings Accuracy |
|----------------|--------------|-----------------|---------------------|
| Naive Bayes    | 0.8500       | 0.8300          | **0.8900**          |
| KNN            | 0.5300       | 0.8150          | **0.8900**          |
| Decision Tree  | 0.6350       | 0.5950          | 0.7150              |
| K-Means        | 0.5600       | 0.6150          | 0.8400              |

**Key Insights:**
- **Sentence Embeddings** provide the most robust representation, consistently yielding the highest accuracy across most models.
- **Naive Bayes** and **KNN** paired with embeddings achieve the top overall performance (89% accuracy).
- Traditional BoW models struggle with semantic nuances in scientific abstracts compared to contextual embeddings.

---

## Evaluation Details

Each experiment includes:
- Precision, Recall, and F1-score for each class
- Overall classification accuracy
- Confusion matrices for error analysis

All evaluation reports are saved in both **`.txt` and `.json` formats** to support reproducibility and further analysis.
