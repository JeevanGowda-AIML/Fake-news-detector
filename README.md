# 🛡️ Misinformation & Fake News Detection Expert System

[![Live Demo](https://img.shields.io/badge/Live%20App-truth--guard--ai.streamlit.app-00F0FF?style=for-the-badge&logo=streamlit&logoColor=black)](https://truth-guard-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Group--Safe%20Accuracy-98.67%25-brightgreen)](model/metrics.json)

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify news articles across multi-class nuances: **REAL**, **FAKE**, **MISLEADING / PARTIALLY TRUE**, and **REAL (Debunk)**.

> 🚀 **Live Demo:** Access the live interactive web dashboard at **[https://truth-guard-ai.streamlit.app/](https://truth-guard-ai.streamlit.app/)**

---

## 🏛️ System Architecture

```
                               ┌────────────────────────┐
                               │    Input Raw Text      │
                               └───────────┬────────────┘
                                           │
                                           ▼
                   ┌─────────────────────────────────────────────────┐
                   │ 1. Sentence-Level Claim & Caveat Parser         │
                   │ • Sentence Tokenization & Role Tagging          │
                   │ • Contrastive Caveat Detection (Half-Truths)    │
                   │ • NLI Contradiction & Claim-to-Correction Ratio │
                   └───────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                   ┌─────────────────────────────────────────────────┐
                   │ 2. Stylometry & Source Credibility Layer        │
                   │ • Institutional Authority (WHO, CDC, Gov Press) │
                   │ • 10-Factor Normalized Stylometric Diagnostics  │
                   │ • Temporal Context Analysis                     │
                   └───────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                   ┌─────────────────────────────────────────────────┐
                   │ 3. Offline Factual Grounding & In-Memory Cache  │
                   │ • Verified Ground Truth Matcher                 │
                   │ • Known Debunked Hoax Registry                  │
                   │ • < 1ms In-Memory LRU Query Cache               │
                   └───────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                   ┌─────────────────────────────────────────────────┐
                   │ 4. 3-Stage Probability Calibration Pipeline     │
                   │ • Multi-Class Balanced Logistic Regression      │
                   │ • Temperature Softmax Scaling (T=1.5)           │
                   │ • Class Rebalancing & Normalization             │
                   └───────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                   ┌─────────────────────────────────────────────────┐
                   │ 5. Hierarchical Deterministic Decision Engine   │
                   │ ├── Layer 1: Contrastive Caveat & Half-Truths   │
                   │ ├── Layer 2: Gated Fact-Check / Debunk Tree     │
                   │ ├── Layer 3: Contradiction & Claim Ratio Tree   │
                   │ ├── Layer 4: Misleading vs. Fake Boundary Check │
                   │ ├── Layer 5: Factual Knowledge & Source Trust   │
                   │ ├── Layer 6: Calibrated Statistical Classifier  │
                   │ └── Layer 7: Uncertainty Zone Safety Guard      │
                   └───────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                   ┌─────────────────────────────────────────────────┐
                   │ 6. Explainable Streamlit Dashboard UI           │
                   │ • 🟢 Factual / Institutional Declarations       │
                   │ • 🔵 Fact-Check Refutations                     │
                   │ • 🔴 Unverified Claims / Viral Rumors           │
                   │ • 📊 10-Factor Stylometric Diagnostics Radar    │
                   └─────────────────────────────────────────────────┘
```

---

## 📁 Clean Repository Structure

```
Fake news detection/
├── app/
│   └── app.py                     # Streamlit Frontend (Single Article, Batch CSV, Model Insights, Dataset Guide)
├── data/
│   └── download_data.py           # Multi-domain dataset generator & balancer
├── model/
│   ├── evaluate.py                # Plotly visualizations & metrics charts (4-class Confusion Matrix)
│   ├── metrics.json               # Serialized model metrics & per-class stats
│   ├── model.pkl                  # Serialized calibrated classifier pipeline
│   ├── predict.py                 # Multi-layer hierarchical decision engine
│   └── train.py                   # Calibrated model training pipeline
├── utils/
│   ├── claim_extractor.py         # Claim extraction, contrastive caveat parsing & NLI tension
│   ├── data_loader.py             # Standardized multi-class dataset loader
│   ├── knowledge_base.py          # Offline verified ground truth & hoax registry with LRU cache
│   ├── preprocess.py              # Text cleaning, academic filter & exaggeration detection
│   ├── source_credibility.py      # Institutional source authority & temporal context
│   └── stylometry.py              # 10-factor normalized stylometric credibility extractor
├── .gitignore                     # Git ignore rules for virtual environments & datasets
├── requirements.txt               # Project dependencies
├── run.bat                        # Windows 1-click startup batch script
├── run.ps1                        # PowerShell launch script
└── README.md                      # Project documentation
```

---

## ⚡ Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/<YOUR-USERNAME>/fake-news-detection.git
cd fake-news-detection
```

### 2. Set Up Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
# Option A: Double-click run.bat (Windows)

# Option B: Run via Streamlit command
streamlit run app/app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🔬 Model Performance & Anti-Leakage Benchmarks

To address dataset duplication and publisher watermark leakage, the training pipeline incorporates:

1. **Pre-Split Deduplication**: Removed **5,800 duplicate rows (11.28%)** prior to splitting, producing **45,598 unique articles**.
2. **Wire Dateline & Watermark Sanitization**: Stripped wire agency datelines (`(Reuters) -`, `(AP)`) and publisher footers (`Via: Breitbart`, `Via: Gateway Pundit`) so the model evaluates semantic credibility rather than source signatures.
3. **Group-Safe Story Splitting**: Syndicated story variants are grouped together using title clustering to guarantee **0% story cross-contamination** between train (36,348 samples) and test (9,250 samples).

### Verified Holdout Evaluation (9,250 Out-of-Story Samples):

| Metric                     |    Score     | Definition                                                |
| :------------------------- | :----------: | :-------------------------------------------------------- |
| ⚡ **Group-Safe Accuracy** | **`98.67%`** | Overall verification rate on held-out unseen news stories |
| ⚖️ **Balanced Accuracy**   | **`99.03%`** | Average recall weighted evenly across all 3 classes       |
| 🎯 **Macro Precision**     | **`99.01%`** | Average positive purity across each prediction category   |
| 🔄 **Macro Recall**        | **`99.03%`** | Deceptive & nuanced claim detection catch rate            |
| 📊 **Macro F1-Score**      | **`99.02%`** | Harmonic balance across minority and majority classes     |
| 📈 **Multi-Class ROC-AUC** | **`0.9992`** | One-vs-Rest class discrimination area under the curve     |

### Detailed Per-Class Breakdown:

| Category          | Precision | Recall (Sensitivity) | F1-Score | Holdout Support |
| :---------------- | :-------: | :------------------: | :------: | :-------------: |
| 🔴 **FAKE**       | `98.15%`  |       `98.40%`       | `98.28%` |      3,562      |
| ⚠️ **MISLEADING** | `100.00%` |      `100.00%`       | `100.0%` |       607       |
| 🟢 **REAL**       | `98.88%`  |       `98.70%`       | `98.79%` |      5,081      |

### Multi-Class Confusion Matrix (Predicted vs Actual):

| Actual Class \ Predicted Class |   FAKE    | MISLEADING |   REAL    |
| :----------------------------- | :-------: | :--------: | :-------: |
| **FAKE**                       | **3,505** |     0      |    57     |
| **MISLEADING**                 |     0     |  **607**   |     0     |
| **REAL**                       |    66     |     0      | **5,015** |

> ℹ️ **Operational Scope Note:** The statistical ML model evaluates _linguistic deception patterns, stylistic anomalies, and sensationalism_. For evolving real-world news, truth classification is performed in synergy with deterministic fact-checking refutations, contrastive caveat parsing, and offline verified knowledge grounding.

---

## 🖥️ Application Features

1. **Single Article Classifier**:
   - Real-time prediction with probability bars and donut charts.
   - Sentence-by-sentence color-coded visual highlighter (🟢 Factual, 🔵 Debunk, 🔴 Claim).
   - Factual grounding cards and institutional authority trust scores.
   - 10-factor stylometric diagnostics (sensationalism, formality, emotional intensity, numeric density).

2. **Batch CSV Predictor**:
   - Upload any CSV file containing news text.
   - Batch classification with progress tracking and distribution charts.
   - Instant 1-click export of results to CSV.

3. **Model Metrics & Insights**:
   - Live interactive KPIs, 5-fold cross-validation benchmarks, and 4-class Confusion Matrix heatmap.
   - Per-class precision, recall, and F1-score breakdown tables.

4. **Dataset & Architectural Guide**:
   - Interactive pipeline documentation and class definitions.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
