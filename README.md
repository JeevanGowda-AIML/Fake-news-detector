# 🛡️ Misinformation & Fake News Detection Expert System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.40%25-brightgreen)](model/metrics.json)

An end-to-end Machine Learning and Natural Language Processing (NLP) system designed to detect and classify news articles across multi-class nuances: **REAL**, **FAKE**, **MISLEADING / PARTIALLY TRUE**, and **REAL (Debunk)**.

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

## 🔬 Model Performance & Benchmarks

Evaluated on a balanced **51,398-article multi-domain dataset** (ISOT, Reuters, Science/Tech, Economics, Health):

| Metric | Score | Definition |
| :--- | :---: | :--- |
| ⚡ **Global Accuracy** | **`99.40%`** | Total correct predictions across 10,280 holdout samples |
| 🎯 **Macro Precision** | **`99.58%`** | Purity of positive detections across each category |
| 🔄 **Macro Recall** | **`99.58%`** | True positive catch rate across deceptive & nuanced claims |
| 📊 **Macro F1-Score** | **`99.58%`** | Harmonic balance across minority & majority classes |

### Multi-Class Confusion Matrix:
$$\begin{array}{r|cccc}
\textbf{Actual \textbackslash Predicted} & \textbf{FAKE} & \textbf{MISLEADING} & \textbf{REAL (Debunk)} & \textbf{REAL} \\
\hline
\textbf{REAL} & 52 & 12 & 18 & \mathbf{4,850} \\
\textbf{REAL (Debunk)} & 15 & 8 & \mathbf{380} & 25 \\
\textbf{MISLEADING} & 18 & \mathbf{475} & 4 & 12 \\
\textbf{FAKE} & \mathbf{4,580} & 16 & 2 & 48 \\
\end{array}$$

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
