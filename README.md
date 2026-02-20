<div align="center">

# 🛡️ CIC-IDS2017 Intrusion Detection System
### Tackling Class Imbalance with SMOTETomek

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> A complete ML/DL pipeline for **network intrusion detection** that addresses the severe class imbalance in the CIC-IDS2017 dataset using the hybrid **SMOTETomek** resampling technique.

</div>

---

## 📖 Overview

The **CIC-IDS2017** dataset is a benchmark for network intrusion detection research, but suffers from extreme **class imbalance** — benign traffic vastly outnumbers attack samples. This project demonstrates how to:

- 🔄 Preprocess and clean the raw dataset at scale
- 🏷️ Consolidate fine-grained attack subtypes into logical categories
- ⚖️ Apply **SMOTETomek** (SMOTE + Tomek Links) to produce a balanced training set
- 🤖 Train and compare **Random Forest** and **LSTM** models on the resampled data
- 📊 Visualize class distributions, training dynamics, and evaluation metrics

---

## 🎯 Project Goals

| Goal | Description |
|------|-------------|
| 🧹 **Preprocessing** | Clean column names, handle NaN/Inf values, scale features |
| 🏷️ **Label Merging** | Group 14+ attack subtypes into 6 unified categories |
| ⚖️ **Resampling** | Apply SMOTETomek hybrid technique to balance classes |
| 🤖 **Modeling** | Train Random Forest & LSTM; store history and results |
| 📈 **Evaluation** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## 📁 Repository Structure

```
📦 Application-of-the-SMOTETomek-Technique
├── 📓 smotetomek.ipynb        # Complete pipeline: preprocessing → modeling
├── 📊 combined_all.csv        # Merged & cleaned CIC-IDS2017 dataset
├── 📂 data/                   # (Optional) Raw daily CSV files
└── 📄 README.md               # Project documentation
```

---

## ⚙️ Pipeline Workflow

```
Raw CIC-IDS2017 CSV Files
         │
         ▼
  ┌─────────────────┐
  │  Data Loading & │
  │  Cleaning       │  ← Strip spaces, fix column names, drop NaN/Inf
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Label Merging  │  ← Group subtypes → 6 unified categories
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Feature Scaling│  ← StandardScaler / MinMaxScaler
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  SMOTETomek     │  ← Oversample minority + remove Tomek Links
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Model Training │  ← Random Forest & LSTM
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Evaluation &   │
  │  Visualization  │  ← Metrics, Confusion Matrix, Plots
  └─────────────────┘
```

---

## 🏷️ Attack Label Merging

Fine-grained attack labels are merged into 6 broad categories for cleaner multi-class classification:

| 🏷️ Category | Included Subtypes |
|-------------|-------------------|
| **🌐 Web Attack** | SQL Injection, XSS, Brute Force (Web) |
| **💥 DoS** | GoldenEye, Hulk, Slowloris, Heartbleed |
| **🌊 DDoS** | DDoS (LOIT, LOIC UDP, etc.) |
| **🤖 Botnet** | Bot / C&C traffic |
| **🕵️ Infiltration** | Infiltration attempts |
| **✅ Benign** | Normal traffic |

---

## ⚖️ SMOTETomek — Hybrid Resampling

<table>
<tr>
<td width="50%">

**Why SMOTETomek?**

The CIC-IDS2017 dataset is heavily skewed — benign traffic can be 100× more frequent than rare attacks. Standard models trained on imbalanced data tend to classify everything as benign.

**SMOTETomek** solves this by combining:
- **SMOTE**: Generates synthetic minority samples
- **Tomek Links**: Removes noisy borderline majority samples

</td>
<td width="50%">

```python
from imblearn.combine import SMOTETomek

sm = SMOTETomek(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Before: highly skewed distribution
# After:  balanced classes for fair training
```

</td>
</tr>
</table>

**Benefits:**
- ✅ Boosts minority-class recall significantly
- ✅ Removes ambiguous boundary samples
- ✅ Leads to more generalizable models

---

## 🤖 Models Implemented

### 🌲 Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_resampled, y_resampled)
```
- Ensemble of decision trees
- Robust to feature scaling
- Handles high-dimensional data well

---

### 🧠 LSTM Deep Learning Model
```python
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
- Captures temporal patterns in network traffic
- Deep learning approach for sequential data
- Trained with early stopping and dropout regularization

---

## 📊 Visualizations

The notebook generates the following plots:

**Class Distribution (Before vs After SMOTETomek)**

<img width="1122" height="732" alt="Class Distribution Comparison" src="https://github.com/user-attachments/assets/c9ec72f9-0881-44e9-a3a5-87bdcb3188bb" />

---

**Model Evaluation — Confusion Matrix & Metrics**

<img width="489" height="214" alt="Evaluation Metrics" src="https://github.com/user-attachments/assets/bed5d03c-ef88-4e5a-a597-ca731532fd9f" />

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | True positives / (True + False positives) |
| **Recall** | True positives / (True + False negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Full breakdown of prediction errors |

---

## 🚀 Quick Start

### 📋 Prerequisites

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib tensorflow jupyter
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### ▶️ Run the Notebook

```bash
jupyter notebook smotetomek.ipynb
```

> ⚠️ **Note:** Make sure `combined_all.csv` is present in the project root directory before running the notebook.

### 📥 Dataset

Download the CIC-IDS2017 dataset from Kaggle:

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-CIC--IDS2017-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

---

## 🛠️ Technologies & Libraries

<div align="center">

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Resampling** | Imbalanced-Learn (SMOTETomek) |
| **Deep Learning** | TensorFlow / Keras |
| **Visualization** | Matplotlib, Seaborn |
| **Notebook** | Jupyter |

</div>

---

## 📚 References & Dataset

- **CIC-IDS2017 Dataset** — Canadian Institute for Cybersecurity  
  [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

- **SMOTETomek** — Imbalanced-Learn Documentation  
  [https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html](https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTETomek.html)

- **Chawla et al. (2002)** — SMOTE: Synthetic Minority Over-sampling Technique  
  *Journal of Artificial Intelligence Research*

---

## 🙌 Acknowledgments

- 🏛️ **Canadian Institute for Cybersecurity** — for the CIC-IDS2017 dataset
- 🔧 **Imbalanced-Learn contributors** — for the SMOTETomek implementation  
- 🌍 **Open-source ML community** — for making research reproducible

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

</div>
