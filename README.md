
```markdown
# 🚀 CIC-IDS2017 Intrusion Detection  
### Handling Class Imbalance Using SMOTETomek

This repository contains a complete machine-learning pipeline for network intrusion detection using the **CIC-IDS2017** dataset.  
Because the dataset is highly imbalanced, we apply **SMOTETomek**, a hybrid oversampling–undersampling technique, to create a cleaner and more balanced dataset for model training.

---

## 📌 Project Goals

- Preprocess and clean the CIC-IDS2017 dataset  
- Merge granular attack types into unified categories  
- Apply **SMOTETomek** to handle class imbalance  
- Train ML & DL models (Random Forest, LSTM, etc.)  
- Visualize metrics and compare model performance  

---

## 📁 Repository Structure

```

├── smotetomek.ipynb        # Full notebook with preprocessing, balancing, and modeling
├── combined_all.csv        # Merged CIC-IDS2017 dataset
├── data/                   # (Optional) Raw CSV files
└── README.md               # Project documentation

````

---

## ⚙️ Workflow Overview

### 1️⃣ Load Dataset

```python
data = pd.read_csv("combined_all.csv")
````

Column names are cleaned and the class distribution is analyzed.

---

### 2️⃣ Attack Label Merging

Many attack subtypes in CIC-IDS2017 are grouped into broader categories:

| Category         | Includes                               |
| ---------------- | -------------------------------------- |
| **Web Attack**   | SQL Injection, XSS, Brute Force Web    |
| **DoS**          | GoldenEye, Hulk, Slowloris, Heartbleed |
| **DDoS**         | DDoS attacks                           |
| **Botnet**       | Bot attacks                            |
| **Infiltration** | Infiltration attempts                  |
| **Benign**       | Normal traffic                         |

---

### 3️⃣ Preprocessing

Steps include:

* Cleaning column names
* Handling missing and infinite values
* Scaling numerical features
* Splitting into features (X) and labels (y)
* Train-test splitting

---

## ⚖️ 4️⃣ Balancing with SMOTETomek

To address heavy class imbalance:

```python
from imblearn.combine import SMOTETomek
sm = SMOTETomek()
X_res, y_res = sm.fit_resample(X, y)
```

**SMOTETomek = SMOTE Oversampling + Tomek Links Undersampling**

Benefits:

* Increases minority samples
* Removes noisy borderline samples
* Improves model performance

---

## 🤖 5️⃣ Model Training

Implemented models:

* **Random Forest Classifier**
* **LSTM Deep Learning Model**

Training history and evaluation results are stored in:

```python
history_dict = {}
results_dict = {}
```

---

## 📊 Visualization

The notebook produces:

* Class distribution charts (before/after SMOTETomek)
* Accuracy vs. Epoch
* Loss vs. Epoch
* Model comparison charts
<img width="1122" height="732" alt="image" src="https://github.com/user-attachments/assets/c9ec72f9-0881-44e9-a3a5-87bdcb3188bb" />

Using:

```python
matplotlib
numpy
```

---

## 📈 Evaluation Metrics

The project uses:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix


<img width="489" height="214" alt="image" src="https://github.com/user-attachments/assets/bed5d03c-ef88-4e5a-a597-ca731532fd9f" />

---

## ▶️ How to Run

### Install Dependencies

```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib tensorflow
```

### Run the Notebook

```bash
jupyter notebook smotetomek.ipynb
```

Make sure `combined_all.csv` is available in the project directory.

---

## 🛠️ Technologies Used

* Python
* Pandas / NumPy
* Scikit-Learn
* Imbalanced-Learn
* TensorFlow / Keras
* Matplotlib

---

## 📚 Dataset

**CIC-IDS2017**
Published by the Canadian Institute for Cybersecurity.

---

## 🙌 Acknowledgments

* CIC for providing the dataset
* Imbalanced-Learn contributors
* Open-source machine learning community



