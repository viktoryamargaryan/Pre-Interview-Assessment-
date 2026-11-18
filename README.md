# Pre-Interview Candidate Acceptance Analysis & Prediction

This repository contains an end-to-end Data Science and Machine Learning project analyzing candidate pre-interview parameters (`logatta.csv`). The objective is to perform feature engineering, exploratory data analysis (EDA), and train classification models to predict candidate acceptance (`accepted for the interview`).

---

## Project Overview

Understanding candidate selection patterns helps streamline the recruitment process. This notebook processes raw HR/applicant data, handles categorical/continuous features, performs correlation analysis, and benchmarks three standard classification models.

### Key Workflow Steps:
* **Data Cleaning & Preprocessing**: 
  * Handled categorical features using `OrdinalEncoder` (`BusinessTravel`, `MaritalStatus`, `OverTime`).
  * Created dummy variables / One-Hot Encoding (`pd.get_dummies`) for `OverTime` and `Gender`.
  * Normalized continuous numerical features (`Age`, `DailyRate`) using `StandardScaler`.
  * Dropped irrelevant identifier columns (`EmployeeNumber`).
* **Exploratory Data Analysis (EDA)**:
  * Analyzed class balance on the target variable (`accepted for the interview`).
  * Plotted feature distributions (`DailyRate` and `Age` via histograms with KDEs).
  * Generated correlation heatmaps and boxplots comparing key numerical metrics against acceptance outcomes.
* **Model Benchmark & Evaluation**:
  * Evaluated **Logistic Regression**, **Gaussian Naive Bayes**, and **K-Nearest Neighbors (KNN)** using stratified train-test splits.
  * Generated accuracy comparisons and confusion matrices.

---

## Dataset Overview

The analysis relies on `logatta.csv`, which contains key demographic and workplace features:

| Feature Name | Description / Type | Preprocessing Applied |
| :--- | :--- | :--- |
| `Age` | Numerical continuous | Scaled via `StandardScaler` |
| `DailyRate` | Numerical continuous | Scaled via `StandardScaler` |
| `Education` | Numerical ordinal | Kept as numeric |
| `BusinessTravel` | Categorical | Encoded via `OrdinalEncoder` |
| `MaritalStatus` | Categorical | Encoded via `OrdinalEncoder` |
| `OverTime` | Binary categorical | Encoded via `OrdinalEncoder` & `get_dummies` |
| `Gender` | Binary categorical | One-Hot Encoded (`Gender_Female`, `Gender_Male`) |
| `EmployeeNumber` | Unique Identifier | Dropped before model training |
| **`accepted for the interview`** | **Target Variable (Binary)** | Boolean target label (`True`/`False`) |

---

## Model Performance & Results

Models were evaluated on an 80/20 train-test split (`random_state=42`):

* **Logistic Regression**: Serves as the primary baseline linear classifier.
* **Gaussian Naive Bayes**: Evaluated for probabilistic classification performance.
* **K-Nearest Neighbors (KNN)**: Evaluated using $k=5$ neighbors.

> *Metrics including exact accuracy scores, precision, recall, and confusion matrices are generated dynamically upon running the notebook.*

---

## Note on Notebook Outputs

To generate and view the visualizations, confusion matrices, and metrics:
1. **Google Colab / Kaggle**: Upload `Pre-Interview Assessment.ipynb` and `logatta.csv` directly to Colab or run it in a Kaggle Notebook environment.
2. **Local Execution**: Follow the quickstart guide below to execute all cells locally.
3. **For visualizations**: You can view the screenshots uploaded.

---

##  Quickstart & Setup Guide

### 1. Prerequisites
Ensure Python 3.8+ is installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```
### 2. Repository Setup
Clone the repository and navigate to the project directory:

```bash
git clone [https://github.com/viktoryamargaryan/Pre-Interview-Assessment-.git](https://github.com/viktoryamargaryan/Pre-Interview-Assessment-.git)
cd Pre-Interview-Assessment-
```

