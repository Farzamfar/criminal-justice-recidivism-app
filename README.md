# ⚖️ Criminal Justice Recidivism Analysis & Prediction App

🌐 **Live App:** [https://marzieh-criminal-justice-app.streamlit.app/](https://marzieh-criminal-justice-app.streamlit.app/)

## 📌 Project Overview
A comprehensive machine learning project analyzing the COMPAS recidivism 
dataset — the same algorithm used in real US courtrooms to predict reoffending risk. 
This project goes beyond building a model to critically examine **algorithmic bias 
in criminal justice AI**.

## 📁 Repository Contents

| File | Description |
|---|---|
| `app.py` | Interactive Streamlit web application |
| `main.py` | FastAPI REST API |
| `Recidivism_bias_analysis.ipynb` | Full ML analysis notebook |
| `unsupervised_learning_compas.ipynb` | Unsupervised learning notebook |
| `model_interpretation.ipynb` | Model interpretation notebook |
| `compas.csv` | COMPAS dataset (ProPublica) |
| `requirements.txt` | Python dependencies |

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `Recidivism_bias_analysis.ipynb` | Full ML analysis — Random Forest, SHAP, Bias Analysis |
| `unsupervised_learning_compas.ipynb` | Unsupervised Learning — K-Means, Hierarchical, GMM, Anomaly Detection |
| `model_interpretation.ipynb` | Model Interpretation — Permutation Importance, PDP, Group Effects |

## 🔍 Key Analyses

| Analysis | Description |
|---|---|
| Exploratory Data Analysis | Recidivism rates by race, age, and sex |
| Random Forest Model | 64.5% accuracy with cross validation |
| Confusion Matrix | Where the model makes mistakes |
| SHAP Values | Why the model makes each individual prediction |
| Bias Analysis | False positive rates broken down by race |
| Race Removal Experiment | Does removing race fix the bias? (Spoiler: No) |
| Unsupervised Learning | K-Means, Hierarchical, GMM, Anomaly Detection |
| Model Interpretation | Permutation Importance, PDP, Group Effects |

## 🚨 Key Findings
- African-American individuals are wrongly labeled high risk at **nearly double** 
  the rate of Caucasian individuals (24% vs 13.7%)
- Removing race from the model does **not** fix the bias — proxy variables carry 
  the same discrimination
- Unsupervised learning independently identified the same high-risk group 
  as the supervised model — validating the findings
- A 72% recidivism rate was found in the repeat offender cluster 
  (14+ prior convictions) discovered without any labels
- Only **age and prior convictions** are genuinely useful predictors — 
  everything else adds noise or bias

## 🛠️ Built With
- Python 3.11
- Scikit-learn (Random Forest, KMeans, PCA, GMM)
- SHAP (Explainable AI)
- FastAPI + Uvicorn (REST API)
- Streamlit (Web App)
- AWS Elastic Beanstalk (Cloud Deployment)
- Pandas, NumPy, Matplotlib, Seaborn

## ⚠️ Ethics Note
The COMPAS dataset has documented racial bias. This project is for 
educational purposes only and should never be used for real criminal 
justice decisions. The goal is to demonstrate that **responsible AI 
requires both technical skill and domain expertise**.

## 🚀 How to Run Locally
```bash
conda create -n criminal_justice python=3.11
conda activate criminal_justice
pip install -r requirements.txt
streamlit run app.py
```

## 👤 Author
**Marzieh Farzamfar**
Senior Analyst — Criminal Justice | Aspiring Data Scientist
