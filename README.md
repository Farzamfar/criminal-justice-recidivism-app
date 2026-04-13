# ⚖️ Criminal Justice Recidivism Analysis & Prediction App

🌐 **Live App:** [https://marzieh-criminal-justice-app.streamlit.app/](https://marzieh-criminal-justice-app.streamlit.app/)
🔌 **Live API:** http://criminal-justice-env.eba-g3zhx8gr.us-west-2.elasticbeanstalk.com
📖 **API Docs:** http://criminal-justice-env.eba-g3zhx8gr.us-west-2.elasticbeanstalk.com/docs

## 📌 Project Overview
A comprehensive machine learning project analyzing the COMPAS recidivism 
dataset — the same algorithm used in real US courtrooms to predict reoffending risk. 
This project goes beyond building a model to critically examine **algorithmic bias 
in criminal justice AI**.

## 📁 Repository Contents

| File | Description |
|---|---|
| `app.py` | Interactive Streamlit web application |
| `Recidivism_bias_analysis.ipynb` | Full analysis notebook |
| `compas.csv` | COMPAS dataset (ProPublica) |
| `requirements.txt` | Python dependencies |

## 🔍 What's Inside the Notebook

| Analysis | Description |
|---|---|
| Exploratory Data Analysis | Recidivism rates by race, age, and sex |
| Random Forest Model | 64.5% accuracy with cross validation |
| Confusion Matrix | Where the model makes mistakes |
| SHAP Values | Why the model makes each individual prediction |
| Bias Analysis | False positive rates broken down by race |
| Race Removal Experiment | Does removing race fix the bias? (Spoiler: No) |
| Unsupervised Learning | PCA + K-Means clustering without labels |
| Cluster Profiling | Natural groups found in the data |

## 🚨 Key Findings
- African-American individuals are wrongly labeled high risk at **nearly double** 
  the rate of Caucasian individuals (24% vs 13.7%)
- Removing race from the model does **not** fix the bias — proxy variables carry 
  the same discrimination
- Unsupervised learning independently identified the same high-risk group 
  as the supervised model — validating the findings
- A 72% recidivism rate was found in the repeat offender cluster 
  (14+ prior convictions) discovered without any labels

## 🛠️ Built With
- Python 3.11
- Scikit-learn (Random Forest, KMeans, PCA)
- SHAP (Explainable AI)
- Streamlit (Web App)
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
