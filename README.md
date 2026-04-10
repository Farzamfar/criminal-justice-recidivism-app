🌐 **Live App:** [https://marzieh-criminal-justice-app.streamlit.app/](https://marzieh-criminal-justice-app.streamlit.app/)
# ⚖️ Recidivism Risk Prediction App

A machine learning web application that predicts the likelihood of reoffending within two years, built using the COMPAS dataset.

## 🔍 Overview
This app demonstrates the use of machine learning in criminal justice — and highlights the importance of understanding bias in algorithmic decision-making.

## 🛠️ Built With
- Python 3.11
- Scikit-learn (Random Forest Classifier)
- Streamlit
- Pandas & NumPy
- COMPAS Dataset (ProPublica)

## 📊 Features
- Predicts recidivism risk based on age, sex, race, prior convictions, and charge degree
- Shows model accuracy
- Displays feature importance chart
- Includes bias warning about COMPAS dataset

## ⚠️ Ethics Note
The COMPAS dataset has documented racial bias. This app is for educational purposes only and should never be used for real criminal justice decisions.

## 🚀 How to Run Locally
```bash
conda activate criminal_justice
streamlit run app.py
```

## 👤 Author
Built as part of a data science portfolio focusing on criminal justice analytics.
