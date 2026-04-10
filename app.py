import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Load and prepare data ---
@st.cache_data
def load_data():
    df = pd.read_csv('compas.csv')
    df = df[['age', 'sex', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid']].dropna()
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['c_charge_degree'] = df['c_charge_degree'].map({'F': 1, 'M': 0})
    df['race'] = df['race'].astype('category').cat.codes
    return df

# --- Train model ---
@st.cache_resource
def train_model(df):
    X = df.drop('two_year_recid', axis=1)
    y = df['two_year_recid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, X.columns.tolist()

# --- App ---
st.title("⚖️ Recidivism Risk Prediction")
st.markdown("This app predicts the likelihood of reoffending within two years based on the COMPAS dataset.")

df = load_data()
model, acc, features = train_model(df)

st.sidebar.header("Model Performance")
st.sidebar.metric("Accuracy", f"{acc:.1%}")

st.header("Enter Case Details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    race_options = sorted(df['race'].unique())
    race = st.selectbox("Race Code (0–5)", race_options)

with col2:
    priors = st.slider("Prior Convictions", 0, 30, 0)
    charge = st.selectbox("Charge Degree", ["Felony", "Misdemeanor"])

sex_val = 1 if sex == "Male" else 0
charge_val = 1 if charge == "Felony" else 0

input_data = pd.DataFrame([[age, sex_val, race, priors, charge_val]], columns=features)
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

st.header("Prediction")
if prediction == 1:
    st.error(f"⚠️ High Risk of Recidivism — Probability: {probability:.1%}")
else:
    st.success(f"✅ Low Risk of Recidivism — Probability: {probability:.1%}")

st.header("Feature Importance")
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
st.bar_chart(importance_df.set_index('Feature'))

st.header("⚠️ Bias Warning")
st.warning("The COMPAS dataset has documented racial bias. This app is for educational purposes only and should never be used for real criminal justice decisions.")