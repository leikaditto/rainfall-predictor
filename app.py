import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

from keras.layers import TFSMLayer
from utils.preprocessing import load_dataset, get_recent_sequence, inverse_scale_prediction
from utils.classification_preprocessing import load_classification_data, prepare_classification_input

# --------------------------
# CONFIG & CONSTANTS
# --------------------------
# Paths
DL_MODEL_PATHS = {
    "LSTM": "models/converted/model_lstm",
    "GRU": "models/converted/model_gru",
    "CNN-LSTM": "models/converted/model_cnnlstm"
}

CLF_MODEL_PATHS = "models/model_RanFor.pkl"

CATEGORY_LABELS = {
    0: "No Rain",
    1: "Light Rain",
    2: "Moderate Rain",
    3: "Heavy Rain",
    4: "Extreme Rain"
}

CLASSIFICATION_FEATURES = [
    'r1h', 'r3h', 'rfh_lag_1', 'r1h_lag_2', 'r3h_lag_3',
    'rfh_roll_mean_3', 'r1h_roll_std_5', 'is_wet_season'
]

# --------------------------
# LOADING FUNCTIONS
# --------------------------

@st.cache_data
def load_dl_data():
    return load_dataset("data/Ph-Rainfall-DL.xlsx")

@st.cache_data
def load_clf_data():
    return load_classification_data("data/Ph-Rainfall-Tree.xlsx")

@st.cache_resource
def load_dl_model(name):
    return TFSMLayer(DL_MODEL_PATHS[name], call_endpoint="serving_default")

@st.cache_resource
def load_clf_model():
    return joblib.load(CLF_MODEL_PATHS)

# --------------------------
# STREAMLIT UI
# --------------------------

st.set_page_config(page_title="PH Rainfall Prediction", layout="centered")
st.title("🌧️ PH Rainfall Prediction App")

st.markdown("""
Predict rainfall intensity based on Philippine region and date using either deep learning or classification model.
""")

# Mode selection
mode = st.radio("Choose Prediction Mode", ["Regression", "Classification"])

# Input: Region, Date
if mode == "Regression":
    df = load_dl_data()
else:
    df = load_clf_data()

regions = sorted(df["Region"].dropna().unique())
region = st.selectbox("Select Region", regions)
date = st.date_input("Select Date for Prediction", pd.to_datetime("2023-07-01"))

# Model selection
if mode == "Regression":
    model_name = st.selectbox("Select Deep Learning Model", list(DL_MODEL_PATHS.keys()))
else:
    model_name = "Random Forest" # Static label

# Predict Button
if st.button("Predict"):
    try:
        if mode == "Regression":
            df_region = df[df["Region"] == region]
            X_input, scaler = get_recent_sequence(df_region, region, date, window=10)

            model = load_dl_model(model_name)
            prediction = model.predict(X_input)
            rainfall_value = list(prediction.values())[0][0]
                
            rainfall_mm = inverse_scale_prediction(np.array([[rainfall_value]]), scaler)

            # Basic classification of mm
            if rainfall_mm < 50:
                category = "Below Normal"
            elif rainfall_mm < 150:
                category = "Normal"
            else:
                category = "Above Normal"

            st.success(f"📏 Predicted Rainfall: **{rainfall_mm:.2f} mm**")
            st.info(f"🌡️ Rainfall Category (estimated): **{category}**")

        else:
            input_row = prepare_classification_input(df, region, date, CLASSIFICATION_FEATURES)
            clf_model = load_clf_model()
            predicted_class = clf_model.predict(input_row)[0]
            label = CATEGORY_LABELS.get(predicted_class, "Unknown")

            st.success(f"🧠 Predicted Rainfall Category: **{label}** ({predicted_class})")

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")