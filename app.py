import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

from keras.layers import TFSMLayer
from utils.preprocessing import load_dataset, get_recent_sequence, inverse_scale_prediction

# -----------------------------------
# Model paths (SavedModel folders)
# -----------------------------------
DL_MODEL_PATHS = {
    "LSTM": "models/converted/model_lstm",
    "GRU": "models/converted/model_gru",
    "CNN-LSTM": "models/converted/model_cnnlstm"
}

# -----------------------------------
# Rain category labels
# -----------------------------------
CATEGORY_LABELS = {
    0: "No Rain",
    1: "Light Rain",
    2: "Moderate Rain",
    3: "Heavy Rain",
    4: "Extreme Rain"
}

# -----------------------------------
# Quantile bin logic
# -----------------------------------
@st.cache_data
def load_dl_data():
    return load_dataset("data/Ph-Rainfall-DL.xlsx")

@st.cache_data
def get_rainfall_quantile_bins(df):
    # Use 'r1h' as the base feature for categorization
    bins = df['r1h'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    return [0] + bins + [np.inf]

def categorize_rain(rain, bins):
    if rain <= bins[1]:
        return 0  # No Rain
    elif rain <= bins[2]:
        return 1  # Light Rain
    elif rain <= bins[3]:
        return 2  # Moderate Rain
    elif rain <= bins[4]:
        return 3  # Heavy Rain
    else:
        return 4  # Extreme Rain

@st.cache_resource
def load_dl_model(name):
    return TFSMLayer(DL_MODEL_PATHS[name], call_endpoint="serving_default")

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="PH Rainfall Forecast", layout="centered")
st.title("🌧️ PH Rainfall Forecast & Category Dashboard")

st.markdown("""
Predict future rainfall and automatically classify it into risk categories using deep learning.
""")

# Input: Region, Date, Model
df = load_dl_data()
regions = sorted(df["Region"].dropna().unique())
region = st.selectbox("Select Region", regions)
date = st.date_input("Select Prediction Date", pd.to_datetime("2025-12-01"))
model_name = st.selectbox("Select Forecasting Model", list(DL_MODEL_PATHS.keys()))

# Predict Button
if st.button("Predict"):
    try:
        # 1. Prepare model input
        df_region = df[df["Region"] == region]
        X_input, scaler = get_recent_sequence(df_region, region, date)

        # 2. Predict with selected model
        model = load_dl_model(model_name)
        prediction_dict = model(X_input)
        # Extract the value
        if isinstance(prediction_dict, dict):
            rainfall_value = list(prediction_dict.values())[0][0]
        else:
            rainfall_value = prediction_dict[0][0]

        rainfall_mm = inverse_scale_prediction(np.array([[rainfall_value]]), scaler)

        # 3. Quantile-based classification
        quantile_bins = get_rainfall_quantile_bins(df)
        rain_category = categorize_rain(rainfall_mm, quantile_bins)
        rain_label = CATEGORY_LABELS.get(rain_category, "Unknown")

        # 4. Output
        st.success(f"📏 Predicted Rainfall: **{rainfall_mm:.2f} mm**")
        st.info(f"🧠 Forecasted Rainfall Category (Quantile-based): **{rain_label} ({rain_category})**")

        # Optional: show bin values
        st.caption(f"Quantile Thresholds (based on r1h): {quantile_bins}")

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")