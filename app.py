import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

from utils.preprocessing import (
    load_dataset,
    get_recent_sequence,
    inverse_transform_rainfall,
    recursive_forecast
)

# ------------------------------
# Configuration
# ------------------------------
st.set_page_config(
    page_title="PH Rainfall Forecast Dashboard",
    layout="wide"
)

# ------------------------------
# Load Model (LSTM Only)
# ------------------------------
@st.cache_resource
def load_lstm_model():
    return TFSMLayer("models/converted/model_lstm", call_endpoint="serving_default")

# ------------------------------
# Rainfall Categories
# ------------------------------
CATEGORY_LABELS = {
    0: "No Rain",
    1: "Light Rain",
    2: "Moderate Rain",
    3: "Heavy Rain",
    4: "Extreme Rain"
}

EMOJIS = {
    0: "☀️",
    1: "🌦️",
    2: "🌧️",
    3: "🌧️🌧️",
    4: "⛈️⚠️"
}

GUIDANCE = {
    0: "Dry weather. Low impact.",
    1: "Light rain. Minimal disruption.",
    2: "Seasonal rain. Normal pattern.",
    3: "Heavy rain. Stay alert.",
    4: "Extreme rainfall. Possible flooding!"
}

# ------------------------------
# Rain Category Binning
# ------------------------------
@st.cache_data
def load_data():
    return load_dataset("data/Ph-Rainfall-DL.xlsx")

@st.cache_data
def get_rainfall_bins(df):
    bins = df["r1h"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    return [0] + bins + [np.inf]

def categorize_rain(rain, bins):
    if rain <= bins[1]:
        return 0
    elif rain <= bins[2]:
        return 1
    elif rain <= bins[3]:
        return 2
    elif rain <= bins[4]:
        return 3
    else:
        return 4

# ------------------------------
# Sidebar
# ------------------------------
df = load_data()
regions = sorted(df["Region"].dropna().unique())

with st.sidebar:
    st.header("🌏 Forecast Controls")
    region = st.selectbox("Select Region", regions)
    date = st.date_input("Start Forecast Date", pd.to_datetime("2025-12-01"))
    forecast_type = st.radio("Forecast Type", ["1-Day Forecast", "30-Day Forecast"])
    predict_btn = st.button("📈 Run Forecast")

# ------------------------------
# Forecasting Logic
# ------------------------------
if predict_btn:
    try:
        model = load_lstm_model()
        df_region = df[df["Region"] == region]
        X_input, scaler = get_recent_sequence(df_region, region, date)

        # One-Day Forecast
        if forecast_type == "1-Day Forecast":
            output = model(X_input)
            predicted = list(output.values())[0] if isinstance(output, dict) else output
            predicted_value = predicted[0][0]
            rainfall_mm = inverse_transform_rainfall(predicted_value, scaler)

            bins = get_rainfall_bins(df)
            category = categorize_rain(rainfall_mm, bins)

            st.subheader("📊 Forecast Results")
            st.metric("📏 Predicted Rainfall", f"{rainfall_mm:.2f} mm")
            st.metric("🧠 Rainfall Category", f"{EMOJIS[category]} {CATEGORY_LABELS[category]}")
            st.info(f"📌 {GUIDANCE[category]}")
            st.caption(f"Thresholds (r1h): {bins}")

        # 30-Day Forecast
        else:
            forecast_values = recursive_forecast(model, X_input, scaler, steps=30)
            future_dates = pd.date_range(start=date, periods=30, freq="D")

            bins = get_rainfall_bins(df)
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Rainfall (mm)": forecast_values,
                "Category": [CATEGORY_LABELS[categorize_rain(x, bins)] for x in forecast_values]
            })

            st.subheader("📆 30-Day Rainfall Forecast")
            st.line_chart(forecast_df.set_index("Date")["Rainfall (mm)"])

            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")