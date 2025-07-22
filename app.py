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
from dashboard import show_dashboard
from utils.shared import categorize_rain, CATEGORY_LABELS, get_rainfall_quantile_bins

# ------------------------------
# Configuration
# ------------------------------
st.set_page_config(page_title="Rainfall Forecast (PH)", layout="wide")
st.title("🌧️ PH Rainfall Forecast")

DL_MODEL_PATH = "models/converted/model_lstm"

EMOJIS = {
    0: "☀️",
    1: "🌤️",
    2: "🌧️",
    3: "🌧️🌧️",
    4: "⛈️⚠️"
}

EXPLANATIONS = {
    0: "Clear skies and dry weather.",
    1: "Minor drizzle or brief light showers.",
    2: "Typical wet-season rain, safe for normal activities.",
    3: "Prepare for strong rain. Some local flooding possible.",
    4: "Severe rain expected. Stay safe, avoid low areas!"
}


# ─────────────────────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return TFSMLayer(DL_MODEL_PATH, call_endpoint="serving_default")

@st.cache_data
def load_dl_data():
    return load_dataset("data/Ph-Rainfall-DL.xlsx")

def inverse_transform_rainfall(value, scaler, target_index=0):
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0][target_index] = value
    return scaler.inverse_transform(dummy)[0][target_index]

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────

df = load_dl_data()
regions = sorted(df["Region"].dropna().unique())

# Layout Split
col_sidebar, col_main = st.columns([1, 3])

# ── 🟥 SIDEBAR CONTROLS ──────────────────────────────────────
with col_sidebar:
    st.header("🌍 Forecast Controls")
    region = st.selectbox("Select Region", regions)
    date = st.date_input("Start Forecast Date", pd.to_datetime("2025-12-01"))
    forecast = st.button("📊 Run Forecast", type="primary")

# ── 🟩 MAIN PANEL OUTPUT ─────────────────────────────────────
with col_main:
    if not forecast:
        st.subheader("📘 How it Works")
        st.markdown("""
        This tool forecasts **daily rainfall** for selected Philippine regions using a trained **model**.
        It predicts the **amount of rain** and classifies it into easy-to-understand categories:

        | Category | Label          | Emoji       | Description                        |
        |----------|----------------|-------------|------------------------------------|
        | 0        | No Rain        | ☀️           | Dry weather (Light drizzle)       |
        | 1        | Light Rain     | 🌤️           | Light Rain                      |
        | 2        | Moderate Rain  | 🌧️           | Typical wet-season rainfall        |
        | 3        | Heavy Rain     | 🌧️🌧️         | May cause flooding in some areas  |
        | 4        | Extreme Rain   | ⛈️⚠️          | Dangerous rainfall, stay alert!   |
        """)
        st.info("Please select a region and date on the left to begin the forecast.")
        
    else:
        try:
            df_region = df[df["Region"] == region]
            X_input, scaler = get_recent_sequence(df_region, region, date)
            model = load_model()
            result = model(X_input)

            # Get prediction
            output_array = list(result.values())[0] if isinstance(result, dict) else result
            predicted_value = output_array[0][0]
            rainfall_mm = inverse_transform_rainfall(predicted_value, scaler)

            # Classify
            quantile_bins = get_rainfall_quantile_bins(df)
            rain_category = categorize_rain(rainfall_mm, quantile_bins)
            rain_label = CATEGORY_LABELS[rain_category]
            emoji = EMOJIS[rain_category]
            explain = EXPLANATIONS[rain_category]

            # Output
            st.subheader("📊 Forecast Results")
            st.markdown(f"### ✏️ Predicted Rainfall\n**{rainfall_mm:.2f} mm**")
            st.markdown(f"### 🧠 Rainfall Category\n{emoji} **{rain_label}**")
            st.info(f"📌 {explain}")

            # Dashboard Visuals
            show_dashboard(start_date=str(date))

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")