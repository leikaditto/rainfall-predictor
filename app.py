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

def inverse_transform_rainfall(value, scaler, target_index=0):
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0][target_index] = value
    return scaler.inverse_transform(dummy)[0][target_index]

# -----------------------------------
# Streamlit UI
# -----------------------------------
df = load_dl_data()
st.set_page_config(page_title="PH Rainfall Forecast", layout="centered")

# Sidebar controls (region, forecast date, model, rf type, period)
st.sidebar.header("📋 Dashboard Controls")
region = st.sidebar.selectbox("Select Region", sorted(df["Region"].dropna().unique()))
date = st.sidebar.date_input("Forecast Date", pd.to_datetime("2025-12-01"))
model_name = st.sidebar.selectbox("Forecasting Model", list(DL_MODEL_PATHS.keys()))
# Rainfall type (e.g., r1h, r3h, rfq — placeholder now)
rainfall_feature = st.sidebar.selectbox("Rainfall Feature", ["r1h", "r3h", "rfq"])  # You can expand this later
forecast_mode = st.sidebar.radio("Forecast Period", ["1-Day Forecast", "30-Day Forecast"])

st.title("🌧️ PH Rainfall Forecast & Category Dashboard")
st.markdown("""
Predict future rainfall and automatically classify it into risk categories using deep learning.
""")

st.markdown(f"**🗺️ Region:** {region}  |  **📅 Date:** {date.strftime('%Y-%m-%d')}  |  **🧠 Model:** {model_name}  |  **🔄 Forecast:** {forecast_mode}")


tab1, tab2, = st.tabs(["🔮 Forecast", "📊 Dashboard"])

with tab1:
    # Predict Button
    if st.button("Predict"):
        try:
            # 1. Prepare model input
            df_region = df[df["Region"] == region]
            X_input, scaler = get_recent_sequence(df_region, region, date)

            st.text(f"Scaler expects: {scaler.n_features_in_} features")
            st.text(f"Model input shape: {X_input.shape}")

            # 2. Predict
            model = load_dl_model(model_name)
            result = model(X_input)

            # 3. Extract predicted value
            if isinstance(result, dict):
                output_array = list(result.values())[0]
            else:
                output_array = result

            predicted_value = output_array[0][0]  # shape (1, 1)

            # 4. Inverse transform using dummy 8-feature row
            rainfall_mm = inverse_transform_rainfall(predicted_value, scaler)

            # 5. Classify
            quantile_bins = get_rainfall_quantile_bins(df)
            rain_category = categorize_rain(rainfall_mm, quantile_bins)
            rain_label = CATEGORY_LABELS.get(rain_category, "Unknown")

            # Store results in session state
            st.session_state["rainfall_mm"] = rainfall_mm
            st.session_state["rain_category"] = rain_category
            st.session_state["rain_label"] = rain_label
            st.session_state["note"] = note

            # 6. Output — User-Friendly Messaging
            emoji_map = {
                0: "☀️",
                1: "🌤️",
                2: "🌧️",
                3: "🌧️🌧️",
                4: "⛈️⚠️"
            }

            guidance = {
                0: "No rainfall expected. It may be a good time for irrigation or dry-season crop activities.",
                1: "Light rainfall. Minor field impact — low risk for flooding or crop damage.",
                2: "Moderate rainfall. Normal wet-season conditions. Proceed with standard precautions.",
                3: "Heavy rainfall expected. Watch for flooding in low-lying areas. Consider drainage checks.",
                4: "Extreme rainfall! High risk of flooding. Secure equipment, monitor local alerts, delay planting if needed."
            }

            # Get icon and guidance
            icon = emoji_map.get(rain_category, "")
            note = guidance.get(rain_category, "No guidance available.")

            # Show output
            st.markdown(f"### {icon} Predicted Rainfall: **{rainfall_mm:.2f} mm**")
            st.markdown(f"**Category:** {rain_label} ({rain_category}) — Based on historical patterns")
            st.markdown(f"#### 📌 Guidance:\n{note}")
            st.caption(f"Quantile Thresholds (from training data): {quantile_bins}")

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")

with tab2:
    st.header("📊 Rainfall Dashboard")

    # Dummy METRICS section
    col1, col2, col3 = st.columns(3)
    # Use session_state or fallback values
    rainfall_mm = st.session_state.get("rainfall_mm", None)
    rain_label = st.session_state.get("rain_label", "—")
    rain_category = st.session_state.get("rain_category", None)
    note = st.session_state.get("note", "Run a forecast to generate guidance.")

    if rainfall_mm is not None:
        col1.metric("Predicted Rainfall", f"{rainfall_mm:.2f} mm", delta=None)
    else:
        col1.metric("Predicted Rainfall", "—", delta=None)

    col2.metric("Category", rain_label, delta=None)

    if rain_category is not None:
        risk_level = guidance.get(rain_category, "—").split('.')[0]
        col3.metric("Risk Level", risk_level)
    else:
        col3.metric("Risk Level", "—")

    st.divider()

    # LINE CHART (Recent Rainfall Trend)
    df_region_all = df[df["Region"] == region].sort_values("date").tail(100)
    st.subheader("📈 Recent Rainfall (r1h)")
    st.line_chart(data=df_region_all, x="date", y="r1h", use_container_width=True)

    st.divider()

    # DONUT CHART (Category Frequency)
    bins = get_rainfall_quantile_bins(df_region_all)
    df_region_all["rain_cat"] = df_region_all["r1h"].apply(lambda x: categorize_rain(x, bins))
    counts = df_region_all["rain_cat"].value_counts().sort_index()
    labels = [CATEGORY_LABELS[i] for i in counts.index]
    donut_data = pd.DataFrame({"Category": labels, "Frequency": counts.values})

    st.subheader("🍩 Rainfall Category Distribution")
    st.bar_chart(donut_data.set_index("Category"))

    st.divider()

    # Placeholder for future heatmap
    st.subheader("🟦 Heatmap (Coming Soon)")
    st.info("A visual heatmap of rainfall by week and day will appear here in the next version.")