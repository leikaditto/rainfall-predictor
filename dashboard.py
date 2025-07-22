# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing import load_dataset
from utils.shared import categorize_rain, CATEGORY_LABELS, get_rainfall_quantile_bins

@st.cache_data
def load_dashboard_data():
    return load_dataset("data/Ph-Rainfall-DL.xlsx")

def show_dashboard():
    st.title("📊 Rainfall Forecast Dashboard")

    df = load_dashboard_data()
    quantile_bins = get_rainfall_quantile_bins(df)

    # Simulate 30-day prediction for visualization
    dates = pd.date_range(start="2025-12-01", periods=30, freq="D")
    rainfall = np.random.normal(loc=5, scale=1.2, size=30)
    rainfall = np.clip(rainfall, 0, None)  # no negative rain
    categories = [categorize_rain(val, quantile_bins) for val in rainfall]
    labels = [CATEGORY_LABELS[cat] for cat in categories]

    forecast_df = pd.DataFrame({
        "Date": dates,
        "Rainfall (mm)": rainfall,
        "Category": labels
    })

    # Chart 1: Line + Dot forecast
    st.subheader("📅 30-Day Rainfall Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecast_df["Date"], forecast_df["Rainfall (mm)"], color="orange", marker="o")
    ax.set_title("📉 30-Day Rainfall Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall (mm)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # Chart 2: Rainfall Category Frequency
    st.subheader("📊 Rainfall Category Frequency")
    cat_freq = forecast_df["Category"].value_counts().reindex(CATEGORY_LABELS.values(), fill_value=0)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(cat_freq.index, cat_freq.values, color="skyblue")
    ax2.set_ylabel("Days")
    ax2.set_xlabel("Rainfall Category")
    ax2.set_title("📊 Rainfall Category Frequency")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    # Optional Table View
    st.dataframe(forecast_df)
