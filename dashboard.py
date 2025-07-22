# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from utils.preprocessing import load_dataset
from app import categorize_rain, CATEGORY_LABELS, get_rainfall_quantile_bins

@st.cache_data
def load_dashboard_data():
    return load_dataset("data/Ph-Rainfall-DL.xlsx")

def show_dashboard():
    st.subheader("📈 Regional Rainfall Trends & Summary")

    df_region = df[df["Region"] == region].copy()

    # Use recent 90 days for visual clarity
    df_region = df_region.sort_values("date").tail(90)

    # Plot 1: Line Chart of r1h (1-month rainfall)
    st.markdown("**Rainfall Over Time (r1h)**")
    fig1, ax1 = plt.subplots()
    ax1.plot(df_region["date"], df_region["r1h"], marker='o', linestyle='-', color='blue')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("1-Month Rainfall (mm)")
    ax1.set_title(f"Rainfall Trend for {region}")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Plot 2: Rainfall Category Distribution
    st.markdown("**Rainfall Category Distribution**")
    bins = get_rainfall_quantile_bins(df)
    df_region["category"] = df_region["r1h"].apply(lambda x: CATEGORY_LABELS[categorize_rain(x, bins)])
    counts = df_region["category"].value_counts().reindex(CATEGORY_LABELS.values(), fill_value=0)
    st.bar_chart(counts)

    # Table: Last 10 Days
    st.markdown("**🔍 Recent 10-Day Summary**")
    st.dataframe(df_region[["date", "r1h", "category"]].tail(10).reset_index(drop=True))