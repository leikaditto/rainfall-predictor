# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from utils.preprocessing import load_dataset
from utils.shared import categorize_rain, CATEGORY_LABELS, get_rainfall_quantile_bins

@st.cache_data
def load_dashboard_data():
    return load_dataset("data/Ph-Rainfall-DL.xlsx")

def show_dashboard():
    st.title("📊 Rainfall Forecast Dashboard")

    df = load_dashboard_data()
    regions = sorted(df["Region"].dropna().unique())
    region = st.selectbox("Select Region", regions, key="dashboard_region")

    # Filter
    df_region = df[df["Region"] == region].copy()
    quantile_bins = get_rainfall_quantile_bins(df_region)
    df_region["Category"] = df_region["r1h"].apply(lambda x: categorize_rain(x, quantile_bins))
    df_region["Label"] = df_region["Category"].map(CATEGORY_LABELS)

    # Visual 1: Line chart (Rainfall over time)
    st.subheader("📈 Historical Rainfall Trend")
    df_region["Date"] = pd.to_datetime(df_region["date"], errors='coerce')
    df_line = df_region.dropna(subset=["Date"])[["Date", "r1h"]].sort_values("Date")

    line = alt.Chart(df_line).mark_line().encode(
        x="Date:T",
        y=alt.Y("r1h:Q", title="Rainfall (mm)"),
        tooltip=["Date:T", "r1h"]
    ).properties(width=700, height=300)
    st.altair_chart(line, use_container_width=True)

    # Visual 2: Bar chart of rainfall category distribution
    st.subheader("📊 Rainfall Category Frequency")
    category_counts = df_region["Label"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]

    bar = alt.Chart(category_counts).mark_bar().encode(
        x="Category:N",
        y="Count:Q",
        color="Category:N",
        tooltip=["Category", "Count"]
    ).properties(width=500)
    st.altair_chart(bar, use_container_width=True)

    # Table of recent forecasts
    st.subheader("📋 Recent Observations (10-day)")
    st.dataframe(df_line.tail(10).rename(columns={"r1h": "Rainfall (mm)"}))