# utils/classification_preprocessing.py

# import pandas as pd
# import datetime

# def load_classification_data(path):
#     """Load data prepared for classification (Tree-based model)."""
#     df = pd.read_excel(path)
#     df["date"] = pd.to_datetime(df["date"])
#     return df

# def prepare_classification_input(df, region, date, feature_columns):
#     """Prepare a single row of input for the classification model."""
#     date = pd.to_datetime(date)
#     df_region = df[df["Region"] == region]
    
#     if df_region.empty:
#         raise ValueError("No data available before selected date.")
    
#     # Use the most recent row before the target date
#     latest_row = df_region.sort_values("date").iloc[-1]
    
#     # Extract required features
#     input_data = latest_row[feature_columns].values.reshape(1, -1)
#     return input_data