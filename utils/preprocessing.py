# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def load_dataset(file_path):
    """
    Loads the rainfall dataset from Excel.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not Found at {file_path}")
    
    df = pd.read_excel(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_recent_sequence(df, region, target_date, window=12):
    """
    Prepares the input sequence for prediction.

    Args:
        df: full dataframe
        region: region name (string)
        target_date: prediction target date (string: 'YYYY-MM-DD')
        window: number of past time steps to use
    
    Returns:
        X_input: model-ready array
        scaler: fitted MinMaxScaler
    """
    # Filter for the region
    df_region = df[df['Region'] == region].sort_values('date')

    # Covert to datetime
    target_date = pd.to_datetime(target_date)

    # Filter data before target date
    df_past = df_region[df_region['date'] < target_date].tail(window)

    if len(df_past) < window:
        raise ValueError(f"Not enough part data for region '{region}' before {target_date.strftime('%Y-%m-%d')}")
    
    # Extract rainfall column
    values = df_past['rfh'].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Reshape for LSTM/CNN/GRU input: [1, time steps, features]
    X_input = scaled_values.reshape((1, window, 1))

    return X_input, scaler

def inverse_scale_prediction(pred, scaler):
    """
    Converts prediction back to original scale (mm).
    """
    pred = np.array(pred).reshape(-1, 1)
    return scaler.inverse_transform(pred)[0][0]