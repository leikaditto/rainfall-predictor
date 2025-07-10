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

def get_recent_sequence(df, region, target_date, window=10):
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['Region'] == region].sort_values('date')

    target_date = pd.to_datetime(target_date)
    df = df[df['date'] < target_date]
    seq = df.tail(window)

    if len(seq) < window:
        raise ValueError("Not enough past data available to create a sequence.")

    # Select the 8 features your DL model was trained on
    features = [
        'r1h', 'r3h', 'rfq', 'r1q', 'r3q',
        'is_wet_season', 'rfh_roll_mean_3', 'r1h_roll_std_5'
    ]
    
    rainfall_seq = seq[features]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(rainfall_seq)

    X_input = scaled.reshape(1, window, len(features))
    return X_input, scaler

def inverse_scale_prediction(pred, scaler):
    """
    Converts prediction back to original scale (mm).
    """
    pred = np.array(pred).reshape(-1, 1)
    return scaler.inverse_transform(pred)[0][0]