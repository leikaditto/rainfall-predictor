# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

from app import inverse_transform_rainfall

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

def recursive_forecast(model, initial_sequence, scaler, steps=30):
    """
    Simulate a 30-day forecast by rolling forward the predicted value.
    """
    sequence = initial_sequence.copy()
    predictions = []

    for _ in range(steps):
        result = model(sequence)

        # Get predictied normalized value
        if isinstance(result, dict):
            predicted = list(result.values())[0][0][0]
        else:
            predicted = result[0][0]
        
        # Inverse transform predicted rainfall
        rainfall_mm = inverse_transform_rainfall(predicted, scaler)

        predictions.append(rainfall_mm)

        # prepare next sequence
        next_input = np.zeros((1, sequence.shape[1], sequence.shape[2]))
        next_input[0, :-1, :] = sequence[0, 1:, :] # shift window left
        next_input[0, -1, 0] = predicted # insert new prediction
        sequence = next_input

    return predicted