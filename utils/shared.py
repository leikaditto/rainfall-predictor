import numpy as np

CATEGORY_LABELS = {
    0: "No Rain",
    1: "Light Rain",
    2: "Moderate Rain",
    3: "Heavy Rain",
    4: "Extreme Rain"
}

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

def get_rainfall_quantile_bins(df):
    bins = df['r1h'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
    return [0] + bins + [np.inf]