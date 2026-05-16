import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import entropy

def calculate_entropy(x):
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    if np.all(values == values[0]):
        return 0.0
    hist, _ = np.histogram(values, bins=min(10, len(np.unique(values))))
    probs = hist[hist > 0] / hist.sum()
    return entropy(probs)

def preprocess_data(df):
    df = df.sort_values(['patient_id', 'hour'])
    
    # 1. Forward fill vitals (limit 2)
    vitals = ['HR', 'BP', 'SpO2', 'respiration', 'temp']
    df[vitals] = df.groupby('patient_id')[vitals].ffill(limit=2)
    
    # 2. KNN impute labs
    labs = ['creatinine', 'WBC', 'lactate']
    imputer = KNNImputer(n_neighbors=5)
    df[labs] = imputer.fit_transform(df[labs])
    
    # 3. 6-hour rolling windows
    features = vitals + labs
    rolling_df = df.groupby('patient_id')[features].rolling(window=6, min_periods=1)
    
    df_mean = rolling_df.mean().reset_index(level=0, drop=True).add_suffix('_mean')
    df_std = rolling_df.std().reset_index(level=0, drop=True).add_suffix('_std')  # Volatility
    df_min = rolling_df.min().reset_index(level=0, drop=True).add_suffix('_min')
    df_max = rolling_df.max().reset_index(level=0, drop=True).add_suffix('_max')
    
    def get_slope(x):
        if len(x) < 2: return 0
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    
    def get_accel(x):
        if len(x) < 3: return 0
        return np.polyfit(np.arange(len(x)), x, 2)[0]

    # Rolling slopes (3h and 6h)
    df_slope_3h = df.groupby('patient_id')[features].rolling(window=3, min_periods=2).apply(get_slope).reset_index(level=0, drop=True).add_suffix('_slope_3h')
    df_slope_6h = df.groupby('patient_id')[features].rolling(window=6, min_periods=2).apply(get_slope).reset_index(level=0, drop=True).add_suffix('_slope_6h')
    
    df_accel = df.groupby('patient_id')[features].rolling(window=6, min_periods=3).apply(get_accel).reset_index(level=0, drop=True).add_suffix('_accel')
    df_entropy = df.groupby('patient_id')[features].rolling(window=6, min_periods=2).apply(calculate_entropy).reset_index(level=0, drop=True).add_suffix('_entropy')
    
    df = pd.concat([df, df_mean, df_std, df_min, df_max, df_slope_3h, df_slope_6h, df_accel, df_entropy], axis=1)
    return df.fillna(0)
