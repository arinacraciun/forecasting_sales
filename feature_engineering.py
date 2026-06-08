import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest

def add_anomaly_flags(df):
    """Uses Isolation Forest to flag multivariate anomalies as a model feature."""
    print("Generating Isolation Forest anomaly features...")
    
    # We use historical columns available at the time of prediction
    features = ['sales_roll_mean_7', 'sales_roll_std_7', 'onpromotion']
    
    # Temporarily fill NaNs in rolling features for the forest
    temp_df = df[features].fillna(0)
    
    iso_forest = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
    
    # Fit and predict. Convert -1 (anomaly) and 1 (normal) to 1 and 0.
    predictions = iso_forest.fit_predict(temp_df)
    df['is_anomaly_profile'] = (predictions == -1).astype(int)
    
    return df

def add_fourier_terms(df, date_col, period=365.25, order=3):
    t = df[date_col].dt.dayofyear
    for k in range(1, order + 1):
        df[f'sin_{period}_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'cos_{period}_{k}'] = np.cos(2 * np.pi * k * t / period)
    return df

def engineer_features(input_path='./data/processed/train_processed.parquet', output_path='./data/processed/features_ready.parquet'):
    print("Loading processed data...")
    df = pd.read_parquet(input_path)
    df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

    HORIZON = 16 

    print("Generating temporal features...")
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month    
    df['year'] = df['date'].dt.year   

    # Expand Calendar Features
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days
    df = add_fourier_terms(df, 'date')

    print("Generating time delay features (Lags & Rolling)...")
    grouped = df.groupby(['store_nbr', 'family'])['sales']
    
    # Direct Lags
    for lag_offset in [0, 7, 14, 23, 30, 349]: # Added 23 and 30 for targeted multi-week seasonal offsets
        df[f'sales_lag_{HORIZON + lag_offset}'] = grouped.shift(HORIZON + lag_offset)

    # Rolling Aggregations (Shifted by Horizon first)
    shifted_sales = grouped.shift(HORIZON)
    for window in [7, 14, 28]:
        df[f'sales_roll_mean_{window}'] = shifted_sales.rolling(window=window, min_periods=1).mean()
        df[f'sales_roll_std_{window}'] = shifted_sales.rolling(window=window, min_periods=1).std().fillna(0)

    # Exponentially Weighted Moving Average (EWMA)
    for span in [7, 28]:
        df[f'sales_ewma_{span}'] = shifted_sales.ewm(span=span, min_periods=1).mean()

    # Exogenous
    df['oil_lag_16'] = df.groupby('store_nbr')['oil_price'].shift(HORIZON)

    # Create rolling anomaly features
    df = add_anomaly_flags(df)

    print("Cleaning and Encoding...")
    df = df.dropna().reset_index(drop=True)

    categorical_cols = ['family', 'city', 'state', 'store_type', 'holiday_type', 'locale']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    df.to_parquet(output_path)
    print(f"Feature engineering complete. Saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    engineer_features()