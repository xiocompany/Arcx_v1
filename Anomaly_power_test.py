import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# 2. Feature engineering (volume and price)
def preprocess_features(df, window_sizes=[1, 3, 5, 15, 30, 45, 60, 90]):
    df = df.copy()
    for w in window_sizes:
        # Rolling statistics for Volume
        df[f'volume_mean_{w}'] = df['Volume'].rolling(window=w, min_periods=1).mean()
        df[f'volume_std_{w}'] = df['Volume'].rolling(window=w, min_periods=1).std()
        df[f'volume_change_{w}'] = df['Volume'].diff(w)
        df[f'volume_ratio_{w}'] = df['Volume'] / (df[f'volume_mean_{w}'] + 1e-6)

        # Rolling statistics for Close price
        df[f'price_mean_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()
        df[f'price_std_{w}'] = df['Close'].rolling(window=w, min_periods=1).std()
        df[f'price_change_{w}'] = df['Close'].diff(w)
        df[f'price_ratio_{w}'] = df['Close'] / (df[f'price_mean_{w}'] + 1e-6)
    return df

# 3. Labeling: set anomaly if future trend changes using percentage price change
def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    df['true_anomaly'] = 1  # default: normal (1), anomaly (-1)
    
    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        df[f'price_change_{w}'] = ((df[f'future_close_{w}'] - df['Close']) / df['Close']) * 100
        df[f'price_direction_{w}'] = np.sign(df[f'price_change_{w}'])
    
    for w in future_windows:
        col = f'price_direction_{w}'
        df[f'trend_change_{w}'] = df[col] != df[col].shift(1)
    
    trend_cols = [f'trend_change_{w}' for w in future_windows]
    df['true_anomaly'] = df[trend_cols].any(axis=1).astype(int).replace({0: 1, 1: -1})
    
    print("True anomaly labels created.")
    return df

# 4. Remove leakage-prone features
def remove_leakage_features(all_features):
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_', '']
    clean_features = [f for f in all_features if not any(kw in f for kw in leakage_keywords)]
    return clean_features

if __name__ == "__main__":
    # Update with your new dataset file path
    new_data_filepath = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_test_4pct.csv"
    
    # Load new dataset
    df_new = load_data(new_data_filepath)
    df_new = preprocess_features(df_new)
    df_new = label_change_in_trend(df_new)
    
    # Calculate anomaly strength as the maximum absolute percentage price change
    future_windows = [15, 30, 60, 90]
    price_change_cols = [f'price_change_{w}' for w in future_windows]
    df_new['anomaly_strength'] = df_new[price_change_cols].abs().max(axis=1)
    
    # Load primary and meta models
    primary_model = joblib.load('xgb_tuned_model.pkl')
    primary_feature_list = joblib.load('xgb_feature_list.pkl')
    meta_model = joblib.load('xgb_meta_model.pkl')
    meta_feature_list = joblib.load('xgb_meta_feature_list.pkl')
    
    # Filter available features
    existing_features = [feat for feat in primary_feature_list if feat in df_new.columns]
    df_new_meta = df_new.dropna(subset=existing_features).copy()
    
    # Compute anomaly probability
    X_meta_base = df_new_meta[existing_features]
    meta_probs = primary_model.predict_proba(X_meta_base)[:, 1]
    df_new_meta['anomaly_prob'] = meta_probs
    
    # Prepare meta features
    existing_meta_features = [feat for feat in meta_feature_list if feat in df_new_meta.columns]
    X_meta_new = df_new_meta[existing_meta_features]
    
    # Predict anomaly strength
    predicted_anomaly_strength = meta_model.predict(X_meta_new)
    df_new_meta['predicted_anomaly_strength'] = predicted_anomaly_strength
    
    # Evaluate
    mse = mean_squared_error(df_new_meta['anomaly_strength'], df_new_meta['predicted_anomaly_strength'])
    r2 = r2_score(df_new_meta['anomaly_strength'], df_new_meta['predicted_anomaly_strength'])
    
    print("=== New Data Performance Metrics ===")
    print("MSE: {:.4f}".format(mse))
    print("R2 Score: {:.4f}".format(r2))
    
    # Display a few predictions
    print("Predictions on new dataset:")
    print(df_new_meta[['timestamp', 'anomaly_strength', 'predicted_anomaly_strength']].head())
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(df_new_meta['timestamp'], df_new_meta['anomaly_strength'], label='Actual Anomaly Strength', marker='o')
    plt.plot(df_new_meta['timestamp'], df_new_meta['predicted_anomaly_strength'], label='Predicted Anomaly Strength', marker='x')
    plt.title('Actual vs Predicted Anomaly Strength on New Dataset')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Strength (%)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ✅ Save true vs predicted values to CSV
    df_new_meta[['timestamp', 'anomaly_strength', 'predicted_anomaly_strength']].to_csv(
        'anomaly_strength_predictions.csv', index=False
    )
    print("Results saved to 'anomaly_strength_predictions.csv'")
