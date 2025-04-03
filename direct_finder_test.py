import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Load Data ---
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# --- Preprocess Features ---
def preprocess_features(df, window_sizes=[1, 3, 5, 15, 30, 45, 60, 90]):
    df = df.copy()
    for w in window_sizes:
        df[f'volume_mean_{w}'] = df['Volume'].rolling(window=w, min_periods=1).mean()
        df[f'volume_std_{w}'] = df['Volume'].rolling(window=w, min_periods=1).std()
        df[f'volume_change_{w}'] = df['Volume'].diff(w)
        df[f'volume_ratio_{w}'] = df['Volume'] / (df[f'volume_mean_{w}'] + 1e-6)

        df[f'price_mean_{w}'] = df['Close'].rolling(window=w, min_periods=1).mean()
        df[f'price_std_{w}'] = df['Close'].rolling(window=w, min_periods=1).std()
        df[f'price_change_{w}'] = df['Close'].diff(w)
        df[f'price_ratio_{w}'] = df['Close'] / (df[f'price_mean_{w}'] + 1e-6)
    return df

# --- Label future trend direction (for test accuracy calculation) ---
def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        df[f'price_change_{w}'] = df[f'future_close_{w}'] - df['Close']
        df[f'price_direction_{w}'] = np.sign(df[f'price_change_{w}'])
    for w in future_windows:
        col = f'price_direction_{w}'
        df[f'trend_change_{w}'] = df[col] != df[col].shift(1)
    trend_cols = [f'trend_change_{w}' for w in future_windows]
    df['is_anomaly'] = df[trend_cols].any(axis=1).astype(int)
    direction_cols = [f'price_direction_{w}' for w in future_windows]
    df['avg_direction'] = df[direction_cols].mean(axis=1)
    df['true_anomaly'] = 0
    df.loc[df['is_anomaly'] == 1, 'true_anomaly'] = np.sign(df.loc[df['is_anomaly'] == 1, 'avg_direction'])
    return df

# --- Main Evaluation ---
def evaluate_model(test_filepath, model_path=r'D:\project\xgb_direct_model.pkl', feature_list_path=r'D:\project\xgb_feature_list_direct.pkl'):
    df_test = load_data(test_filepath)
    print(f"✅ Loaded {len(df_test)} rows of test data")

    df_test = preprocess_features(df_test)
    df_test = label_change_in_trend(df_test)

    model = joblib.load(model_path)
    features = joblib.load(feature_list_path)

    df_test_clean = df_test.dropna(subset=features + ['true_anomaly'])
    X_test = df_test_clean[features]
    y_true = df_test_clean['true_anomaly'].map({-1: 0, 0: 1, 1: 2})  # same label mapping used during training

    y_pred = model.predict(X_test)

    print("\n=== Evaluation Metrics on Test Data ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    test_filepath = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_test_4pct.csv"
    evaluate_model(test_filepath)
