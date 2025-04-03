import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor

# Load and preprocess the data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

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

def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        df[f'price_change_{w}'] = ((df[f'future_close_{w}'] - df['Close']) / df['Close']) * 100
        df[f'price_direction_{w}'] = np.sign(df[f'price_change_{w}'])
    for w in future_windows:
        col = f'price_direction_{w}'
        df[f'trend_change_{w}'] = df[col] != df[col].shift(1)
    trend_cols = [f'trend_change_{w}' for w in future_windows]
    df['true_anomaly'] = df[trend_cols].any(axis=1).astype(int).replace({0: 1, 1: -1})
    return df

# Load models and features
clf_model = joblib.load("xgb_tuned_model.pkl")
clf_features = joblib.load("xgb_feature_list.pkl")
reg_model = joblib.load("xgb_meta_model.pkl")
reg_features = joblib.load("xgb_meta_feature_list.pkl")

# Load new data file (example, same as earlier)
data_path = r'D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_test_4pct.csv'
df_new = load_data(data_path)
df_new = preprocess_features(df_new)
df_new = label_change_in_trend(df_new)

# Ensure no NaNs
clf_features_exist = [f for f in clf_features if f in df_new.columns]
df_filtered = df_new.dropna(subset=clf_features_exist).copy()

# Classification
X_clf = df_filtered[clf_features_exist]
clf_preds = clf_model.predict(X_clf)
df_filtered["anomaly_label"] = clf_preds
df_filtered["anomaly_label"] = df_filtered["anomaly_label"].replace({0: 1, 1: -1})  # match labeling

# Probability for regression model
df_filtered["anomaly_prob"] = clf_model.predict_proba(X_clf)[:, 1]

# Regression
reg_features_exist = [f for f in reg_features if f in df_filtered.columns]
X_reg = df_filtered[reg_features_exist]
reg_preds = reg_model.predict(X_reg)
df_filtered["anomaly_strength"] = reg_preds

# Final output
final_df = df_filtered[["timestamp", "Close", "anomaly_label", "anomaly_strength"]]
final_df.to_csv("final_anomaly_results.csv", index=False)