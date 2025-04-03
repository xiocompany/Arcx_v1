import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

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
        df[f'price_change_{w}'] = df['Close'].diff(w)  # This will be replaced in labeling with percentage change
        df[f'price_ratio_{w}'] = df['Close'] / (df[f'price_mean_{w}'] + 1e-6)
    return df

# 3. Labeling: set anomaly if future trend changes with percentage price change
def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    df['true_anomaly'] = 1  # default: normal (1), anomaly (-1)

    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        # Calculate percentage price change: ((future - current) / current) * 100
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
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_','price_change_']
    clean_features = [f for f in all_features if not any(kw in f for kw in leakage_keywords)]
    return clean_features

# Main execution
# 1. Load main data (for example, test data)
train_filepath = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_train_80pct.csv"
df = load_data(train_filepath)

# 2. Apply preprocessing and create initial features
df = preprocess_features(df)
df = label_change_in_trend(df)

# 3. Feature selection (same as the primary model)
all_features = [col for col in df.columns if col.startswith('volume_') or col.startswith('price_')]
valid_features = [col for col in all_features if df[col].isnull().mean() < 0.1]
clean_features = remove_leakage_features(valid_features)
print(f"{len(clean_features)} clean features selected for the primary model.")

# 4. Calculate meta target: anomaly strength defined as the maximum absolute percentage price change
future_windows = [15, 30, 60, 90]
price_change_cols = [f'price_change_{w}' for w in future_windows]
df['anomaly_strength'] = df[price_change_cols].abs().max(axis=1)

# 5. Load the primary model and its feature list
tuned_model = joblib.load('xgb_tuned_model.pkl')
feature_list = joblib.load('xgb_feature_list.pkl')

# 6. Drop rows with missing values in the selected features and meta target
df_meta = df.dropna(subset=feature_list + ['anomaly_strength']).copy()

# 7. Extract meta features: use the primary model's anomaly probability as a meta feature.
# Note: The primary model was trained for binary classification (0: normal, 1: anomaly).
X_meta_base = df_meta[feature_list]
meta_probs = tuned_model.predict_proba(X_meta_base)[:, 1]
df_meta['anomaly_prob'] = meta_probs

# Additional meta features can be added here.
meta_features = feature_list + ['anomaly_prob']

# 8. Set the meta target variable (anomaly strength)
y_meta = df_meta['anomaly_strength']

# 9. Split data into training and testing sets (preserving temporal order)
X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
    df_meta[meta_features], y_meta, test_size=0.2, random_state=42, shuffle=False
)

# 10. Build and train the regression model (using XGBRegressor) to predict anomaly strength
meta_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    tree_method='hist',     # Use hist tree method
    device='cuda'           # Use GPU if available
)
meta_model.fit(X_train_meta, y_train_meta)

# 11. Evaluate the meta model
y_pred_meta = meta_model.predict(X_test_meta)
mse = mean_squared_error(y_test_meta, y_pred_meta)
r2 = r2_score(y_test_meta, y_pred_meta)
print("=== Meta Model Performance ===")
print("MSE: {:.4f}".format(mse))
print("R2 Score: {:.4f}".format(r2))

# 12. Plot Actual vs Predicted Anomaly Strength (Line Plot)
# Retrieve timestamps for the test set (assuming 'timestamp' is still available)
test_timestamps = df_meta.loc[X_test_meta.index, 'timestamp']
plt.figure(figsize=(12, 6))
plt.plot(test_timestamps, y_test_meta, label='Actual Anomaly Strength', marker='o')
plt.plot(test_timestamps, y_pred_meta, label='Predicted Anomaly Strength', marker='x')
plt.title('Actual vs Predicted Anomaly Strength')
plt.xlabel('Timestamp')
plt.ylabel('Anomaly Strength (%)')
plt.legend()
plt.tight_layout()
plt.show()

# 13. Save the meta model and its feature list
joblib.dump(meta_model, 'xgb_meta_model.pkl')
joblib.dump(meta_features, 'xgb_meta_feature_list.pkl')
print("Meta model and feature list saved.")
