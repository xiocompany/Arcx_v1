# --- TESTING CODE ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# 2. Feature engineering (volume + price)
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

# 3. Labeling: anomaly if trend changes in future
def label_change_in_trend(df, future_windows=[15, 30, 60, 90]):
    df = df.copy()
    df['true_anomaly'] = 1  # default: normal (1), anomaly (-1)

    for w in future_windows:
        df[f'future_close_{w}'] = df['Close'].shift(-w)
        df[f'price_change_{w}'] = df[f'future_close_{w}'] - df['Close']
        df[f'price_direction_{w}'] = np.sign(df[f'price_change_{w}'])

    for w in future_windows:
        col = f'price_direction_{w}'
        df[f'trend_change_{w}'] = df[col] != df[col].shift(1)

    trend_cols = [f'trend_change_{w}' for w in future_windows]
    df['true_anomaly'] = df[trend_cols].any(axis=1).astype(int).replace({0: 1, 1: -1})

    print("✅ True anomaly labels created for test data.")
    return df

# 4. Remove leakage-prone features
def remove_leakage_features(all_features):
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_']
    clean_features = [f for f in all_features if not any(kw in f for kw in leakage_keywords)]
    return clean_features

# 6. Plot anomalies
def plot_anomalies(df, pred_col='xgb_pred'):
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df['Volume'], label='Volume')
    anomalies = df[df[pred_col] == -1]
    plt.scatter(anomalies['timestamp'], anomalies['Volume'], color='red', label='Anomaly', s=30)
    plt.title("Detected Anomalies on Test Data (XGBoost)")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === MAIN (TESTING) ===
if __name__ == "__main__":
    print("\n--- TESTING PHASE ---")
    # Specify the paths
    model_path = 'xgb_tuned_model.pkl'
    feature_list_path = 'xgb_feature_list.pkl'
    test_filepath = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_test_4pct.csv"  # Replace with your test data path

    # 1. Load the trained model
    try:
        loaded_model = joblib.load(model_path)
        print(f"✅ Model loaded from '{model_path}'")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{model_path}'. Make sure you have trained the model first.")
        exit()

    # 2. Load the feature list
    try:
        loaded_features = joblib.load(feature_list_path)
        print(f"✅ Feature list loaded from '{feature_list_path}'")
    except FileNotFoundError:
        print(f"❌ Error: Feature list file not found at '{feature_list_path}'.")
        exit()

    # 3. Load the test data
    df_test = load_data(test_filepath)
    print(f"✅ Loaded {len(df_test)} rows of test data")

    # 4. Preprocess the test data using the same feature engineering
    df_test = preprocess_features(df_test)
    print("✅ Features created for test data")

    # 5. Label anomalies in the test data
    df_test = label_change_in_trend(df_test)

    # 6. Remove leakage features from the test data features
    all_test_features = [col for col in df_test.columns if col.startswith('volume_') or col.startswith('price_')]
    valid_test_features = [col for col in all_test_features if df_test[col].isnull().mean() < 0.1]
    clean_test_features = remove_leakage_features(valid_test_features)

    # 7. Ensure the test data has the same features as the training data
    if not set(loaded_features).issubset(df_test.columns):
        missing_features = set(loaded_features) - set(df_test.columns)
        print(f"❌ Error: The following features are missing in the test data: {missing_features}")
        exit()

    # Use the loaded features directly to ensure correct columns and order
    features_for_prediction = loaded_features

    # 8. Prepare features and target variable for testing
    df_test_clean = df_test.dropna(subset=features_for_prediction + ['true_anomaly'])
    X_test = df_test_clean[features_for_prediction]
    y_test = df_test_clean['true_anomaly'].replace({-1: 1, 1: 0})

    # 9. Make predictions on the test data
    y_pred_test = loaded_model.predict(X_test)

    # 10. Evaluate the model on the test data
    print("\n=== Evaluation Metrics (Test Data) ===")
    print(f"Accuracy on Test Data: {accuracy_score(y_test, y_pred_test):.4f}")
    print("Classification Report on Test Data:")
    print(classification_report(y_test, y_pred_test, digits=4))
    print("Confusion Matrix on Test Data:")
    print(confusion_matrix(y_test, y_pred_test))

    # Optional: Plot anomalies on the test data
    df_test_clean['xgb_pred'] = y_pred_test
    df_test_clean['xgb_pred'] = df_test_clean['xgb_pred'].replace({0: 1, 1: -1})

    print("\n--- Anomalies on Test Data ---")
    plot_anomalies(df_test_clean)

    # 11. Save the processed test dataframe with predictions
    output_test_filepath = "processed_test_with_xgboost.csv"
    df_test_clean.to_csv(output_test_filepath, index=False)
    print(f"✅ Processed test data saved to '{output_test_filepath}'")