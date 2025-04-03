import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import joblib
import optuna

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

# 3. Labeling: anomaly if trend changes in future, with direction
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

    print("✅ True anomaly labels with direction created.")
    return df

# 4. Remove leakage-prone features
def remove_leakage_features(all_features):
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_']
    clean_features = [f for f in all_features if not any(kw in f for kw in leakage_keywords)]
    return clean_features

# 5. Train model using best parameters from Optuna (3-class classification)
def tune_model_with_optuna(df, features, label_col='true_anomaly',
                           model_path='xgb_direct_model.pkl',
                           feature_list_path='xgb_feature_list_direct.pkl'):
    best_params = {
        'n_estimators': 847,
        'max_depth': 12,
        'learning_rate': 0.03339231749379355,
        'gamma': 1.4886730647003619,
        'min_child_weight': 10,
        'subsample': 0.711356844189682,
        'colsample_bytree': 0.9336052029752122,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }

    print("\n🏆 Using Best Parameters Found by Optuna:")
    print(best_params)

    df_clean = df.dropna(subset=features + [label_col])
    X = df_clean[features]
    label_map = {-1: 0, 0: 1, 1: 2}  # 0=neg anomaly, 1=normal, 2=pos anomaly
    y = df_clean[label_col].map(label_map)

    final_model = XGBClassifier(
        objective="multi:softmax",
        use_label_encoder=False,
        eval_metric="mlogloss",
        num_class=3,
        random_state=42,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        **best_params
    )

    final_model.fit(X, y)
    y_pred = final_model.predict(X)
    print("\n=== Evaluation Metrics (Training Data with Best Parameters) ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    joblib.dump(final_model, model_path)
    joblib.dump(features, feature_list_path)
    print(f"✅ Tuned model saved to '{model_path}'")
    print(f"✅ Feature list saved to '{feature_list_path}'")

    return final_model, features, label_map

# 6. Plot anomalies
def plot_anomalies(df, pred_col='xgb_pred'):
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df['Volume'], label='Volume')
    plt.scatter(df[df[pred_col] == 0]['timestamp'], df[df[pred_col] == 0]['Volume'], color='red', label='Neg Anomaly', s=30)
    plt.scatter(df[df[pred_col] == 2]['timestamp'], df[df[pred_col] == 2]['Volume'], color='green', label='Pos Anomaly', s=30)
    plt.title("Detected Anomalies with Direction (XGBoost)")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":
    print("--- TRAINING PHASE WITH DIRECTIONAL ANOMALY DETECTION ---")
    train_filepath = r"D:\\project\\split_output_data\\BTCUSDT_ohlc_data_1m(5year)_train_80pct.csv"

    df_train = load_data(train_filepath)
    print(f"✅ Loaded {len(df_train)} rows of training data")

    df_train = preprocess_features(df_train)
    print("✅ Features created for training data")

    df_train = label_change_in_trend(df_train)

    all_train_features = [col for col in df_train.columns if col.startswith('volume_') or col.startswith('price_')]
    valid_train_features = [col for col in all_train_features if df_train[col].isnull().mean() < 0.1]
    clean_train_features = remove_leakage_features(valid_train_features)
    print(f"✅ {len(clean_train_features)} clean features selected for training (no leakage)")

    tuned_model, final_features, label_map = tune_model_with_optuna(df_train, clean_train_features)

    df_train_clean = df_train.dropna(subset=final_features)
    df_train_clean['xgb_pred'] = tuned_model.predict(df_train_clean[final_features])

    df_train = df_train.merge(df_train_clean[['timestamp', 'xgb_pred']], on='timestamp', how='left')

    print("\n--- Anomalies on Training Data (Tuned Model) ---")
    plot_anomalies(df_train)
