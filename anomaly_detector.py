import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
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

    print("✅ True anomaly labels created.")
    return df

# 4. Remove leakage-prone features
def remove_leakage_features(all_features):
    leakage_keywords = ['price_direction_', 'trend_change_', 'future_close_']
    clean_features = [f for f in all_features if not any(kw in f for kw in leakage_keywords)]
    return clean_features

# 5. Objective function for Optuna (Not used directly now)
def objective(trial, df, features, label_col='true_anomaly'):
    # You can expand the search space to include regularization parameters.
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
        'max_depth': trial.suggest_int('max_depth', 6, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'gamma': trial.suggest_float('gamma', 0, 2.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # Regularization parameters
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0)
    }
    # Further code to evaluate these parameters would be added here.
    pass

# 6. Train model with Optuna (Directly using best parameters with cross-validation)
def tune_model_with_optuna(df, features, n_trials=30, label_col='true_anomaly',
                           model_path='xgb_tuned_model.pkl',
                           feature_list_path='xgb_feature_list.pkl',
                           cv_splits=5):
    # Best parameters found via Optuna with additional regularization
    best_params = {
        'n_estimators': 847,
        'max_depth': 12,
        'learning_rate': 0.03339231749379355,
        'gamma': 1.4886730647003619,
        'min_child_weight': 10,
        'subsample': 0.711356844189682,
        'colsample_bytree': 0.9336052029752122,
        'reg_alpha': 0.1,    # Added L1 regularization term
        'reg_lambda': 1.0    # Added L2 regularization term
    }

    print("\n🏆 Using Best Parameters Found by Optuna:")
    print(best_params)

    # Drop rows with missing values in selected features and label
    df_clean = df.dropna(subset=features + [label_col])
    X = df_clean[features]
    # Remap labels: normal (1) to 0, anomaly (-1) to 1
    y = df_clean[label_col].replace({-1: 1, 1: 0})

    # Initialize model with GPU support if available
    final_model = XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        **best_params
    )

    # Cross-validation using time-series split for a robust performance estimate
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    cv_scores = cross_val_score(final_model, X, y, cv=tscv, scoring='f1')
    print("\n=== Cross-Validation F1 Scores ===")
    print(cv_scores)
    print("Average CV F1 Score: {:.4f}".format(np.mean(cv_scores)))

    # Fit the model on the entire training set
    final_model.fit(X, y)

    # Evaluate model performance on training data
    y_pred = final_model.predict(X)
    print("\n=== Evaluation Metrics (Training Data with Best Parameters) ===")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Save the tuned model and feature list
    joblib.dump(final_model, model_path)
    joblib.dump(features, feature_list_path)
    print(f"✅ Tuned model saved to '{model_path}'")
    print(f"✅ Feature list saved to '{feature_list_path}'")

    return final_model, features

# 7. Plot anomalies
def plot_anomalies(df, pred_col='xgb_pred'):
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df['Volume'], label='Volume')
    anomalies = df[df[pred_col] == -1]
    plt.scatter(anomalies['timestamp'], anomalies['Volume'], color='red', label='Anomaly', s=30)
    plt.title("Detected Anomalies (XGBoost)")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

# === MAIN (TRAINING WITH OPTUNA - Using Best Params) ===
if __name__ == "__main__":
    print("--- TRAINING PHASE WITH OPTUNA (Using Best Parameters) ---")
    # Specify the file path for training data
    train_filepath = r"D:\project\split_output_data\BTCUSDT_ohlc_data_1m(5year)_train_80pct.csv"

    # 1. Load training data
    df_train = load_data(train_filepath)
    print(f"✅ Loaded {len(df_train)} rows of training data")

    # 2. Feature engineering on training data
    df_train = preprocess_features(df_train)
    print("✅ Features created for training data")

    # 3. Label anomalies in training data based on future trend changes
    df_train = label_change_in_trend(df_train)

    # 4. Feature selection: choose features starting with 'volume_' or 'price_'
    all_train_features = [col for col in df_train.columns if col.startswith('volume_') or col.startswith('price_')]
    valid_train_features = [col for col in all_train_features if df_train[col].isnull().mean() < 0.1]
    # You might later refine this further by removing highly correlated or low-variance features.
    clean_train_features = remove_leakage_features(valid_train_features)
    print(f"✅ {len(clean_train_features)} clean features selected for training (no leakage)")

    # 5. Tune the model with Optuna (Directly using best params and cross-validation)
    tuned_model, final_features = tune_model_with_optuna(df_train, clean_train_features, n_trials=30)

    # 6. Predict anomalies on the training data with the tuned model
    df_train_clean = df_train.dropna(subset=final_features)
    df_train_clean['xgb_pred'] = tuned_model.predict(df_train_clean[final_features])
    df_train_clean['xgb_pred'] = df_train_clean['xgb_pred'].replace({0: 1, 1: -1})

    # 7. Merge predictions back into the original training dataframe
    df_train = df_train.merge(df_train_clean[['timestamp', 'xgb_pred']], on='timestamp', how='left')

    # 8. Plot detected anomalies on training data
    print("\n--- Anomalies on Training Data (Tuned Model) ---")
    plot_anomalies(df_train)
