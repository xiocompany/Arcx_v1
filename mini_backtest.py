#!/usr/bin/env python
"""
live_prediction.py
------------------
Read `chart_data.csv`, keep a rolling history, build features on-the-fly,
and output per-bar predictions:

YYYY-MM-DD HH:MM:SS → Anomaly:+1/-1 (p=0.xxx) | Power: xx.xx% | Direction:+1/0/-1
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import warnings # Import the warnings module

# Ignore the specific FutureWarning related to downcasting in fillna/ffill/bfill
# This warning is addressed by using .infer_objects(copy=False) later in the code.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated"
)

# ─────────────────────────── model & feature assets ──────────────────────────
# Load pre-trained models and feature lists
# Ensure these .pkl files are in the same directory as the script, or provide full paths.
try:
    an_clf    = joblib.load(r"D:\project\Arcx_v1\xgb_model.pkl")
    an_feats  = joblib.load(r"D:\project\Arcx_v1\xgb_feature_list.pkl")

    pw_reg    = joblib.load("xgb_meta_model.pkl")
    pw_feats  = joblib.load("xgb_meta_feature_list.pkl")

    dir_clf   = joblib.load("xgb_direct_model.pkl")
    dir_feats = joblib.load("xgb_feature_list_direct.pkl")
except FileNotFoundError as e:
    print(f"Error loading model file: {e}. Make sure all .pkl files are present.")
    exit()

label_rev = {0: -1, 1: 0, 2: 1} # Mapping for direction model output

# Power model always needs anomaly_prob
if "anomaly_prob" not in pw_feats:
    pw_feats.append("anomaly_prob")

# Force all boosters to CPU prediction (avoids GPU mismatch warnings if models were trained on GPU)
# This ensures the script runs correctly even on machines without a GPU or with different GPU setups.
try:
    for booster in (an_clf.get_booster(),
                    pw_reg.get_booster(),
                    dir_clf.get_booster()):
        booster.set_param({"device": "cpu", "predictor": "cpu_predictor"})
except AttributeError:
    # This might happen if the loaded objects are not XGBoost models (e.g., scikit-learn wrappers)
    # In such cases, direct booster manipulation might not be needed or possible.
    # Depending on the actual model types, alternative ways to set CPU usage might be required.
    print("Note: Could not set 'device' and 'predictor' params directly on boosters. "
          "If using scikit-learn wrappers for XGBoost, CPU usage is typically default.")


# ─────────────────────────── feature construction ────────────────────────────
WINDOWS       = [1, 3, 5, 15, 30, 45, 60, 90] # Rolling window sizes for feature calculation
KEEP_LAST_N   = 5000 # Number of recent data points to keep in history
WARMUP_ROWS   = max(WINDOWS)  # Minimum number of rows needed before making predictions (90 in this case)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling features from the input DataFrame.
    Ensures no NaNs in the output by forward-filling and then back-filling.
    """
    out = df.copy()

    # Iterate through specified window sizes to create various rolling features
    for w in WINDOWS:
        # Volume-based features
        out[f'volume_mean_{w}']   = out['Volume'].rolling(w, min_periods=1).mean()
        out[f'volume_std_{w}']    = out['Volume'].rolling(w, min_periods=1).std()
        out[f'volume_change_{w}'] = out['Volume'].diff(w) # Difference from w periods ago
        out[f'volume_ratio_{w}']  = out['Volume'] / (out[f'volume_mean_{w}'] + 1e-6) # Ratio to mean, 1e-6 to avoid division by zero

        # Price-based features (using 'Close' price)
        out[f'price_mean_{w}']    = out['Close'].rolling(w, min_periods=1).mean()
        out[f'price_std_{w}']     = out['Close'].rolling(w, min_periods=1).std()
        out[f'price_change_{w}']  = out['Close'].diff(w) # Difference from w periods ago
        out[f'price_ratio_{w}']   = out['Close'] / (out[f'price_mean_{w}'] + 1e-6) # Ratio to mean

    # One-time NaN cleanup after rolling calculations.
    # ffill() propagates last valid observation forward.
    # bfill() fills NA with next valid observation.
    # infer_objects(copy=False) attempts to infer better dtypes for object columns.
    out = out.ffill().bfill().infer_objects(copy=False)
    return out

# ─────────────────────────── live predictor ──────────────────────────────────
# Initialize an empty DataFrame to store historical data
history = pd.DataFrame(columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])

def predict_row(row: pd.Series) -> None:
    """
    Appends a new data row to the history, builds features,
    runs the prediction models, and prints the formatted result.
    """
    global history # Use the global history DataFrame

    # Append the new row to history and keep only the last KEEP_LAST_N rows
    # .to_frame().T converts the Series to a single-row DataFrame for concatenation
    history = pd.concat([history, row.to_frame().T], ignore_index=True).tail(KEEP_LAST_N)

    ts = row['timestamp'] # Extract timestamp for printing

    # Check if enough data has accumulated for a reliable prediction (warm-up period)
    if len(history) < WARMUP_ROWS:
        print(f"{ts} – warming-up ({len(history)}/{WARMUP_ROWS}) …")
        return # Skip prediction if not enough data

    # Build features on the current history
    feats = build_features(history)

    # --- Anomaly Prediction ---
    # Select the last row of features for prediction
    # an_feats contains the list of features the anomaly model was trained on
    current_an_feats = feats.iloc[[-1]][an_feats]
    an_prob = an_clf.predict_proba(current_an_feats)[0, 1] # Probability of class 1 (anomaly)
    an_flag = -1 if an_prob > 0.5 else 1 # Convert probability to a flag (-1 for anomaly, 1 for normal)

    # --- Power Regression ---
    # The power model might require the anomaly probability as an input feature
    pw_input = feats.iloc[[-1]].copy() # Take the last row of features
    pw_input["anomaly_prob"] = an_prob # Add the calculated anomaly probability
    # pw_feats contains the list of features the power model was trained on
    current_pw_feats = pw_input[pw_feats]
    power_val = pw_reg.predict(current_pw_feats)[0] # Predict power value

    # --- Direction Classification ---
    # dir_feats contains the list of features the direction model was trained on
    current_dir_feats = feats.iloc[[-1]][dir_feats]
    dir_code = int(dir_clf.predict(current_dir_feats)[0]) # Predict direction code
    dir_flag = label_rev[dir_code] # Convert code to a meaningful flag (-1, 0, 1)

    # Print the formatted prediction output
    print(f"{ts} → Anomaly:{an_flag:+d} (p={an_prob:.3f}) | "
          f"Power:{power_val:6.2f}% | Direction:{dir_flag:+d}")

# ─────────────────────────── main execution ──────────────────────────────────
if __name__ == "__main__":
    try:
        # Load data from CSV file
        # Assumes 'chart_data.csv' is in the same directory as the script.
        df = pd.read_csv(r"D:\project\BTCUSDT_ohlc_data_1m(1days).csv")
        # Convert 'Datetime' column to pandas datetime objects and set as 'timestamp'
        df['timestamp'] = pd.to_datetime(df.pop('timestamp'))

        # Iterate over each row in the DataFrame and make predictions
        for _, row in df.iterrows(): # _ is used for the index, which is not needed here
            predict_row(row)

    except FileNotFoundError:
        print("Error: chart_data.csv not found. Please ensure the file is in the correct directory.")
    except KeyError as e:
        if 'Datetime' in str(e):
            print("Error: 'Datetime' column not found in chart_data.csv. Please check the CSV file header.")
        else:
            print(f"Error: Missing expected column in chart_data.csv: {e}")
    except Exception as exc:
        # Catch any other unexpected errors during execution
        print(f"An unexpected error occurred: {exc}")
