import pandas as pd
import numpy as np
import os # Import the 'os' module for creating directories and paths

# --- Data Loading Function (keep from previous code) ---
def load_data(filepath):
    """Loads data from CSV, converts timestamp, and sorts chronologically."""
    try:
        df = pd.read_csv(filepath)
        time_col = next((col for col in ['timestamp', 'Date', 'time'] if col in df.columns), None)

        if time_col:
            print(f"Detected time column: '{time_col}'")
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)
            print(f"✅ Loaded and sorted {len(df)} rows by '{time_col}' from {filepath}")
        else:
            print("⚠️ Warning: No standard time column ('timestamp', 'Date', 'time') found.")
            print("   Assuming data is already sorted chronologically.")
            print(f"✅ Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading data from {filepath}: {e}")
        return None

# --- Chronological Splitting Function (keep from previous code) ---
def split_data_chronological(df, train_pct=0.80, backtest_pct=0.15):
    """
    Splits a DataFrame chronologically into training, backtesting, and testing sets.
    (Implementation is the same as provided in the previous answer)
    """
    if df is None or df.empty:
        print("❌ Input DataFrame is None or empty. Cannot split.")
        return None, None, None
    n_total = len(df)
    if n_total == 0:
         print("❌ Input DataFrame has zero rows. Cannot split.")
         return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
    test_pct = 1.0 - train_pct - backtest_pct
    if not (train_pct > 0 and backtest_pct > 0 and test_pct > 0):
         print(f"❌ Invalid percentages: train={train_pct:.2f}, backtest={backtest_pct:.2f}, test={test_pct:.2f}.")
         print("   All calculated percentages must be greater than 0.")
         return None, None, None
    if not np.isclose(train_pct + backtest_pct + test_pct, 1.0):
        print(f"❌ Percentages do not sum close to 1 (Sum = {train_pct + backtest_pct + test_pct:.4f}).")
        return None, None, None
    train_end_idx = int(n_total * train_pct)
    backtest_end_idx = int(n_total * (train_pct + backtest_pct))
    df_train = df.iloc[:train_end_idx]
    df_backtest = df.iloc[train_end_idx:backtest_end_idx]
    df_test = df.iloc[backtest_end_idx:]
    print("\n--- Data Split Summary ---")
    print(f"Total rows: {n_total}")
    print(f"Training set:   {len(df_train):>7} rows ({train_pct*100:>5.1f}%) | Index: 0 to {train_end_idx-1 if train_end_idx > 0 else 'N/A'}")
    print(f"Backtesting set:{len(df_backtest):>7} rows ({backtest_pct*100:>5.1f}%) | Index: {train_end_idx} to {backtest_end_idx-1 if backtest_end_idx > train_end_idx else 'N/A'}")
    print(f"Testing set:    {len(df_test):>7} rows ({test_pct*100:>5.1f}%) | Index: {backtest_end_idx} to {n_total-1 if n_total > backtest_end_idx else 'N/A'}")
    print("--------------------------\n")
    if len(df_train) + len(df_backtest) + len(df_test) != n_total:
         print("❌ Warning: Row counts of splits do not sum up exactly to the original total.")
    return df_train, df_backtest, df_test

# === Main Execution Block ===
if __name__ == "__main__":
    # --- Configuration ---
    # Input file path (change this!)
    input_filepath = r"D:\project\Data_test\BTCUSDT_ohlc_data_1m(5year).csv" # <--- YOUR FULL DATASET PATH

    # Split percentages
    train_percentage = 0.80
    backtest_percentage = 0.15
    test_percentage = 1.0 - train_percentage - backtest_percentage # Calculated: 0.05

    # --- Output Configuration ---
    # Directory where split files will be saved
    output_directory = "split_output_data"

    # Create base filename from input file (optional, makes names informative)
    # Example: "D:\project\BTCUSDT_ohlc_data_1m.csv" -> "BTCUSDT_ohlc_data_1m"
    input_basename = os.path.splitext(os.path.basename(input_filepath))[0]

    # Define output filenames
    train_filename = f"{input_basename}_train_{int(train_percentage*100)}pct.csv"
    backtest_filename = f"{input_basename}_backtest_{int(backtest_percentage*100)}pct.csv"
    test_filename = f"{input_basename}_test_{int(test_percentage*100)}pct.csv"

    # Construct full output paths
    train_save_path = os.path.join(output_directory, train_filename)
    backtest_save_path = os.path.join(output_directory, backtest_filename)
    test_save_path = os.path.join(output_directory, test_filename)

    # --- Create Output Directory (if it doesn't exist) ---
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            print(f"Created output directory: '{output_directory}'")
        else:
            print(f"Output directory already exists: '{output_directory}'")
    except OSError as e:
        print(f"❌ Error creating directory '{output_directory}': {e}")
        # Decide if you want to exit here or try saving in the current directory
        output_directory = "." # Fallback to current directory
        print(f"⚠️ Attempting to save files in the current directory instead.")
        train_save_path = train_filename
        backtest_save_path = backtest_filename
        test_save_path = test_filename


    # --- Load Data ---
    print(f"\nAttempting to load data from: {input_filepath}")
    full_df = load_data(input_filepath)

    if full_df is not None:
        # --- Split the Data ---
        print("Splitting data chronologically...")
        train_df, backtest_df, test_df = split_data_chronological(
            full_df,
            train_pct=train_percentage,
            backtest_pct=backtest_percentage
        )

        # --- Proceed only if split was successful ---
        if train_df is not None:
            print("\n--- Saving Split Data ---")
            try:
                # Save Training Data
                if not train_df.empty:
                    train_df.to_csv(train_save_path, index=False) # index=False prevents writing row numbers
                    print(f"✅ Training data ({len(train_df)} rows) saved to: '{train_save_path}'")
                else:
                     print("⚠️ Training data is empty, not saving file.")

                # Save Backtesting Data
                if not backtest_df.empty:
                    backtest_df.to_csv(backtest_save_path, index=False)
                    print(f"✅ Backtesting data ({len(backtest_df)} rows) saved to: '{backtest_save_path}'")
                else:
                    print("⚠️ Backtesting data is empty, not saving file.")

                # Save Testing Data
                if not test_df.empty:
                     test_df.to_csv(test_save_path, index=False)
                     print(f"✅ Testing data ({len(test_df)} rows) saved to: '{test_save_path}'")
                else:
                     print("⚠️ Testing data is empty, not saving file.")

            except Exception as e:
                print(f"\n❌ An error occurred while saving the files: {e}")
                print("   Please check file paths and permissions.")

        else:
            print("Data splitting failed. Files not saved.")
    else:
        print("Could not load data, stopping the process.")