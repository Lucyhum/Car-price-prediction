# ...existing code...
import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # Resolve paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, ".."))
    dataset_path = os.path.normpath(os.path.join(project_root, "datasets", "car_price_dataset.csv"))
    save_path = os.path.normpath(os.path.join(base_dir))

    os.makedirs(save_path, exist_ok=True)

    # Load dataset
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    print("Loaded dataset preview:")
    print(df.head())  # fixed: call the method

    # Ensure target exists and is numeric
    if "Price" not in df.columns:
        print("ERROR: 'Price' column not found in dataset.")
        sys.exit(1)

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Handle missing values (drop rows with any NA after conversion)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before - after} rows with missing values. Remaining rows: {after}")

    # One-hot encode only feature columns (exclude target)
    feature_df = pd.get_dummies(df.drop(columns=["Price"]), drop_first=True)

    # Define features and target
    X = feature_df
    y = df["Price"]

    # Save feature names for inference
    feature_columns = X.columns.tolist()
    with open(os.path.join(save_path, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)
    print(f"Saved feature columns ({len(feature_columns)}) to: {os.path.join(save_path, 'feature_columns.pkl')}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(os.path.join(save_path, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to: {os.path.join(save_path, 'scaler.pkl')}")

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    with open(os.path.join(save_path, "car_price_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {os.path.join(save_path, 'car_price_model.pkl')}")

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:\n MAE: {mae:.2f}\n MSE: {mse:.2f}\n RMSE: {rmse:.2f}\n R² Score: {r2:.2f}")
    print(f"Model and related files saved in: {save_path}")

if __name__ == "__main__":
    main()
# ...existing code...
