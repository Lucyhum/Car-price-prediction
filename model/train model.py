import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the target directory
save_path = r"C:\Users\Dell\OneDrive\Desktop\Car price prediction model\model"

# Ensure the directory exists
os.makedirs(save_path, exist_ok=True)

# Load Dataset
df = pd.read_csv("datasets/car_price_dataset.csv")
print(df.head)

# Handle Missing Values
df.dropna(inplace=True)

# Encode Categorical Features (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)  

# Define Features & Target
X = df.drop(columns=['Price'])  # Independent Features
y = df['Price']  # Target Variable

# Save Column Names for Consistency in Prediction
feature_columns = X.columns.tolist()
with open(os.path.join(save_path, "feature_columns.pkl"), "wb") as file:
    pickle.dump(feature_columns, file)

# Train-Test Split (Before Scaling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Normalize X and Save Scaler)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open(os.path.join(save_path, "scaler.pkl"), "wb") as file:
    pickle.dump(scaler, file)

# Train Model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model as Pickle File
with open(os.path.join(save_path, "car_price_model.pkl"), "wb") as file:
    pickle.dump(model, file)

# Make Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred) 
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:\n MAE: {mae:.2f}\n MSE: {mse:.2f}\n RMSE: {rmse:.2f}\n R² Score: {r2:.2f}")
print(f"Model and related files saved in: {save_path}")
