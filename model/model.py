import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Automatically detect the folder where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model files (relative, works on Render + local)
model_path = os.path.join(BASE_DIR, "car_price_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
feature_columns_path = os.path.join(BASE_DIR, "feature_columns.pkl")

# Load the trained model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load the saved scaler
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Load feature column names
with open(feature_columns_path, "rb") as file:
    feature_columns = pickle.load(file)

# Function to predict car price
def predict_price(features):
    # Ensure the input matches the required feature format
    input_array = np.zeros(len(feature_columns))
    for i, feature in enumerate(features):
        input_array[i] = feature
    
    # Scale input
    features_scaled = scaler.transform([input_array])
    prediction = model.predict(features_scaled)[0]  # Get prediction
    return round(prediction, 2)

if __name__ == "__main__":
    sample_input = np.random.rand(len(feature_columns))  # Example input
    print("Predicted Car Price:", predict_price(sample_input))

print("Model loaded successfully!")
