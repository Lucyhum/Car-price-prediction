import pickle
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Define the directory where the .pkl files are stored
load_path = r"C:\Users\jainh\Desktop\Car price prediction model\model"

# Load the trained model
with open(os.path.join(load_path, "car_price_model.pkl"), "rb") as file:
    model = pickle.load(file)

# Load the saved scaler
with open(os.path.join(load_path, "scaler.pkl"), "rb") as file:
    scaler = pickle.load(file)

# Load feature column names
with open(os.path.join(load_path, "feature_columns.pkl"), "rb") as file:
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
