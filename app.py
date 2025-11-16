from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import os
import sys

# Add the "model" folder to the system path
sys.path.append(os.path.join(os.getcwd(), "model"))

# Now import `predict_price` from `model.py`
from model import predict_price

app = Flask(__name__)

# Load dataset before encoding
df = pd.read_csv("datasets/car_price_dataset.csv")

# Extract unique brands and models **before encoding**
brands = df["Brand"].unique().tolist()
models = df.groupby("Brand")["Model"].unique().apply(list).to_dict()

# Load feature column names for consistent input processing
with open(os.path.join("model", "feature_columns.pkl"), "rb") as file:
    feature_columns = pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html", brands=brands, models=models)

@app.route('/get_models', methods=['POST'])
def get_models():
    """Returns the models for the selected brand."""
    brand = request.json.get("brand")
    model_list = models.get(brand, [])
    return jsonify(model_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        brand = request.form.get("brand")
        model = request.form.get("model")
        year = float(request.form.get("year"))
        fuel_type = request.form.get("fuel_type")
        transmission = request.form.get("transmission")
        mileage = float(request.form.get("mileage"))
        engine_size = float(request.form.get("engine_size"))
        doors = int(request.form.get("doors"))
        owner_count = int(request.form.get("owner_count"))

        # Prepare input in the correct feature order
        input_data = np.zeros(len(feature_columns))
        input_dict = {
            "Year": year,
            "Mileage": mileage,
            "Engine_Size": engine_size,
            "Doors": doors,
            "Owner_Count": owner_count,
            f"Brand_{brand}": 1,
            f"Model_{model}": 1,
            f"Fuel_Type_{fuel_type}": 1,
            f"Transmission_{transmission}": 1
        }

        for i, feature in enumerate(feature_columns):
            if feature in input_dict:
                input_data[i] = input_dict[feature]

        # Make prediction   ``
        prediction = predict_price(input_data)

        return render_template("index.html", prediction_text=f"Estimated Car Price: ${prediction}", brands=brands, models=models)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
