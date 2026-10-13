
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("sales_model.pkl")

FEATURES = [
    "Quantity_Sold",
    "Unit_Cost",
    "Discount",
    "Region_Code",
    "Product_Code"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "sales_model.pkl not found. Train the model first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "sales-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        if not all(
            sample[feature].map(
                lambda value: pd.notna(value) and value >= 0
            ).all()
            for feature in ["Quantity_Sold", "Unit_Cost", "Unit_Price"]
        ):
            return jsonify({
                "error": "Quantity and prices must be non-negative"
            }), 400

        if not 0 <= sample["Discount"].iloc[0] <= 1:
            return jsonify({
                "error": "Discount must be between 0 and 1"
            }), 400

        model = load_model()
        prediction = float(model.predict(sample)[0])

        return jsonify({
            "prediction": round(prediction, 2)
        })

    except (TypeError, ValueError):
        return jsonify({
            "error": "All feature values must be valid numbers"
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
