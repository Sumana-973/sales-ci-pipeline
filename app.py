
from pathlib import Path
import logging
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

app = Flask(__name__)

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

MODEL_PATH = Path("sales_model.pkl")

FEATURES = [
    "Quantity_Sold",
    "Unit_Cost",
    "Discount",
    "Region_Code",
    "Product_Code"
]


@app.before_request
def start_request_tracking():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000, 2
    )

    response.headers["X-Request-ID"] = request_id

    app.logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "request_id": getattr(g, "request_id", "unknown")
    }), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(
        "Internal server error request_id=%s",
        getattr(g, "request_id", "unknown"),
        exc_info=True
    )
    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(g, "request_id", "unknown")
    }), 500


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "sales-prediction"
    })


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "sales_model.pkl not found. Train the model first."
        )
    return joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON object request body is required",
            "request_id": g.request_id
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        app.logger.warning(
            "Missing fields request_id=%s fields=%s",
            g.request_id, missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
            "request_id": g.request_id
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
            for feature in [
                "Quantity_Sold",
                "Unit_Cost",
                "Region_Code",
                "Product_Code"
            ]
        ):
            return jsonify({
                "error": "Quantity, cost, and codes must be non-negative",
                "request_id": g.request_id
            }), 400

        discount = sample["Discount"].iloc[0]

        if not pd.notna(discount) or not 0 <= discount <= 1:
            return jsonify({
                "error": "Discount must be between 0 and 1",
                "request_id": g.request_id
            }), 400

        model = load_model()
        prediction = float(model.predict(sample)[0])

        app.logger.info(
            "Prediction succeeded request_id=%s",
            g.request_id
        )

        return jsonify({
            "prediction": round(prediction, 2),
            "request_id": g.request_id
        })

    except (TypeError, ValueError):
        app.logger.warning(
            "Invalid feature values request_id=%s",
            g.request_id
        )
        return jsonify({
            "error": "All feature values must be valid numbers",
            "request_id": g.request_id
        }), 400

    except Exception:
        app.logger.exception(
            "Prediction failed request_id=%s",
            g.request_id
        )
        return jsonify({
            "error": "Prediction failed",
            "request_id": g.request_id
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
