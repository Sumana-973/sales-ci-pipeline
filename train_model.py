
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def train_model():
    print("Creating sales dataset...")

    rng = np.random.default_rng(42)
    n = 500

    data = pd.DataFrame({
        "Quantity_Sold": rng.integers(1, 21, n),
        "Unit_Cost": rng.uniform(10, 500, n).round(2),
        "Discount": rng.uniform(0, 0.30, n).round(2),
        "Region_Code": rng.integers(0, 4, n),
        "Product_Code": rng.integers(0, 10, n)
    })

    data["Unit_Price"] = (
        data["Unit_Cost"] * rng.uniform(1.1, 1.8, n)
    ).round(2)

    data["Sales_Amount"] = (
        data["Quantity_Sold"]
        * data["Unit_Price"]
        * (1 - data["Discount"])
    ).round(2)

    data.to_csv("sales_training_data.csv", index=False)
    print("Records:", len(data))

    features = [
        "Quantity_Sold", "Unit_Cost", "Discount",
        "Region_Code", "Product_Code"
    ]

    X = data[features]
    y = data["Sales_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Training model...")

    model = RandomForestRegressor(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation")
    print("MAE:", round(float(mae), 2))
    print("R2 Score:", round(float(r2), 4))

    joblib.dump(model, "sales_model.pkl")

    metrics = {
        "mae": float(mae),
        "r2_score": float(r2),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model saved: sales_model.pkl")
    print("Metrics saved: metrics.json")


if __name__ == "__main__":
    train_model()
