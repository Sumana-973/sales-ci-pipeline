
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("sales_training_data.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("sales_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_mae_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["mae"], 0)

    def test_r2_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertTrue(-1 <= metrics["r2_score"] <= 1)

    def test_model_prediction(self):
        model = joblib.load("sales_model.pkl")

        sample = pd.DataFrame([{
            "Quantity_Sold": 10,
            "Unit_Cost": 100.0,
            "Discount": 0.1,
            "Region_Code": 1,
            "Product_Code": 2
        }])

        prediction = model.predict(sample)[0]
        self.assertGreater(prediction, 0)

    def test_training_records(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertEqual(metrics["training_records"], 400)
        self.assertEqual(metrics["testing_records"], 100)


if __name__ == "__main__":
    unittest.main()
