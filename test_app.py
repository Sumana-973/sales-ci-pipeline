
import unittest
from unittest.mock import patch

from app import app


class TestSalesPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    @patch("app.load_model")
    def test_prediction_endpoint(self, mock_load_model):
        mock_model = mock_load_model.return_value
        mock_model.predict.return_value = [1250.50]

        response = self.client.post(
            "/predict",
            json={
    "Quantity_Sold": 10,
    "Unit_Cost": 50,
    "Discount": 0.1,
    "Region_Code": 1,
    "Product_Code": 2
}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"], 1250.50
        )

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={"Quantity_Sold": 10}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_invalid_discount(self):
        response = self.client.post(
            "/predict",
            json={
                "Quantity_Sold": 10,
                "Unit_Cost": 50,
                "Unit_Price": 150,
                "Discount": 1.5
            }
        )

        self.assertEqual(response.status_code, 400)

    def test_negative_quantity(self):
        response = self.client.post(
            "/predict",
            json={
    "Quantity_Sold": -10,
    "Unit_Cost": 50,
    "Discount": 0.1,
    "Region_Code": 1,
    "Product_Code": 2
}
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
