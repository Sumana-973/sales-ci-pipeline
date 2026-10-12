
import unittest
import pandas as pd
import numpy as np

from sales_pipeline import (
    clean_sales_data,
    calculate_sales_amount,
    summarize_sales
)


class TestSalesPipeline(unittest.TestCase):

    def test_sales_amount_calculation(self):
        self.assertEqual(
            calculate_sales_amount(10, 100, 0.1), 900.0
        )

    def test_zero_discount(self):
        self.assertEqual(
            calculate_sales_amount(5, 200, 0), 1000.0
        )

    def test_negative_quantity(self):
        with self.assertRaises(ValueError):
            calculate_sales_amount(-5, 100, 0.1)

    def test_invalid_discount(self):
        with self.assertRaises(ValueError):
            calculate_sales_amount(5, 100, 1.5)

    def test_cleaning_missing_values(self):
        sample = pd.DataFrame({
            "Sales_ID": ["S1", "S2"],
            "Sales_Date": ["2023-01-01", "2023-01-02"],
            "Product_Category": ["Grocery", "Clothing"],
            "Region": ["North", "South"],
            "Quantity_Sold": [5, np.nan],
            "Unit_Cost": [20, 30],
            "Unit_Price": [50, 60],
            "Discount": [0.1, 0.2],
            "Sales_Amount": [225, 96]
        })

        result = clean_sales_data(sample)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Sales_ID"].iloc[0], "S1")

    def test_sales_summary(self):
        sample = pd.DataFrame({
            "Region": ["North", "North", "South"],
            "Sales_Amount": [100, 200, 300],
            "Quantity_Sold": [1, 2, 3],
            "Sales_ID": ["S1", "S2", "S3"]
        })

        result = summarize_sales(sample)
        north = result[result["Region"] == "North"].iloc[0]

        self.assertEqual(north["Total_Sales"], 300)
        self.assertEqual(north["Total_Quantity"], 3)
        self.assertEqual(north["Transactions"], 2)


if __name__ == "__main__":
    unittest.main()
