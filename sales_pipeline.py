
import pandas as pd


def clean_sales_data(data):
    data = data.copy()

    required = [
        "Sales_ID", "Sales_Date", "Product_Category",
        "Quantity_Sold", "Unit_Cost", "Unit_Price",
        "Discount", "Sales_Amount", "Region"
    ]

    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.drop_duplicates(subset=["Sales_ID"])
    data["Sales_Date"] = pd.to_datetime(
        data["Sales_Date"], errors="coerce"
    )

    numeric = [
        "Quantity_Sold", "Unit_Cost", "Unit_Price",
        "Discount", "Sales_Amount"
    ]

    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[
        "Sales_ID", "Sales_Date", "Quantity_Sold",
        "Unit_Cost", "Unit_Price", "Discount", "Sales_Amount"
    ])

    valid = (
        (data["Quantity_Sold"] >= 0)
        & (data["Unit_Cost"] >= 0)
        & (data["Unit_Price"] >= 0)
        & data["Discount"].between(0, 1)
    )

    return data.loc[valid].reset_index(drop=True)


def calculate_sales_amount(quantity, unit_price, discount):
    if quantity < 0 or unit_price < 0:
        raise ValueError("Quantity and price must be non-negative")
    if not 0 <= discount <= 1:
        raise ValueError("Discount must be between 0 and 1")

    return round(quantity * unit_price * (1 - discount), 2)


def summarize_sales(data):
    return data.groupby("Region", as_index=False).agg(
        Total_Sales=("Sales_Amount", "sum"),
        Total_Quantity=("Quantity_Sold", "sum"),
        Transactions=("Sales_ID", "count")
    )
