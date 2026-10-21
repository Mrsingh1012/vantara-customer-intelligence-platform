"""Validate the saved, non-destructive transaction cleaning baseline."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "transactions_cleaned.csv"

REQUIRED_COLUMNS = {
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer_ID",
    "Country",
    "is_exact_duplicate",
    "has_missing_customer_id",
    "has_missing_description",
    "is_negative_quantity",
    "is_zero_quantity",
    "is_negative_price",
    "is_zero_price",
    "is_cancel_invoice",
    "requires_transaction_review",
}


def validate_processed_data() -> None:
    """Check schema, row count, dates, and core quality flags."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Processed dataset is empty.")

    if df["InvoiceDate"].isna().any():
        raise ValueError("InvoiceDate contains missing or invalid values.")

    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Some InvoiceDate values could not be parsed.")

    expected_start = pd.Timestamp("2009-12-01")
    expected_end = pd.Timestamp("2011-12-10")

    if dates.min() < expected_start or dates.max() >= expected_end:
        raise ValueError("InvoiceDate values fall outside the expected date range.")

    # CSV booleans should be parsed as actual booleans.
    flag_columns = [
        "is_exact_duplicate",
        "has_missing_customer_id",
        "has_missing_description",
        "is_negative_quantity",
        "is_zero_quantity",
        "is_negative_price",
        "is_zero_price",
        "is_cancel_invoice",
        "requires_transaction_review",
    ]

    for column in flag_columns:
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"Expected boolean values in flag column: {column}")

    if not df["has_missing_customer_id"].equals(df["Customer_ID"].isna()):
        raise ValueError("Missing Customer_ID flag does not match the source column.")

    if not df["is_negative_quantity"].equals(df["Quantity"] < 0):
        raise ValueError("Negative quantity flag does not match Quantity.")

    if not df["is_zero_quantity"].equals(df["Quantity"] == 0):
        raise ValueError("Zero quantity flag does not match Quantity.")

    if not df["is_negative_price"].equals(df["Price"] < 0):
        raise ValueError("Negative price flag does not match Price.")

    if not df["is_zero_price"].equals(df["Price"] == 0):
        raise ValueError("Zero price flag does not match Price.")

    expected_review = (
        df["is_negative_quantity"]
        | df["is_zero_quantity"]
        | df["is_negative_price"]
        | df["is_zero_price"]
        | df["is_cancel_invoice"]
    )

    if not df["requires_transaction_review"].equals(expected_review):
        raise ValueError("Transaction review flag does not match its component flags.")

    print("Processed data validation passed.")
    print(f"Validated rows: {len(df):,}")
    print(f"Validated columns: {len(df.columns)}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    print(f"Rows requiring transaction review: {expected_review.sum():,}")


if __name__ == "__main__":
    validate_processed_data()