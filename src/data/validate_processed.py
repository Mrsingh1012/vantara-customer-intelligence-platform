
"""Validate the cleaned transaction dataset."""

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
    "is_return_or_cancellation",
    "has_non_positive_price",
    "is_cancel_invoice",
}


def validate_processed_data() -> None:
    """Check schema, row count, dates and quality-flag consistency."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed file not found: {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda column: column in REQUIRED_COLUMNS,
        low_memory=False,
    )

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Processed dataset is empty.")

    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid or missing transaction dates found.")

    if not dates.between(
        pd.Timestamp("2009-12-01"),
        pd.Timestamp("2011-12-10"),
    ).all():
        raise ValueError("Transaction date outside expected dataset range.")

    if not (
        df["has_missing_customer_id"].astype(bool)
        == df["Customer_ID"].isna()
    ).all():
        raise ValueError("Customer ID quality flags are inconsistent.")

    print("Processed data validation passed.")
    print(f"Validated rows: {len(df):,}")
    print(f"Validated columns: {len(df.columns)}")
    print(f"Date range: {dates.min()} to {dates.max()}")


if __name__ == "__main__":
    validate_processed_data()
