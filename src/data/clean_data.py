
"""Create a conservative, auditable cleaning baseline."""

from pathlib import Path

import pandas as pd

from loader import load_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "transactions_cleaned.csv"


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text fields and add quality flags without dropping rows."""
    cleaned = df.copy()

    # Standardize text while preserving the original source workbook.
    cleaned["StockCode"] = cleaned["StockCode"].astype("string").str.strip()
    cleaned["Description"] = cleaned["Description"].astype("string").str.strip()
    cleaned["Country"] = cleaned["Country"].astype("string").str.strip()

    # Preserve potentially meaningful rows; expose quality issues as flags.
    cleaned["is_exact_duplicate"] = cleaned.duplicated(keep="first")
    cleaned["has_missing_customer_id"] = cleaned["Customer_ID"].isna()
    cleaned["has_missing_description"] = cleaned["Description"].isna()
    cleaned["is_return_or_cancellation"] = cleaned["Quantity"] < 0
    cleaned["has_non_positive_price"] = cleaned["Price"] <= 0
    cleaned["is_cancel_invoice"] = (
        cleaned["Invoice"].astype("string").str.startswith("C", na=False)
    )

    return cleaned


def main() -> None:
    """Load, flag and save a non-destructive cleaning baseline."""
    df = load_transactions()
    cleaned = clean_transactions(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Cleaning baseline summary ---")
    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(cleaned):,}")
    print(f"Exact duplicate occurrences flagged: {cleaned['is_exact_duplicate'].sum():,}")
    print(f"Missing customer IDs: {cleaned['has_missing_customer_id'].sum():,}")
    print(f"Returns/cancellations by quantity: {cleaned['is_return_or_cancellation'].sum():,}")
    print(f"Non-positive prices: {cleaned['has_non_positive_price'].sum():,}")
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
