"""Create a conservative, auditable cleaning baseline."""
from pathlib import Path

import pandas as pd

from loader import load_transactions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "transactions_cleaned.csv"


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text and add quality flags without dropping rows."""
    cleaned = df.copy()

    # Normalize text fields without modifying the source workbook.
    for column in ["StockCode", "Description", "Country"]:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    # Audit flags: preserve all rows for later investigation.
    cleaned["is_exact_duplicate"] = cleaned.duplicated(keep="first")
    cleaned["has_missing_customer_id"] = cleaned["Customer_ID"].isna()
    cleaned["has_missing_description"] = cleaned["Description"].isna()
    cleaned["is_negative_quantity"] = cleaned["Quantity"] < 0
    cleaned["is_zero_quantity"] = cleaned["Quantity"] == 0
    cleaned["is_negative_price"] = cleaned["Price"] < 0
    cleaned["is_zero_price"] = cleaned["Price"] == 0
    cleaned["is_cancel_invoice"] = (
        cleaned["Invoice"].astype("string").str.startswith("C", na=False)
    )

    # A combined flag for review, not an automatic deletion rule.
    cleaned["requires_transaction_review"] = (
        cleaned["is_negative_quantity"]
        | cleaned["is_zero_quantity"]
        | cleaned["is_negative_price"]
        | cleaned["is_zero_price"]
        | cleaned["is_cancel_invoice"]
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
    print(f"Rows removed: {len(df) - len(cleaned):,}")
    print(f"Exact duplicate occurrences flagged: {cleaned['is_exact_duplicate'].sum():,}")
    print(f"Missing customer IDs: {cleaned['has_missing_customer_id'].sum():,}")
    print(f"Missing descriptions: {cleaned['has_missing_description'].sum():,}")
    print(f"Negative quantity rows: {cleaned['is_negative_quantity'].sum():,}")
    print(f"Zero quantity rows: {cleaned['is_zero_quantity'].sum():,}")
    print(f"Negative price rows: {cleaned['is_negative_price'].sum():,}")
    print(f"Zero price rows: {cleaned['is_zero_price'].sum():,}")
    print(f"Cancellation-prefixed invoices: {cleaned['is_cancel_invoice'].sum():,}")
    print(
        "Rows requiring transaction review: "
        f"{cleaned['requires_transaction_review'].sum():,}"
    )
    print(f"Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()