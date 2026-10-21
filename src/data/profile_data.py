
"""Profile the Online Retail II dataset for data-quality issues."""

from pathlib import Path

import pandas as pd

from loader import load_transactions


def main() -> None:
    """Print baseline data-quality statistics."""
    df = load_transactions()

    print("\n========== DATA QUALITY REPORT ==========")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\n--- Missing values ---")
    print(df.isna().sum().to_string())

    print("\n--- Exact duplicate rows ---")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n--- Quantity analysis ---")
    print(f"Negative quantity: {(df['Quantity'] < 0).sum():,}")
    print(f"Zero quantity: {(df['Quantity'] == 0).sum():,}")

    print("\n--- Price analysis ---")
    print(f"Negative price: {(df['Price'] < 0).sum():,}")
    print(f"Zero price: {(df['Price'] == 0).sum():,}")

    print("\n--- Customer analysis ---")
    print(f"Missing Customer IDs: {df['Customer_ID'].isna().sum():,}")
    print(
        "Unique identified customers: "
        f"{df['Customer_ID'].nunique():,}"
    )

    print("\n--- Invoice analysis ---")
    print(f"Unique invoices: {df['Invoice'].nunique():,}")
    print(f"Invalid invoice dates: {df['InvoiceDate'].isna().sum():,}")

    print("\n--- Numeric summary ---")
    print(df[["Quantity", "Price"]].describe().round(2).to_string())

    print("\n========== REPORT COMPLETE ==========")


if __name__ == "__main__":
    main()
