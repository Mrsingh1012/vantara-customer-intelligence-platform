
"""Inspect duplicate transactions, returns, and unusual prices."""

from pathlib import Path

import pandas as pd

from loader import load_transactions


def main() -> None:
    """Generate diagnostic summaries without modifying source data."""
    df = load_transactions()

    output_dir = Path(__file__).resolve().parents[2] / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exact duplicate rows: inspect before deleting anything.
    duplicates = df.loc[df.duplicated(keep=False)].copy()
    duplicates.to_csv(
        output_dir / "duplicate_rows_review.csv",
        index=False,
    )

    print("\n--- Duplicate review ---")
    print(f"Rows participating in duplicate groups: {len(duplicates):,}")
    print(
        "Duplicate groups (excluding first occurrence): "
        f"{df.duplicated().sum():,}"
    )

    # 2. Negative quantities: likely returns or cancellations.
    returns = df.loc[df["Quantity"] < 0].copy()

    print("\n--- Negative quantity review ---")
    print(f"Rows with negative quantity: {len(returns):,}")
    print("Sample rows:")
    print(
        returns[
            ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"]
        ].head(10).to_string(index=False)
    )

    # 3. Non-positive prices: investigate, do not delete yet.
    unusual_prices = df.loc[df["Price"] <= 0].copy()
    unusual_prices.to_csv(
        output_dir / "non_positive_prices_review.csv",
        index=False,
    )

    print("\n--- Non-positive price review ---")
    print(f"Rows with price <= 0: {len(unusual_prices):,}")
    print(
        unusual_prices[
            ["Invoice", "StockCode", "Description", "Quantity", "Price"]
        ].head(10).to_string(index=False)
    )

    # 4. Missing descriptions.
    missing_descriptions = df.loc[df["Description"].isna()]

    print("\n--- Missing description review ---")
    print(f"Rows with missing Description: {len(missing_descriptions):,}")
    print(
        "Distinct StockCodes with missing descriptions: "
        f"{missing_descriptions['StockCode'].nunique():,}"
    )

    print(f"\nReview files saved in: {output_dir}")
    print("Source workbook was not modified.")


if __name__ == "__main__":
    main()
