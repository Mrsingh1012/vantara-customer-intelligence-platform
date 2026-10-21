
"""Summarize the size distribution of exact duplicate groups."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "transactions_cleaned.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    source_columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer_ID",
        "Country",
    ]

    duplicate_rows = df.loc[
        df.duplicated(subset=source_columns, keep=False)
    ]

    group_sizes = (
        duplicate_rows.groupby(source_columns, dropna=False)
        .size()
        .rename("group_size")
    )

    print("\n--- Duplicate group size distribution ---")
    print(group_sizes.value_counts().sort_index().to_string())
    print(f"\nTotal duplicate groups: {len(group_sizes):,}")
    print(f"Rows participating in duplicate groups: {len(duplicate_rows):,}")
    print("No rows were removed.")


if __name__ == "__main__":
    main()
