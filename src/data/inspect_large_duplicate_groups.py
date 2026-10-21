
"""Inspect unusually large exact duplicate groups without removing rows."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "large_duplicate_groups_review.csv"


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

    group_sizes = (
        df.groupby(source_columns, dropna=False)
        .size()
        .rename("group_size")
        .reset_index()
    )

    large_groups = group_sizes.loc[group_sizes["group_size"] >= 5].copy()
    large_groups = large_groups.sort_values(
        "group_size", ascending=False
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    large_groups.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Large duplicate groups review ---")
    print(f"Groups with 5 or more identical rows: {len(large_groups):,}")
    print(
        "Total rows represented by these groups: "
        f"{int(large_groups['group_size'].sum()):,}"
    )

    if not large_groups.empty:
        print("\nLargest groups:")
        print(large_groups.head(10).to_string(index=False))

    print(f"\nReview file saved to: {OUTPUT_PATH}")
    print("No transaction rows were removed.")


if __name__ == "__main__":
    main()
