
"""Audit exact duplicate groups before deciding whether to remove rows."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "duplicate_group_audit.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Recompute exact duplicates from source transaction columns,
    # excluding flags added by the cleaning script.
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

    duplicate_mask = df.duplicated(subset=source_columns, keep=False)
    duplicates = df.loc[duplicate_mask].copy()

    if duplicates.empty:
        print("No duplicate groups found.")
        return

    grouped = (
        duplicates.groupby(source_columns, dropna=False)
        .agg(
            group_size=("Invoice", "size"),
            has_negative_quantity=("is_negative_quantity", "any"),
            has_cancel_invoice=("is_cancel_invoice", "any"),
            has_missing_customer_id=("has_missing_customer_id", "any"),
        )
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Duplicate group audit ---")
    print(f"Rows participating in duplicate groups: {len(duplicates):,}")
    print(f"Unique duplicate groups: {len(grouped):,}")
    print(f"Groups with negative quantity: {int(grouped['has_negative_quantity'].sum()):,}")
    print(f"Groups with C-prefixed invoice: {int(grouped['has_cancel_invoice'].sum()):,}")
    print(f"Groups with missing customer ID: {int(grouped['has_missing_customer_id'].sum()):,}")
    print(f"Audit saved to: {OUTPUT_PATH}")
    print("No transaction rows were removed.")


if __name__ == "__main__":
    main()
