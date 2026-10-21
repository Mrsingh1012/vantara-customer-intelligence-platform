
"""Explore country-level transaction activity without changing source data."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT = ROOT / "reports" / "country_transaction_summary.csv"


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)

    df["Revenue"] = df["Quantity"] * df["Price"]

    summary = (
        df.groupby("Country", dropna=False)
        .agg(
            transaction_rows=("Invoice", "size"),
            unique_invoices=("Invoice", "nunique"),
            identified_customers=("Customer_ID", "nunique"),
            signed_revenue=("Revenue", "sum"),
            negative_quantity_rows=("is_negative_quantity", "sum"),
            cancel_invoice_rows=("is_cancel_invoice", "sum"),
        )
        .reset_index()
        .sort_values("signed_revenue", ascending=False)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print("\n--- Top 10 countries by signed revenue ---")
    print(summary.head(10).to_string(index=False))

    print("\n--- Overall summary ---")
    print(f"Countries: {summary['Country'].nunique()}")
    print(f"Rows covered: {summary['transaction_rows'].sum():,}")
    print(f"Summary saved to: {OUTPUT}")
    print("No rows were removed.")


if __name__ == "__main__":
    main()
