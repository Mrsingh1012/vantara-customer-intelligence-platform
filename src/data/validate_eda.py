
"""Validate monthly EDA output against transaction-level calculations."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "transactions_cleaned.csv"
SUMMARY = ROOT / "reports" / "monthly_transaction_summary.csv"


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    monthly = pd.read_csv(SUMMARY)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    expected = df.groupby("month")["Revenue"].sum().sort_index()
    actual = monthly.set_index("month")["net_revenue"].sort_index()

    if len(df) != 1_067_371:
        raise ValueError(f"Unexpected row count: {len(df):,}")

    if not expected.index.equals(actual.index):
        raise ValueError("Month labels or ordering do not match.")

    if not expected.index.is_unique or not actual.index.is_unique:
        raise ValueError("Duplicate month labels detected.")

    if not ((expected - actual).abs() < 0.02).all():
        raise ValueError("Monthly revenue totals do not reconcile.")

    if len(monthly) != 25:
        raise ValueError(f"Expected 25 monthly records, found {len(monthly)}.")

    print("Monthly EDA validation passed.")
    print(f"Transaction rows: {len(df):,}")
    print(f"Monthly records: {len(monthly)}")
    print(f"Transaction-level signed revenue: {df['Revenue'].sum():,.2f}")
    print(f"Monthly total: {monthly['net_revenue'].sum():,.2f}")
    print(f"First month: {monthly['month'].iloc[0]}")
    print(f"Last month: {monthly['month'].iloc[-1]}")


if __name__ == "__main__":
    main()
