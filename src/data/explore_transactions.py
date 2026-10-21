
"""Explore monthly retail trends without modifying the source data."""

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "transactions_cleaned.csv"
REPORTS = ROOT / "reports"
CHART_PATH = REPORTS / "monthly_revenue_trend.png"
SUMMARY_PATH = REPORTS / "monthly_transaction_summary.csv"


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Retain all rows for analysis; classify signals separately.
    df["is_positive_sale"] = (
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["is_cancel_invoice"]
    )

    df["month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    # Descriptive net revenue: signed quantity × price, before any
    # final return/cancellation policy is applied.
    monthly = df.groupby("month").agg(
        transaction_rows=("Invoice", "size"),
        unique_invoices=("Invoice", "nunique"),
        identified_customers=("Customer_ID", "nunique"),
        net_revenue=("Revenue", "sum"),
        positive_sale_rows=("is_positive_sale", "sum"),
        negative_quantity_rows=("is_negative_quantity", "sum"),
        cancel_invoice_rows=("is_cancel_invoice", "sum"),
    ).reset_index()

    REPORTS.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(SUMMARY_PATH, index=False)

    plot_data = monthly.sort_values("month")
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data["month"], plot_data["net_revenue"], marker="o")
    plt.title("Online Retail II: Monthly Net Revenue (Exploratory)")
    plt.xlabel("Month")
    plt.ylabel("Net revenue (quantity × price)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150)
    plt.close()

    print("\n--- Monthly EDA summary ---")
    print(f"Source rows: {len(df):,}")
    print(f"Months covered: {monthly['month'].nunique()}")
    print(f"Total signed revenue: {df['Revenue'].sum():,.2f}")
    print(f"Positive-sale rows under preliminary rule: {df['is_positive_sale'].sum():,}")
    print(f"Monthly summary saved: {SUMMARY_PATH}")
    print(f"Revenue chart saved: {CHART_PATH}")

    print("\nFirst 5 months:")
    print(monthly.head().to_string(index=False))
    print("\nLast 5 months:")
    print(monthly.tail().to_string(index=False))


if __name__ == "__main__":
    main()
