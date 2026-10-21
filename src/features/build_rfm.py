
"""Build a preliminary customer-level RFM dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT = ROOT / "data" / "processed" / "customer_rfm.csv"

# Explicit historical snapshot reference date.
# The source data's last transaction is 2011-12-09.
REFERENCE_DATE = pd.Timestamp("2011-12-10")


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Preliminary purchase-only population:
    # identified customers, positive quantity and price,
    # and invoice not prefixed with "C".
    purchases = df.loc[
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["is_cancel_invoice"]
    ].copy()

    purchases["purchase_value"] = purchases["Quantity"] * purchases["Price"]

    rfm = (
        purchases.groupby("Customer_ID")
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("Invoice", "nunique"),
            monetary=("purchase_value", "sum"),
            purchase_rows=("Invoice", "size"),
        )
        .reset_index()
    )

    rfm["recency_days"] = (
        REFERENCE_DATE - rfm["last_purchase"].dt.normalize()
    ).dt.days

    # Arrange columns for readability.
    rfm = rfm[
        [
            "Customer_ID",
            "recency_days",
            "frequency",
            "monetary",
            "purchase_rows",
            "last_purchase",
        ]
    ].sort_values("Customer_ID")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(OUTPUT, index=False)

    print("\n--- Preliminary RFM summary ---")
    print(f"Eligible transaction rows: {len(purchases):,}")
    print(f"Unique customers: {rfm['Customer_ID'].nunique():,}")
    print(f"Reference date: {REFERENCE_DATE.date()}")
    print(f"Recency minimum: {rfm['recency_days'].min()} days")
    print(f"Recency maximum: {rfm['recency_days'].max()} days")
    print(f"Frequency median: {rfm['frequency'].median():.1f} invoices")
    print(f"Monetary median: {rfm['monetary'].median():,.2f}")
    print(f"Output saved to: {OUTPUT}")
    print("\nSample customers:")
    print(rfm.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
