
"""Validate preliminary customer-level RFM features."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT = ROOT / "data" / "processed" / "customer_rfm.csv"


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    rfm = pd.read_csv(OUTPUT, low_memory=False)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    eligible = df.loc[
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["is_cancel_invoice"]
    ].copy()

    eligible["purchase_value"] = eligible["Quantity"] * eligible["Price"]

    expected = eligible.groupby("Customer_ID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency=("Invoice", "nunique"),
        monetary=("purchase_value", "sum"),
        purchase_rows=("Invoice", "size"),
    ).sort_index()

    actual = rfm.set_index("Customer_ID").sort_index()

    required = {
        "recency_days",
        "frequency",
        "monetary",
        "purchase_rows",
        "last_purchase",
    }
    missing = required - set(rfm.columns)
    if missing:
        raise ValueError(f"Missing RFM columns: {sorted(missing)}")

    if rfm["Customer_ID"].isna().any():
        raise ValueError("RFM contains missing Customer_ID values.")

    if rfm["Customer_ID"].duplicated().any():
        raise ValueError("RFM contains duplicate customer rows.")

    if len(actual) != len(expected):
        raise ValueError("RFM customer count does not match eligible transactions.")

    if not actual.index.equals(expected.index):
        raise ValueError("RFM customer IDs do not match eligible customers.")

    if not (actual["frequency"] == expected["frequency"]).all():
        raise ValueError("Frequency values do not match invoice counts.")

    if not (actual["purchase_rows"] == expected["purchase_rows"]).all():
        raise ValueError("Purchase-row counts do not match.")

    if not (
        (actual["monetary"] - expected["monetary"]).abs() < 0.02
    ).all():
        raise ValueError("Monetary values do not match eligible purchases.")

    actual_last = pd.to_datetime(actual["last_purchase"])
    expected_last = expected["last_purchase"]

    if not actual_last.equals(expected_last):
        raise ValueError("Last purchase dates do not match.")

    if (actual["recency_days"] < 1).any():
        raise ValueError("Unexpected recency below one day.")

    print("RFM validation passed.")
    print(f"Eligible transaction rows: {len(eligible):,}")
    print(f"Unique customers validated: {len(rfm):,}")
    print(f"Total purchase value: {eligible['purchase_value'].sum():,.2f}")
    print(f"Unique invoices across eligible rows: {eligible['Invoice'].nunique():,}")


if __name__ == "__main__":
    main()
