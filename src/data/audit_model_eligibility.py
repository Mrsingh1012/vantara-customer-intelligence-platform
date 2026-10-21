
"""Audit transaction eligibility for customer-level ML features."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "transactions_cleaned.csv"


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    identified = df["Customer_ID"].notna()
    positive_quantity = df["Quantity"] > 0
    positive_price = df["Price"] > 0
    not_cancel = ~df["is_cancel_invoice"]

    # Baseline purchase definition: no deduplication yet.
    eligible = (
        identified
        & positive_quantity
        & positive_price
        & not_cancel
    )

    # Return/cancellation signals are audited separately.
    return_signal = (
        df["is_negative_quantity"]
        | df["is_cancel_invoice"]
    )

    print("\n--- Modeling eligibility audit ---")
    print(f"Total transaction rows: {len(df):,}")
    print(f"Identified customer rows: {identified.sum():,}")
    print(f"Positive quantity rows: {positive_quantity.sum():,}")
    print(f"Positive price rows: {positive_price.sum():,}")
    print(f"Eligible positive-sale rows: {eligible.sum():,}")
    print(f"Excluded from positive-sale baseline: {(~eligible).sum():,}")
    print(f"Return/cancellation signal rows: {return_signal.sum():,}")

    print(
        "Unique eligible customers: "
        f"{df.loc[eligible, 'Customer_ID'].nunique():,}"
    )
    print(
        "Unique eligible invoices: "
        f"{df.loc[eligible, 'Invoice'].nunique():,}"
    )

    print("\nNo rows were removed or files modified.")


if __name__ == "__main__":
    main()
