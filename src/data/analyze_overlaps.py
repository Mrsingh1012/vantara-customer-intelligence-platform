
"""Analyze overlaps between transaction quality flags without dropping rows."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "interim" / "transactions_cleaned.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)

    flags = [
        "is_exact_duplicate",
        "is_negative_quantity",
        "is_cancel_invoice",
        "is_negative_price",
        "is_zero_price",
        "has_missing_customer_id",
    ]

    print("\n--- Individual flag counts ---")
    for flag in flags:
        print(f"{flag}: {int(df[flag].sum()):,}")

    print("\n--- Key overlaps ---")
    pairs = [
        ("is_negative_quantity", "is_cancel_invoice"),
        ("is_exact_duplicate", "is_negative_quantity"),
        ("is_exact_duplicate", "is_cancel_invoice"),
        ("is_negative_quantity", "is_negative_price"),
        ("is_negative_quantity", "is_zero_price"),
    ]

    for first, second in pairs:
        count = int((df[first] & df[second]).sum())
        print(f"{first} AND {second}: {count:,}")

    print("\n--- Invoice-level cancellation check ---")
    invoice_groups = df.groupby("Invoice", dropna=False).agg(
        has_negative_quantity=("is_negative_quantity", "any"),
        has_cancel_invoice=("is_cancel_invoice", "any"),
        transaction_rows=("Invoice", "size"),
    )

    print(
        "Invoices with negative quantity: "
        f"{int(invoice_groups['has_negative_quantity'].sum()):,}"
    )
    print(
        "Invoices with C-prefixed invoice number: "
        f"{int(invoice_groups['has_cancel_invoice'].sum()):,}"
    )
    print(
        "Invoices with both signals: "
        f"{int((invoice_groups['has_negative_quantity'] & invoice_groups['has_cancel_invoice']).sum()):,}"
    )

    print("\nAnalysis complete. No rows were removed.")


if __name__ == "__main__":
    main()
