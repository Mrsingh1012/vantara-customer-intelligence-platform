from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "transactions_cleaned.csv"


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    print("\n--- Churn Modeling: Data Readiness ---")
    print(f"Total transaction rows: {len(df):,}")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    eligible = df[
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["Invoice"].astype(str).str.startswith("C")
    ].copy()

    eligible["Customer_ID"] = eligible["Customer_ID"].astype(str).str.replace(r"\.0$", "", regex=True)

    last_date = eligible["InvoiceDate"].max()
    customer_last_purchase = eligible.groupby("Customer_ID")["InvoiceDate"].max()
    recency_days = (last_date.normalize() - customer_last_purchase.dt.normalize()).dt.days

    print(f"\nEligible positive-purchase rows: {len(eligible):,}")
    print(f"Customers: {customer_last_purchase.size:,}")
    print(f"Last eligible purchase date: {last_date}")
    print("\nRecency distribution at dataset end:")
    print(recency_days.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2).to_string())

    print("\nImportant: This is a readiness audit only.")
    print("Do not train a churn model until cutoff, prediction horizon,")
    print("and churn definition are explicitly chosen.")


if __name__ == "__main__":
    main()
