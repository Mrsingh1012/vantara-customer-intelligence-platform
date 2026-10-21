from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "churn_dataset.csv"
SUMMARY_FILE = ROOT / "reports" / "churn_label_summary.csv"

OBS_START = pd.Timestamp("2009-12-01")
OBS_END = pd.Timestamp("2010-12-08")
FUTURE_START = pd.Timestamp("2010-12-09")
FUTURE_END = pd.Timestamp("2011-03-08")


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    eligible = df[
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["Invoice"].astype(str).str.startswith("C")
    ].copy()

    eligible["Customer_ID"] = (
        eligible["Customer_ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    eligible["date"] = eligible["InvoiceDate"].dt.normalize()
    eligible["purchase_value"] = eligible["Quantity"] * eligible["Price"]

    observation = eligible[
        eligible["date"].between(OBS_START, OBS_END)
    ].copy()

    future = eligible[
        eligible["date"].between(FUTURE_START, FUTURE_END)
    ].copy()

    if observation.empty or future.empty:
        raise ValueError("Observation or prediction period is empty.")

    reference_date = OBS_END + pd.Timedelta(days=1)

    features = observation.groupby("Customer_ID").agg(
        last_purchase_date=("date", "max"),
        frequency=("Invoice", "nunique"),
        monetary=("purchase_value", "sum"),
        purchase_rows=("Invoice", "size"),
    ).reset_index()

    features["recency_days"] = (
        reference_date - features["last_purchase_date"]
    ).dt.days

    future_customers = set(future["Customer_ID"].unique())
    features["churn_label"] = (
        ~features["Customer_ID"].isin(future_customers)
    ).astype(int)

    features = features.drop(columns=["last_purchase_date"])

    if features["Customer_ID"].duplicated().any():
        raise AssertionError("Duplicate customer IDs found.")

    if features.isna().any().any():
        raise AssertionError("Missing values found in churn dataset.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(OUTPUT_FILE, index=False)

    summary = (
        features["churn_label"]
        .value_counts()
        .rename_axis("churn_label")
        .reset_index(name="customers")
    )
    summary["percentage"] = (
        summary["customers"] / len(features) * 100
    ).round(2)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n--- Churn Dataset Created ---")
    print(f"Dataset: {OUTPUT_FILE}")
    print(f"Label summary: {SUMMARY_FILE}")
    print(f"Total customers: {len(features):,}")
    print("\nLabel distribution:")
    print(summary.to_string(index=False))
    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nReminder: churn means no eligible purchase in the defined 90-day horizon.")


if __name__ == "__main__":
    main()
