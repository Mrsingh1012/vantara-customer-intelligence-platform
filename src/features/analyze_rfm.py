
"""Analyze distributions and outliers in the preliminary RFM dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "customer_rfm.csv"
OUTPUT = ROOT / "reports" / "rfm_distribution_summary.csv"


def main() -> None:
    rfm = pd.read_csv(INPUT)

    metrics = ["recency_days", "frequency", "monetary"]
    summary = rfm[metrics].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).T

    summary["skewness"] = rfm[metrics].skew()
    summary["zero_or_negative_count"] = (rfm[metrics] <= 0).sum()

    # IQR-based outlier counts are diagnostic only.
    for column in metrics:
        q1 = rfm[column].quantile(0.25)
        q3 = rfm[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        summary.loc[column, "iqr_outlier_count"] = (
            (rfm[column] < lower) | (rfm[column] > upper)
        ).sum()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT)

    print("\n--- RFM distribution summary ---")
    print(summary.round(3).to_string())
    print(f"\nSaved to: {OUTPUT}")
    print("Outlier flags are diagnostic; no customers were removed.")


if __name__ == "__main__":
    main()
