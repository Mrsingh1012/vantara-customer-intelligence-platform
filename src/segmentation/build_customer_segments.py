from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "customer_rfm.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "customer_segments.csv"
SUMMARY_FILE = ROOT / "reports" / "customer_segment_summary.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    features = ["recency_days", "frequency", "monetary"]

    if not set(features).issubset(df.columns):
        raise ValueError(
            f"Required columns not found. Available columns: {list(df.columns)}"
        )

    if df[features].isna().any().any():
        raise ValueError("Missing RFM values found.")

    if (df[features] < 0).any().any():
        raise ValueError("Unexpected negative RFM values found.")

    # Apply the same log transform and scaler used for evaluation.
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    X = df[features].copy()
    X["frequency"] = np.log1p(X["frequency"])
    X["monetary"] = np.log1p(X["monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    df["cluster"] = model.fit_predict(X_scaled)

    summary = df.groupby("cluster").agg(
        customers=("Customer_ID", "count"),
        avg_recency_days=("recency_days", "mean"),
        median_recency_days=("recency_days", "median"),
        avg_frequency=("frequency", "mean"),
        median_frequency=("frequency", "median"),
        avg_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median"),
    ).reset_index()

    summary["customer_share_pct"] = (
        summary["customers"] / len(df) * 100
    ).round(2)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n--- Customer Segmentation Complete ---")
    print(f"Customers segmented: {len(df):,}")
    print(f"Customer segments saved: {OUTPUT_FILE}")
    print(f"Segment summary saved: {SUMMARY_FILE}")
    print("\nSegment profile:")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
