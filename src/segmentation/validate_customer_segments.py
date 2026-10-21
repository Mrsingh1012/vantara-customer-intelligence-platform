from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEGMENTS_FILE = ROOT / "data" / "processed" / "customer_segments.csv"
RFM_FILE = ROOT / "data" / "processed" / "customer_rfm.csv"
SUMMARY_FILE = ROOT / "reports" / "customer_segment_summary.csv"


def main():
    print("\n--- Customer Segmentation Validation ---")

    segments = pd.read_csv(SEGMENTS_FILE)
    rfm = pd.read_csv(RFM_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    required = {
        "Customer_ID",
        "recency_days",
        "frequency",
        "monetary",
        "cluster",
    }

    missing = required - set(segments.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    checks = {
        "Segment customer IDs are unique":
            segments["Customer_ID"].is_unique,
        "No missing customer IDs":
            segments["Customer_ID"].notna().all(),
        "No missing cluster labels":
            segments["cluster"].notna().all(),
        "Exactly 3 clusters":
            segments["cluster"].nunique() == 3,
        "Every RFM customer is represented":
            set(rfm["Customer_ID"]) == set(segments["Customer_ID"]),
        "No missing RFM feature values":
            segments[["recency_days", "frequency", "monetary"]]
            .notna().all().all(),
        "All cluster labels are 0, 1, 2":
            set(segments["cluster"].unique()) == {0, 1, 2},
        "Cluster summary counts match":
            summary["customers"].sum() == len(segments),
    }

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\nTotal customers: {len(segments):,}")
    print(f"Cluster counts:\n{segments['cluster'].value_counts().sort_index()}")

    if not all(checks.values()):
        raise AssertionError("Validation failed. Review the checks above.")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
