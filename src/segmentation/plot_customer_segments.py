from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "reports"


def main():
    df = pd.read_csv(INPUT_FILE)

    summary = df.groupby("cluster").agg(
        customers=("Customer_ID", "count"),
        recency=("recency_days", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
    ).reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    charts = [
        ("customers", "Number of Customers", axes[0, 0]),
        ("recency", "Median Recency (Days)", axes[0, 1]),
        ("frequency", "Median Purchase Frequency", axes[1, 0]),
        ("monetary", "Median Monetary Value (£)", axes[1, 1]),
    ]

    for column, title, ax in charts:
        ax.bar(summary["cluster"].astype(str), summary[column])
        ax.set_title(title)
        ax.set_xlabel("Cluster")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Vantara Customer Segmentation: K-Means (K=3)")
    fig.tight_layout()

    output_file = OUTPUT_DIR / "customer_segments_overview.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n--- Segmentation Visualization Complete ---")
    print(f"Customers: {len(df):,}")
    print(f"Chart saved: {output_file}")
    print("\nSegment comparison:")
    print(summary.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
