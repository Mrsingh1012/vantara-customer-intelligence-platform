from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "churn_dataset.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    feature_columns = [
        "frequency",
        "monetary",
        "purchase_rows",
        "recency_days",
    ]

    print("\n--- Churn Dataset Validation ---")

    checks = {
        "Dataset contains 4,309 customers": len(df) == 4309,
        "Customer IDs are unique": df["Customer_ID"].is_unique,
        "No missing values": not df.isna().any().any(),
        "Labels contain only 0 and 1":
            set(df["churn_label"].unique()).issubset({0, 1}),
        "Both label classes exist": df["churn_label"].nunique() == 2,
        "Features are non-negative":
            (df[feature_columns] >= 0).all().all(),
        "Customer ID excluded from model features":
            "Customer_ID" not in feature_columns,
        "Label excluded from model features":
            "churn_label" not in feature_columns,
    }

    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print("\nFeature summary:")
    print(df[feature_columns].describe().round(2).to_string())

    print("\nLabel distribution:")
    print(df["churn_label"].value_counts().sort_index().to_string())

    if not all(checks.values()):
        raise AssertionError("One or more validation checks failed.")

    print("\nAll validation checks passed.")
    print("Note: These checks do not replace a time-based leakage audit.")

if __name__ == "__main__":
    main()
