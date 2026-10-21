
"""Prepare transformed and scaled RFM features for clustering."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "customer_rfm.csv"
OUTPUT = ROOT / "data" / "processed" / "customer_rfm_scaled.csv"
SCALER_PATH = ROOT / "models_artifacts" / "rfm_scaler_mean_scale.csv"


def main() -> None:
    rfm = pd.read_csv(INPUT)

    features = rfm[["recency_days", "frequency", "monetary"]].copy()

    if features.isna().any().any():
        raise ValueError("RFM features contain missing values.")

    if (features <= 0).any().any():
        raise ValueError("All RFM features must be positive for this baseline.")

    # Reduce the impact of long-tailed frequency and monetary values.
    features["frequency"] = np.log1p(features["frequency"])
    features["monetary"] = np.log1p(features["monetary"])

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(features)

    output = rfm[["Customer_ID"]].copy()
    output[["recency_scaled", "frequency_scaled", "monetary_scaled"]] = (
        scaled_values
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT, index=False)

    # Store fitted scaler parameters for reproducibility.
    scaler_info = pd.DataFrame({
        "feature": features.columns,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    })
    scaler_info.to_csv(SCALER_PATH, index=False)

    print("\n--- Clustering feature preparation ---")
    print(f"Customers: {len(output):,}")
    print(f"Missing scaled values: {int(output.isna().sum().sum()):,}")
    print(f"Output saved: {OUTPUT}")
    print(f"Scaler parameters saved: {SCALER_PATH}")
    print("\nScaled feature summary:")
    print(output.iloc[:, 1:].describe().round(3).to_string())


if __name__ == "__main__":
    main()
