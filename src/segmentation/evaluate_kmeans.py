from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "customer_rfm_scaled.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "kmeans_evaluation.csv"


def main():
    print("\n--- K-Means Customer Segmentation ---")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    feature_columns = [
        "recency_scaled",
        "frequency_scaled",
        "monetary_scaled",
    ]

    missing_columns = [
        col for col in feature_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    X = df[feature_columns]

    if X.isna().any().any():
        raise ValueError("Missing values found in clustering data.")

    results = []

    print(f"Total customers: {len(X):,}")
    print("Evaluating K from 2 to 10...\n")

    for k in range(2, 11):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X)

        score = silhouette_score(
            X,
            labels,
            sample_size=min(3000, len(X)),
            random_state=42,
        )

        sizes = pd.Series(labels).value_counts()

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": score,
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
        })

        print(
            f"K={k} | Inertia={model.inertia_:.2f} | "
            f"Silhouette={score:.4f} | "
            f"Smallest cluster={sizes.min()} | "
            f"Largest cluster={sizes.max()}"
        )

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    best = results_df.loc[results_df["silhouette_score"].idxmax()]

    print("\n--- Results ---")
    print(f"Results saved: {OUTPUT_FILE}")
    print(
        f"Highest sampled Silhouette Score: K={int(best['k'])}, "
        f"Score={best['silhouette_score']:.4f}"
    )
    print("Note: Do not choose K using silhouette score alone.")


if __name__ == "__main__":
    main()
