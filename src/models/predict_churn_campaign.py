from pathlib import Path
import argparse

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models_artifacts" / "churn_random_forest.joblib"
FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]


def predict_campaign(input_path, output_path, top_k, model_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    model_path = Path(model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input customer file not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    customers = pd.read_csv(input_path, dtype={"customer_id": "string"})

    required = ["customer_id", *FEATURES]
    missing = [column for column in required if column not in customers.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Required columns: {required}"
        )

    if customers["customer_id"].isna().any():
        raise ValueError("Input contains missing customer_id values")
    if customers["customer_id"].duplicated().any():
        raise ValueError(
            "Input contains duplicate customer_id values. "
            "Provide one feature row per customer."
        )

    for column in FEATURES:
        customers[column] = pd.to_numeric(customers[column], errors="coerce")

    if customers[FEATURES].isna().any().any():
        raise ValueError("Feature columns contain missing or non-numeric values")
    if not customers[FEATURES].apply(
        lambda col: col.map(lambda value: pd.notna(value) and abs(value) != float("inf"))
    ).all().all():
        raise ValueError("Feature columns contain infinite values")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Unexpected model artifact format")
    model = artifact["model"]

    probabilities = model.predict_proba(customers[FEATURES])[:, 1]

    results = pd.DataFrame({
        "customer_id": customers["customer_id"].astype(str),
        "churn_probability": probabilities,
    })

    results = results.sort_values(
        "churn_probability", ascending=False, kind="stable"
    ).reset_index(drop=True)

    selected_count = min(top_k, len(results))
    results["rank"] = range(1, len(results) + 1)
    results["selected_for_campaign"] = results["rank"] <= selected_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\n--- Churn Prediction Summary ---")
    print(f"Input customers: {len(results):,}")
    print(f"Selected for campaign: {selected_count:,}")
    print(f"Unique customer IDs: {results['customer_id'].nunique():,}")
    print(f"Model: {artifact.get('model_name', type(model).__name__)}")
    print(f"Output: {output_path}")
    print("\nTop 10 ranked customers:")
    print(results.head(10).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Score customers and rank a churn-retention campaign list."
    )
    parser.add_argument("--input", required=True, help="Customer features CSV")
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "churn_campaign_ranked.csv"),
        help="Output ranked CSV path",
    )
    parser.add_argument(
        "--top-k", type=int, default=1500,
        help="Maximum number of customers selected for the campaign",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to saved model artifact",
    )
    args = parser.parse_args()

    predict_campaign(args.input, args.output, args.top_k, args.model)


if __name__ == "__main__":
    main()
