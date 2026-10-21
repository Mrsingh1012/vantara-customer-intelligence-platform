from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT_FILE = ROOT / "reports" / "churn_threshold_analysis.csv"

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]


def build_dataset(df, obs_start, obs_end, future_start, future_end):
    data = df[
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["Invoice"].astype(str).str.startswith("C")
    ].copy()

    data["Customer_ID"] = (
        data["Customer_ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    data["date"] = data["InvoiceDate"].dt.normalize()
    data["purchase_value"] = data["Quantity"] * data["Price"]

    obs = data[data["date"].between(obs_start, obs_end)].copy()
    future = data[data["date"].between(future_start, future_end)].copy()

    if obs.empty or future.empty:
        raise ValueError("Observation or future period is empty.")

    result = obs.groupby("Customer_ID").agg(
        last_purchase_date=("date", "max"),
        frequency=("Invoice", "nunique"),
        monetary=("purchase_value", "sum"),
        purchase_rows=("Invoice", "size"),
    ).reset_index()

    result["recency_days"] = (
        obs_end + pd.Timedelta(days=1) - result["last_purchase_date"]
    ).dt.days

    future_ids = set(future["Customer_ID"].unique())
    result["churn_label"] = (
        ~result["Customer_ID"].isin(future_ids)
    ).astype(int)

    return result.drop(columns=["last_purchase_date"])


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    train = build_dataset(
        df,
        pd.Timestamp("2009-12-01"),
        pd.Timestamp("2010-05-31"),
        pd.Timestamp("2010-06-01"),
        pd.Timestamp("2010-08-29"),
    )

    test = build_dataset(
        df,
        pd.Timestamp("2011-03-01"),
        pd.Timestamp("2011-05-31"),
        pd.Timestamp("2011-06-01"),
        pd.Timestamp("2011-08-29"),
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(train[FEATURES], train["churn_label"])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    y_true = test["churn_label"]

    rows = []

    for threshold in [
        0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70
    ]:
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "predicted_churn_count": int(predictions.sum()),
        })

    results = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\n--- Churn Threshold Analysis ---")
    print(results.round(4).to_string(index=False))
    print(f"\nResults saved: {OUTPUT_FILE}")
    print("Note: Thresholds are being compared on the same temporal test set.")
    print("Use this comparison for exploration, not as final independent validation.")


if __name__ == "__main__":
    main()
