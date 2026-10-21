from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "transactions_cleaned.csv"
REPORT_DIR = ROOT / "reports"

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def build_dataset(df, obs_start, obs_end, future_start, future_end):
    eligible = df[
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["Invoice"].astype(str).str.startswith("C")
    ].copy()

    eligible["Customer_ID"] = (
        eligible["Customer_ID"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    eligible["date"] = eligible["InvoiceDate"].dt.normalize()
    eligible["purchase_value"] = eligible["Quantity"] * eligible["Price"]

    observation = eligible[
        eligible["date"].between(obs_start, obs_end)
    ]
    future = eligible[
        eligible["date"].between(future_start, future_end)
    ]

    if observation.empty or future.empty:
        raise ValueError("Observation or future-label period is empty.")

    data = observation.groupby("Customer_ID").agg(
        frequency=("Invoice", "nunique"),
        monetary=("purchase_value", "sum"),
        purchase_rows=("Invoice", "size"),
        last_purchase_date=("date", "max"),
    ).reset_index()

    reference_date = obs_end + pd.Timedelta(days=1)
    data["recency_days"] = (
        reference_date - data["last_purchase_date"]
    ).dt.days

    future_ids = set(future["Customer_ID"])
    data["churn_label"] = (
        ~data["Customer_ID"].isin(future_ids)
    ).astype(int)

    return data.drop(columns=["last_purchase_date"])


def evaluate_thresholds(y_true, probabilities, period):
    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "period": period,
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

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    # Training features and their following 90-day labels.
    train = build_dataset(
        df,
        pd.Timestamp("2009-12-01"),
        pd.Timestamp("2010-05-31"),
        pd.Timestamp("2010-06-01"),
        pd.Timestamp("2010-08-29"),
    )

    # Used to select the threshold, not to fit the model.
    validation = build_dataset(
        df,
        pd.Timestamp("2010-09-01"),
        pd.Timestamp("2010-11-30"),
        pd.Timestamp("2010-12-01"),
        pd.Timestamp("2011-02-28"),
    )

    # Later evaluation period. Do not select the threshold from these results.
    test = build_dataset(
        df,
        pd.Timestamp("2011-03-01"),
        pd.Timestamp("2011-05-31"),
        pd.Timestamp("2011-06-01"),
        pd.Timestamp("2011-08-29"),
    )

    for name, dataset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        if dataset["churn_label"].nunique() < 2:
            raise ValueError(f"{name} data does not contain both classes.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(train[FEATURES], train["churn_label"])

    validation_probabilities = model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    validation_results = evaluate_thresholds(
        validation["churn_label"].to_numpy(),
        validation_probabilities,
        "validation",
    )

    # Tie-break by higher precision if F1 scores are equal.
    best = validation_results.sort_values(
        ["f1", "precision"],
        ascending=[False, False],
    ).iloc[0]
    selected_threshold = float(best["threshold"])

    # Evaluate the chosen threshold on the later test period.
    test_probabilities = model.predict_proba(test[FEATURES])[:, 1]
    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)
    y_test = test["churn_label"].to_numpy()

    test_metrics = pd.DataFrame([{
        "selected_threshold": selected_threshold,
        "test_customers": len(test),
        "precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
        "f1": f1_score(y_test, test_predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_probabilities),
        "predicted_churn_count": int(test_predictions.sum()),
    }])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    validation_path = REPORT_DIR / "churn_threshold_validation_v2.csv"
    test_path = REPORT_DIR / "churn_threshold_test_v2.csv"

    validation_results.to_csv(validation_path, index=False)
    test_metrics.to_csv(test_path, index=False)

    print("\n--- Threshold Selection: Validation Period ---")
    print(validation_results.round(4).to_string(index=False))
    print(f"\nSelected threshold: {selected_threshold:.2f}")
    print(f"Validation report: {validation_path}")

    print("\n--- Later Test Period ---")
    print(test_metrics.round(4).to_string(index=False))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, test_predictions))
    print(f"Test report: {test_path}")
    print(
        "\nCaution: the later test period was previously inspected "
        "during threshold analysis. Treat its metrics as exploratory, "
        "not as an untouched final holdout."
    )


if __name__ == "__main__":
    main()
