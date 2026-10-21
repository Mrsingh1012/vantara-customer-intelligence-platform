from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT_FILE = ROOT / "reports" / "churn_time_validation.csv"

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]


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
    ].copy()

    future = eligible[
        eligible["date"].between(future_start, future_end)
    ].copy()

    if observation.empty or future.empty:
        raise ValueError("Observation or future period has no eligible transactions.")

    reference_date = obs_end + pd.Timedelta(days=1)

    data = observation.groupby("Customer_ID").agg(
        last_purchase_date=("date", "max"),
        frequency=("Invoice", "nunique"),
        monetary=("purchase_value", "sum"),
        purchase_rows=("Invoice", "size"),
    ).reset_index()

    data["recency_days"] = (
        reference_date - data["last_purchase_date"]
    ).dt.days

    future_customers = set(future["Customer_ID"].unique())
    data["churn_label"] = (
        ~data["Customer_ID"].isin(future_customers)
    ).astype(int)

    return data.drop(columns=["last_purchase_date"])


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    # Earlier observation window and its following 90-day label horizon.
    train_data = build_dataset(
        df,
        pd.Timestamp("2009-12-01"),
        pd.Timestamp("2010-06-30"),
        pd.Timestamp("2010-07-01"),
        pd.Timestamp("2010-09-28"),
    )

    # Later, non-overlapping observation and label window.
    test_data = build_dataset(
        df,
        pd.Timestamp("2010-03-01"),
        pd.Timestamp("2010-11-30"),
        pd.Timestamp("2010-12-01"),
        pd.Timestamp("2011-02-28"),
    )

    X_train = train_data[FEATURES]
    y_train = train_data["churn_label"]
    X_test = test_data[FEATURES]
    y_test = test_data["churn_label"]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both train and test datasets must contain both label classes.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train_customers": len(train_data),
        "test_customers": len(test_data),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT_FILE, index=False)

    print("\n--- Time-Based Churn Validation ---")
    print(f"Training observation: 2009-12-01 to 2010-06-30")
    print(f"Training label horizon: 2010-07-01 to 2010-09-28")
    print(f"Testing observation: 2010-03-01 to 2010-11-30")
    print(f"Testing label horizon: 2010-12-01 to 2011-02-28")
    print(f"Training customers: {len(train_data):,}")
    print(f"Testing customers: {len(test_data):,}")

    for key, value in metrics.items():
        if key not in ("train_customers", "test_customers"):
            print(f"{key}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"\nResults saved: {OUTPUT_FILE}")
    print("Note: Customer populations can overlap between windows; this is a temporal baseline, not a fully independent cohort test.")


if __name__ == "__main__":
    main()
