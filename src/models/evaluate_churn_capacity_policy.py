from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "transactions_cleaned.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]
CAPACITY = 1500
SELECTED_THRESHOLD = 0.524

PERIODS = {
    "train": ("2009-12-01", "2010-05-31", "2010-06-01", "2010-08-29"),
    "test": ("2011-03-01", "2011-05-31", "2011-06-01", "2011-08-29"),
}


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_col(columns, candidates):
    lookup = {normalize(c): c for c in columns}
    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]
    raise ValueError(f"Missing column {candidates}; found {list(columns)}")


def load_transactions():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    customer = find_col(df.columns, ["Customer_ID", "Customer ID", "CustomerID"])
    date = find_col(df.columns, ["InvoiceDate", "Invoice Date", "Date"])
    quantity = find_col(df.columns, ["Quantity", "Qty"])
    price = find_col(df.columns, ["Price", "UnitPrice", "Unit Price"])
    invoice = find_col(
        df.columns, ["Invoice", "InvoiceNo", "Invoice No", "InvoiceNumber"]
    )

    df = df.rename(columns={
        customer: "customer_id",
        date: "invoice_date",
        quantity: "quantity",
        price: "price",
    })
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["customer_id", "invoice_date", "quantity", "price"])
    df = df[
        (df["quantity"] > 0)
        & (df["price"] > 0)
        & (df["customer_id"].astype(str).str.strip() != "")
    ].copy()
    df = df[
        ~df[invoice].astype(str).str.strip().str.upper().str.startswith("C")
    ].copy()

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["invoice_id"] = df[invoice].astype(str)
    df["row_value"] = df["quantity"] * df["price"]
    return df


def build_cohort(df, period):
    obs_start, obs_end, label_start, label_end = map(pd.Timestamp, period)

    obs = df[
        (df["invoice_date"] >= obs_start)
        & (df["invoice_date"] < obs_end + pd.Timedelta(days=1))
    ]
    future = df[
        (df["invoice_date"] >= label_start)
        & (df["invoice_date"] < label_end + pd.Timedelta(days=1))
    ]

    cohort = obs.groupby("customer_id").agg(
        frequency=("invoice_id", "nunique"),
        monetary=("row_value", "sum"),
        purchase_rows=("invoice_id", "size"),
        last_purchase=("invoice_date", "max"),
    )
    cohort["recency_days"] = (
        obs_end.normalize() - cohort["last_purchase"].dt.normalize()
    ).dt.days
    future_customers = set(future["customer_id"].unique())
    cohort["churn_label"] = (
        ~cohort.index.isin(future_customers)
    ).astype(int)
    return cohort.reset_index()


def metrics_row(name, y_true, pred, probabilities, threshold):
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()
    return {
        "policy": name,
        "threshold_or_cutoff": threshold,
        "test_customers": len(y_true),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "predicted_churn_count": int(pred.sum()),
    }


def main():
    print("Building train and test cohorts...")
    transactions = load_transactions()
    train = build_cohort(transactions, PERIODS["train"])
    test = build_cohort(transactions, PERIODS["test"])

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(train[FEATURES], train["churn_label"])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    y_true = test["churn_label"].to_numpy()

    # Policy A: fixed validation-selected threshold.
    fixed_pred = (probabilities >= SELECTED_THRESHOLD).astype(int)

    # Policy B: strict capacity; contact the top-K customers by risk.
    k = min(CAPACITY, len(test))
    top_indices = np.argsort(probabilities, kind="stable")[-k:]
    top_pred = np.zeros(len(test), dtype=int)
    top_pred[top_indices] = 1
    cutoff = float(probabilities[top_indices].min())

    results = pd.DataFrame([
        metrics_row(
            "Fixed threshold 0.524", y_true, fixed_pred,
            probabilities, SELECTED_THRESHOLD
        ),
        metrics_row(
            f"Top {k} customers", y_true, top_pred,
            probabilities, cutoff
        ),
    ])

    results_path = REPORTS / "churn_rf_capacity_policy_test.csv"
    results.to_csv(results_path, index=False)

    predictions = pd.DataFrame({
        "customer_id": test["customer_id"],
        "actual_churn": y_true,
        "churn_probability": probabilities,
        "fixed_threshold_prediction": fixed_pred,
        "top_capacity_prediction": top_pred,
    })
    predictions = predictions.sort_values(
        "churn_probability", ascending=False
    )
    predictions.to_csv(
        REPORTS / "churn_rf_capacity_policy_predictions.csv", index=False
    )

    print(f"\nTest customers: {len(test):,}")
    print(f"Actual churn labels: {int(y_true.sum()):,}")
    print(f"Top-K risk cutoff: {cutoff:.6f}")
    print("\n--- Test Policy Comparison ---")
    print(results.round(4).to_string(index=False))
    print(f"\nSaved: {results_path}")
    print("Saved: reports/churn_rf_capacity_policy_predictions.csv")
    print(
        "\nCaution: this test period was inspected in earlier experiments. "
        "Treat these results as exploratory, not independent final validation."
    )


if __name__ == "__main__":
    main()
    