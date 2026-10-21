from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "transactions_cleaned.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CAPACITY = 1500
FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]

PERIODS = {
    "train": ("2009-12-01", "2010-05-31", "2010-06-01", "2010-08-29"),
    "validation": ("2010-09-01", "2010-11-30", "2010-12-01", "2011-02-28"),
}


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_col(columns, candidates):
    lookup = {norm(c): c for c in columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    raise ValueError(
        f"Required column not found: {candidates}. "
        f"Available columns: {list(columns)}"
    )


def load_transactions():
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
    df = df.dropna(subset=[
        "customer_id", "invoice_date", "quantity", "price"
    ])

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


def main():
    print("Loading transactions and building validation cohort...")
    df = load_transactions()

    train = build_cohort(df, PERIODS["train"])
    validation = build_cohort(df, PERIODS["validation"])

    print(f"Training customers: {len(train):,}")
    print(f"Validation customers: {len(validation):,}")
    print(f"Campaign capacity: {CAPACITY:,}")

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
    probabilities = model.predict_proba(validation[FEATURES])[:, 1]
    y_true = validation["churn_label"].to_numpy()

    rows = []
    for threshold in np.round(np.arange(0.510, 0.6001, 0.001), 3):
        pred = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "predicted_customers": int(pred.sum()),
            "capacity_remaining": int(CAPACITY - pred.sum()),
            "within_capacity": bool(pred.sum() <= CAPACITY),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })

    results = pd.DataFrame(rows)
    results.to_csv(REPORTS / "churn_rf_capacity_thresholds.csv", index=False)

    eligible = results[results["within_capacity"]]
    if eligible.empty:
        print("No threshold in this range fits campaign capacity.")
        print("Try a higher threshold range.")
        return

    # Maximize campaign use without exceeding 1,500 contacts.
    best = eligible.sort_values(
        ["predicted_customers", "recall", "precision"],
        ascending=[False, False, False],
    ).iloc[0]

    print("\n--- Capacity-Aware Threshold Candidates ---")
    print(
        results[results["threshold"].round(3).isin(
            [0.510, 0.520, 0.530, 0.540, 0.550, 0.560, 0.570, 0.580, 0.590, 0.600]
        )].round(4).to_string(index=False)
    )

    print("\n--- Recommended Validation Threshold ---")
    print(f"Threshold: {best['threshold']:.3f}")
    print(f"Predicted customers: {int(best['predicted_customers'])}")
    print(f"Capacity remaining: {int(best['capacity_remaining'])}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1-score: {best['f1']:.4f}")

    print(
        "\nSaved: reports/churn_rf_capacity_thresholds.csv"
    )
    print(
        "This is a validation-based recommendation, not final test "
        "performance. The validation cohort has 2,887 customers, so its "
        "selected threshold may produce a different count in future campaigns."
    )


if __name__ == "__main__":
    main()
