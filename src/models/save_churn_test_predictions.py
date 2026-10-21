from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "transactions_cleaned.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.05), 2)

PERIODS = {
    "train": ("2009-12-01", "2010-05-31", "2010-06-01", "2010-08-29"),
    "validation": ("2010-09-01", "2010-11-30", "2010-12-01", "2011-02-28"),
    "test": ("2011-03-01", "2011-05-31", "2011-06-01", "2011-08-29"),
}


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def get_col(columns, candidates):
    lookup = {norm(c): c for c in columns}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    raise ValueError(
        f"Column not found: {candidates}. Available columns: {list(columns)}"
    )


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    customer = get_col(df.columns, ["Customer_ID", "Customer ID", "CustomerID"])
    date = get_col(df.columns, ["InvoiceDate", "Invoice Date", "Date"])
    quantity = get_col(df.columns, ["Quantity", "Qty"])
    price = get_col(df.columns, ["Price", "UnitPrice", "Unit Price"])
    invoice = next(
        (c for c in df.columns if norm(c) in {
            "invoice", "invoiceno", "invoicenumber"
        }),
        None,
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

    if invoice is not None:
        df = df[
            ~df[invoice].astype(str).str.strip().str.upper().str.startswith("C")
        ].copy()
        df["invoice_id"] = df[invoice].astype(str)
    else:
        raise ValueError(
            "Invoice identifier is required to reproduce the previous "
            "frequency feature consistently."
        )

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["row_value"] = df["quantity"] * df["price"]
    return df


def make_cohort(df, name):
    obs_start, obs_end, label_start, label_end = map(
        pd.Timestamp, PERIODS[name]
    )

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

    cohort = cohort.reset_index()
    return cohort[["customer_id", *FEATURES, "churn_label"]]


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def select_threshold(y_true, probabilities):
    scores = []
    for threshold in THRESHOLDS:
        pred = (probabilities >= threshold).astype(int)
        scores.append((
            f1_score(y_true, pred, zero_division=0),
            precision_score(y_true, pred, zero_division=0),
            float(threshold),
        ))
    return max(scores)[2]


def main():
    print("Rebuilding chronological cohorts...")
    df = load_data()
    train = make_cohort(df, "train")
    validation = make_cohort(df, "validation")
    test = make_cohort(df, "test")

    print(
        f"Train: {len(train):,} | Validation: {len(validation):,} "
        f"| Test: {len(test):,}"
    )

    X_train, y_train = train[FEATURES], train["churn_label"]
    X_val, y_val = validation[FEATURES], validation["churn_label"]
    X_test, y_test = test[FEATURES], test["churn_label"]

    prediction_frames = []
    matrix_rows = []

    for name, model in build_models().items():
        print(f"\nEvaluating {name}...")
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_prob)

        test_prob = model.predict_proba(X_test)[:, 1]
        pred = (test_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        matrix_rows.append({
            "model": name,
            "selected_threshold": threshold,
            "test_customers": len(y_test),
            "actual_churn_customers": int(y_test.sum()),
            "actual_non_churn_customers": int((y_test == 0).sum()),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, test_prob),
            "predicted_churn_count": int(pred.sum()),
            "predicted_churn_rate": float(pred.mean()),
        })

        prediction_frames.append(pd.DataFrame({
            "customer_id": test["customer_id"].values,
            "model": name,
            "actual_churn": y_test.values,
            "churn_probability": test_prob,
            "selected_threshold": threshold,
            "predicted_churn": pred,
        }))

    matrix_df = pd.DataFrame(matrix_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    matrix_path = REPORTS / "churn_confusion_matrix.csv"
    predictions_path = REPORTS / "churn_test_customer_predictions.csv"
    matrix_df.to_csv(matrix_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\n--- Confusion Matrix and Business Workload ---")
    print(matrix_df.round(4).to_string(index=False))
    print(f"\nSaved report: {matrix_path}")
    print(f"Saved predictions: {predictions_path}")
    print(
        "\nCaution: The test period was previously inspected, and customers "
        "may overlap between cohorts. Treat these as exploratory results."
    )


if __name__ == "__main__":
    main()
