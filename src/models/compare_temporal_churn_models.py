from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "transactions_cleaned.csv"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.05), 2)

# Each observation window is followed by a 90-day label window.
PERIODS = {
    "train": ("2009-12-01", "2010-05-31", "2010-06-01", "2010-08-29"),
    "validation": ("2010-09-01", "2010-11-30", "2010-12-01", "2011-02-28"),
    "test": ("2011-03-01", "2011-05-31", "2011-06-01", "2011-08-29"),
}


def normalize_column_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(columns, candidates, required=True):
    lookup = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(
            f"Required column not found. Tried {candidates}. "
            f"Available columns: {list(columns)}"
        )
    return None


def load_transactions():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    customer_col = find_column(df.columns, ["Customer_ID", "Customer ID", "CustomerID"])
    date_col = find_column(df.columns, ["InvoiceDate", "Invoice Date", "Date"])
    quantity_col = find_column(df.columns, ["Quantity", "Qty"])
    price_col = find_column(df.columns, ["Price", "UnitPrice", "Unit Price"])
    invoice_col = find_column(
        df.columns, ["Invoice", "InvoiceNo", "Invoice No", "InvoiceNumber"],
        required=False,
    )

    df = df.rename(
        columns={
            customer_col: "customer_id",
            date_col: "invoice_date",
            quantity_col: "quantity",
            price_col: "price",
        }
    )

    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["customer_id", "invoice_date", "quantity", "price"])
    df = df[
        (df["quantity"] > 0)
        & (df["price"] > 0)
        & (df["customer_id"].astype(str).str.strip() != "")
    ].copy()

    # Exclude cancelled invoices when an invoice identifier is available.
    if invoice_col:
        df = df[
            ~df[invoice_col].astype(str).str.strip().str.upper().str.startswith("C")
        ].copy()
        df["invoice_id"] = df[invoice_col].astype(str)
    else:
        # Fallback only if the source file has no invoice identifier.
        df["invoice_id"] = df.index.astype(str)
        print("WARNING: No invoice column found; frequency uses transaction rows.")

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["row_value"] = df["quantity"] * df["price"]
    return df


def build_cohort(df, period):
    obs_start, obs_end, label_start, label_end = map(pd.Timestamp, period)

    observation = df[
        (df["invoice_date"] >= obs_start)
        & (df["invoice_date"] < obs_end + pd.Timedelta(days=1))
    ].copy()

    label_period = df[
        (df["invoice_date"] >= label_start)
        & (df["invoice_date"] < label_end + pd.Timedelta(days=1))
    ]

    if observation.empty:
        raise ValueError(f"No observation data found for {period}")

    cohort = observation.groupby("customer_id").agg(
        frequency=("invoice_id", "nunique"),
        monetary=("row_value", "sum"),
        purchase_rows=("invoice_id", "size"),
        last_purchase=("invoice_date", "max"),
    )

    cohort["recency_days"] = (
        obs_end.normalize() - cohort["last_purchase"].dt.normalize()
    ).dt.days

    future_customers = set(label_period["customer_id"].unique())
    cohort["churn_label"] = (
        ~cohort.index.isin(future_customers)
    ).astype(int)

    cohort = cohort.reset_index()
    cohort["cohort"] = "unknown"
    return cohort[["customer_id", *FEATURES, "churn_label"]]


def build_named_cohort(df, name):
    cohort = build_cohort(df, PERIODS[name])
    cohort["cohort"] = name
    return cohort


def make_models():
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
    }


def choose_threshold(y_true, probabilities):
    rows = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(
                    y_true, predictions, zero_division=0
                ),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "predicted_churn_count": int(predictions.sum()),
            }
        )

    # Select by validation F1; break ties using higher precision.
    rows = sorted(
        rows, key=lambda row: (row["f1"], row["precision"]), reverse=True
    )
    return rows[0], rows


def main():
    print("Loading transactions...")
    transactions = load_transactions()

    cohorts = {
        name: build_named_cohort(transactions, name)
        for name in ("train", "validation", "test")
    }

    for name, cohort in cohorts.items():
        print(
            f"{name.title()}: customers={len(cohort):,}, "
            f"churn_rate={cohort['churn_label'].mean():.2%}"
        )

    train = cohorts["train"]
    validation = cohorts["validation"]
    test = cohorts["test"]

    X_train, y_train = train[FEATURES], train["churn_label"]
    X_val, y_val = validation[FEATURES], validation["churn_label"]
    X_test, y_test = test[FEATURES], test["churn_label"]

    model_results = []
    threshold_results = []
    models = make_models()

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        best_threshold, all_thresholds = choose_threshold(y_val, val_prob)

        for row in all_thresholds:
            threshold_results.append(
                {"model": model_name, **row}
            )

        threshold = best_threshold["threshold"]
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        result = {
            "model": model_name,
            "train_customers": len(train),
            "validation_customers": len(validation),
            "test_customers": len(test),
            "selected_threshold": threshold,
            "validation_precision": best_threshold["precision"],
            "validation_recall": best_threshold["recall"],
            "validation_f1": best_threshold["f1"],
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_f1": f1_score(y_test, test_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, test_prob),
            "test_predicted_churn_count": int(test_pred.sum()),
        }
        model_results.append(result)

        print(
            f"Selected validation threshold: {threshold:.2f} | "
            f"Validation F1: {best_threshold['f1']:.4f}"
        )
        print(
            f"Test precision={result['test_precision']:.4f}, "
            f"recall={result['test_recall']:.4f}, "
            f"F1={result['test_f1']:.4f}, "
            f"ROC-AUC={result['test_roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(model_results).sort_values(
        "test_f1", ascending=False
    )
    threshold_df = pd.DataFrame(threshold_results)

    results_path = REPORTS_DIR / "churn_temporal_model_comparison.csv"
    thresholds_path = REPORTS_DIR / "churn_temporal_thresholds.csv"

    results_df.to_csv(results_path, index=False)
    threshold_df.to_csv(thresholds_path, index=False)

    print("\n--- Temporal Model Comparison ---")
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {thresholds_path}")
    print(
        "\nIMPORTANT: This is exploratory, not an untouched final test. "
        "The test period was inspected in earlier threshold experiments, "
        "and customers may appear in more than one temporal cohort."
    )


if __name__ == "__main__":
    main()
