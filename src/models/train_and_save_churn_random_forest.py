from pathlib import Path
import re
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "transactions_cleaned.csv"
ARTIFACT_DIR = ROOT / "models_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = ARTIFACT_DIR / "churn_random_forest.joblib"
FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]

TRAIN_PERIOD = (
    "2009-12-01",
    "2010-05-31",
    "2010-06-01",
    "2010-08-29",
)


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns, candidates):
    lookup = {normalize(c): c for c in columns}
    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]
    raise ValueError(
        f"Missing required column: {candidates}. "
        f"Available columns: {list(columns)}"
    )


def load_transactions():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    customer_col = find_column(
        df.columns, ["Customer_ID", "Customer ID", "CustomerID"]
    )
    date_col = find_column(
        df.columns, ["InvoiceDate", "Invoice Date", "Date"]
    )
    quantity_col = find_column(df.columns, ["Quantity", "Qty"])
    price_col = find_column(df.columns, ["Price", "UnitPrice", "Unit Price"])
    invoice_col = find_column(
        df.columns, ["Invoice", "InvoiceNo", "Invoice No", "InvoiceNumber"]
    )

    df = df.rename(columns={
        customer_col: "customer_id",
        date_col: "invoice_date",
        quantity_col: "quantity",
        price_col: "price",
    })

    df["invoice_date"] = pd.to_datetime(
        df["invoice_date"], errors="coerce"
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(
        subset=["customer_id", "invoice_date", "quantity", "price"]
    )

    df = df[
        (df["quantity"] > 0)
        & (df["price"] > 0)
        & (df["customer_id"].astype(str).str.strip() != "")
    ].copy()

    df = df[
        ~df[invoice_col].astype(str).str.strip().str.upper().str.startswith("C")
    ].copy()

    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df["invoice_id"] = df[invoice_col].astype(str)
    df["row_value"] = df["quantity"] * df["price"]
    return df


def build_training_cohort(df):
    obs_start, obs_end, label_start, label_end = map(
        pd.Timestamp, TRAIN_PERIOD
    )

    observation = df[
        (df["invoice_date"] >= obs_start)
        & (df["invoice_date"] < obs_end + pd.Timedelta(days=1))
    ]

    future = df[
        (df["invoice_date"] >= label_start)
        & (df["invoice_date"] < label_end + pd.Timedelta(days=1))
    ]

    cohort = observation.groupby("customer_id").agg(
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
    print("Loading data...")
    transactions = load_transactions()
    train = build_training_cohort(transactions)

    X_train = train[FEATURES]
    y_train = train["churn_label"]

    print(f"Training customers: {len(train):,}")
    print(f"Churn labels: {int(y_train.sum()):,}")
    print(f"Non-churn labels: {int((y_train == 0).sum()):,}")

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

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    artifact = {
        "model": model,
        "features": FEATURES,
        "model_name": "Random Forest",
        "training_period": {
            "observation_start": TRAIN_PERIOD[0],
            "observation_end": TRAIN_PERIOD[1],
            "label_start": TRAIN_PERIOD[2],
            "label_end": TRAIN_PERIOD[3],
        },
        "selection_policy": "Rank customers by churn probability; select top K",
        "campaign_capacity": 1500,
        "random_state": 42,
    }

    joblib.dump(artifact, OUTPUT_PATH)

    # Reload and perform a smoke test.
    saved = joblib.load(OUTPUT_PATH)
    probabilities = saved["model"].predict_proba(X_train.head(5))[:, 1]

    print(f"\nSaved artifact: {OUTPUT_PATH}")
    print(f"Saved model: {type(saved['model'].named_steps['model']).__name__}")
    print(f"Features: {saved['features']}")
    print(f"Smoke test probabilities: {probabilities.round(4).tolist()}")
    print("Artifact reload and prediction smoke test passed.")
    print(
        "\nNote: This artifact uses the historical training cohort. "
        "It is for project development, not current-customer deployment."
    )


if __name__ == "__main__":
    main()
