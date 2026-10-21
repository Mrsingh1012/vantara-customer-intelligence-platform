from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "churn_customer_features_demo.csv"

OBS_START = pd.Timestamp("2011-03-01")
OBS_END = pd.Timestamp("2011-05-31")


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns, candidates):
    lookup = {normalize(c): c for c in columns}
    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]
    raise ValueError(
        f"Required column not found: {candidates}. "
        f"Available columns: {list(columns)}"
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

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

    df["customer_id"] = (
    df["customer_id"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0+$", "", regex=True)
)
    df["invoice_id"] = df[invoice_col].astype(str)
    df["row_value"] = df["quantity"] * df["price"]

    observation = df[
        (df["invoice_date"] >= OBS_START)
        & (df["invoice_date"] < OBS_END + pd.Timedelta(days=1))
    ].copy()

    if observation.empty:
        raise ValueError("No transactions found in the demo observation window.")

    features = observation.groupby("customer_id").agg(
        frequency=("invoice_id", "nunique"),
        monetary=("row_value", "sum"),
        purchase_rows=("invoice_id", "size"),
        last_purchase=("invoice_date", "max"),
    )

    features["recency_days"] = (
        OBS_END.normalize() - features["last_purchase"].dt.normalize()
    ).dt.days

    features = features.reset_index()
    features = features[
        [
            "customer_id",
            "frequency",
            "monetary",
            "purchase_rows",
            "recency_days",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print("Demo customer feature file created.")
    print(f"Observation window: {OBS_START.date()} to {OBS_END.date()}")
    print(f"Customers: {len(features):,}")
    print(f"Unique customer IDs: {features['customer_id'].nunique():,}")
    print(f"Output: {OUTPUT_PATH}")
    print("\nPreview:")
    print(features.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
