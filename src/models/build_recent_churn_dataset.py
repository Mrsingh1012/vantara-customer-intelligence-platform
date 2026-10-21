from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "transactions_cleaned.csv"
OUTPUT = ROOT / "data" / "processed" / "churn_dataset_recent.csv"
REPORT = ROOT / "reports" / "churn_label_summary_recent.csv"

df = pd.read_csv(INPUT, parse_dates=["InvoiceDate"])

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

obs_end = pd.Timestamp("2011-08-31")
future_start = pd.Timestamp("2011-09-01")
future_end = pd.Timestamp("2011-11-29")
obs_start = pd.Timestamp("2010-12-01")

obs = eligible[eligible["date"].between(obs_start, obs_end)].copy()
future = eligible[eligible["date"].between(future_start, future_end)].copy()

if obs.empty or future.empty:
    raise ValueError("Observation or future-label period is empty.")

features = obs.groupby("Customer_ID").agg(
    last_purchase_date=("date", "max"),
    frequency=("Invoice", "nunique"),
    monetary=("purchase_value", "sum"),
    purchase_rows=("Invoice", "size"),
).reset_index()

features["recency_days"] = (
    obs_end + pd.Timedelta(days=1) - features["last_purchase_date"]
).dt.days

future_ids = set(future["Customer_ID"].unique())
features["churn_label"] = (
    ~features["Customer_ID"].isin(future_ids)
).astype(int)

features = features.drop(columns=["last_purchase_date"])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUTPUT, index=False)

summary = (
    features["churn_label"].value_counts()
    .rename_axis("churn_label")
    .reset_index(name="customers")
)
summary["percentage"] = (summary["customers"] / len(features) * 100).round(2)
summary.to_csv(REPORT, index=False)

print("Observation:", obs_start.date(), "to", obs_end.date())
print("Label horizon:", future_start.date(), "to", future_end.date())
print("Customers:", len(features))
print(summary.to_string(index=False))
print("Dataset saved:", OUTPUT)
print("Summary saved:", REPORT)
print("Diagnostic dataset only; do not treat as independent final validation.")
