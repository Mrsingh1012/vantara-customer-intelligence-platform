from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "interim" / "transactions_cleaned.csv"

def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["InvoiceDate"])

    eligible = df[
        df["Customer_ID"].notna()
        & (df["Quantity"] > 0)
        & (df["Price"] > 0)
        & ~df["Invoice"].astype(str).str.startswith("C")
    ].copy()

    eligible["Customer_ID"] = (
        eligible["Customer_ID"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    eligible["date"] = eligible["InvoiceDate"].dt.normalize()

    observation_start = pd.Timestamp("2009-12-01")
    observation_end = pd.Timestamp("2010-12-08")
    prediction_start = pd.Timestamp("2010-12-09")
    prediction_end = pd.Timestamp("2011-03-08")

    observation = eligible[
        eligible["date"].between(observation_start, observation_end)
    ]

    prediction = eligible[
        eligible["date"].between(prediction_start, prediction_end)
    ]

    active_customers = set(observation["Customer_ID"].unique())
    future_customers = set(prediction["Customer_ID"].unique())

    churn_candidates = active_customers - future_customers
    retained_customers = active_customers & future_customers

    print("\n--- Churn Cutoff Validation ---")
    print(f"Observation period: {observation_start.date()} to {observation_end.date()}")
    print(f"Prediction period:  {prediction_start.date()} to {prediction_end.date()}")
    print(f"Observation transactions: {len(observation):,}")
    print(f"Prediction transactions:  {len(prediction):,}")
    print(f"Customers active in observation period: {len(active_customers):,}")
    print(f"Customers with future purchases: {len(retained_customers):,}")
    print(f"Potential churn labels: {len(churn_candidates):,}")
    print(f"Customers without future purchases: {len(churn_candidates):,}")

    if not observation.empty and not prediction.empty:
        print("\nCutoff periods contain eligible purchases.")
    else:
        print("\nWARNING: One or both periods have no eligible purchases.")

    print("This is a preliminary count only, not a final model dataset.")

if __name__ == "__main__":
    main()
