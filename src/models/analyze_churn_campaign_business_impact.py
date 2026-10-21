from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "reports" / "churn_rf_capacity_policy_predictions.csv"
OUTPUT = ROOT / "reports" / "churn_campaign_business_impact.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Scenario analysis for churn retention campaign economics."
    )
    parser.add_argument(
        "--cost-per-contact", type=float, required=True,
        help="Campaign cost per contacted customer, in INR."
    )
    parser.add_argument(
        "--value-per-retained-customer", type=float, required=True,
        help="Estimated business value of one successfully retained customer, INR."
    )
    args = parser.parse_args()

    if args.cost_per_contact < 0 or args.value_per_retained_customer < 0:
        parser.error("Cost and customer value must be non-negative.")

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input report: {INPUT}")

    df = pd.read_csv(INPUT)
    required = {
        "customer_id", "actual_churn",
        "fixed_threshold_prediction", "top_capacity_prediction"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in [
        "actual_churn",
        "fixed_threshold_prediction",
        "top_capacity_prediction",
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if df["customer_id"].nunique() != len(df):
        raise ValueError("Duplicate customer IDs found in input.")

    actual_churners = int(df["actual_churn"].sum())
    policies = {
        "Fixed threshold 0.524": "fixed_threshold_prediction",
        "Top 1500 customers": "top_capacity_prediction",
    }

    rows = []
    for policy, col in policies.items():
        selected = df[df[col] == 1]
        selected_count = len(selected)
        tp = int(selected["actual_churn"].sum())
        fp = selected_count - tp
        fn = actual_churners - tp
        precision = tp / selected_count if selected_count else 0
        recall = tp / actual_churners if actual_churners else 0
        campaign_cost = selected_count * args.cost_per_contact

        for success_rate in [0.05, 0.10, 0.15]:
            expected_retained = tp * success_rate
            expected_value = (
                expected_retained * args.value_per_retained_customer
            )
            rows.append({
                "policy": policy,
                "customers_contacted": selected_count,
                "actual_churners_identified": tp,
                "false_positives": fp,
                "missed_churners": fn,
                "precision": precision,
                "recall": recall,
                "assumed_retention_success_rate": success_rate,
                "expected_retained_customers": expected_retained,
                "assumed_cost_per_contact_inr": args.cost_per_contact,
                "estimated_campaign_cost_inr": campaign_cost,
                "assumed_value_per_retained_customer_inr":
                    args.value_per_retained_customer,
                "estimated_retained_value_inr": expected_value,
                "estimated_net_value_inr": expected_value - campaign_cost,
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)

    print("\n--- Churn Campaign Business Impact: Scenario Analysis ---")
    print(f"Input customers: {len(df):,}")
    print(f"Actual churners in cohort: {actual_churners:,}")
    print(f"Cost per contact: INR {args.cost_per_contact:,.2f}")
    print(
        "Value per successfully retained customer: "
        f"INR {args.value_per_retained_customer:,.2f}"
    )
    print("\nScenario results:")
    print(result[[
        "policy",
        "assumed_retention_success_rate",
        "customers_contacted",
        "expected_retained_customers",
        "estimated_campaign_cost_inr",
        "estimated_retained_value_inr",
        "estimated_net_value_inr",
    ]].round(2).to_string(index=False))
    print(f"\nSaved report: {OUTPUT}")
    print(
        "\nCaution: Retention rates and customer values are assumptions, "
        "not measured outcomes. Validate them with real campaign data."
    )


if __name__ == "__main__":
    main()
