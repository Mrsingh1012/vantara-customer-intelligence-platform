import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


class TestVantaraReports(unittest.TestCase):

    def test_campaign_report_exists(self):
        self.assertTrue(
            (REPORTS / "churn_campaign_ranked_demo.csv").exists()
        )

    def test_customer_ids_are_unique(self):
        df = pd.read_csv(REPORTS / "churn_campaign_ranked_demo.csv")
        self.assertEqual(df["customer_id"].nunique(), len(df))
        self.assertFalse(df["customer_id"].isna().any())

    def test_campaign_capacity(self):
        df = pd.read_csv(REPORTS / "churn_campaign_ranked_demo.csv")
        selected = df["selected_for_campaign"].astype(str).str.lower().eq("true")
        self.assertEqual(int(selected.sum()), 1500)

    def test_probabilities_are_valid(self):
        df = pd.read_csv(REPORTS / "churn_campaign_ranked_demo.csv")
        probabilities = pd.to_numeric(df["churn_probability"], errors="coerce")
        self.assertFalse(probabilities.isna().any())
        self.assertTrue(probabilities.between(0, 1).all())

    def test_business_impact_report(self):
        path = REPORTS / "churn_campaign_business_impact.csv"
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        required = {
            "policy",
            "assumed_retention_success_rate",
            "estimated_campaign_cost_inr",
            "estimated_retained_value_inr",
            "estimated_net_value_inr",
        }
        self.assertTrue(required.issubset(df.columns))
        self.assertEqual(len(df), 6)


if __name__ == "__main__":
    unittest.main()
