from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim" / "transactions_cleaned.csv"
REPORT = ROOT / "reports" / "churn_temporal_model_comparison.csv"

results = pd.read_csv(REPORT)

print("\n--- Existing Model Comparison ---")
print(
    results[
        [
            "model",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_predicted_churn_count",
        ]
    ].to_string(index=False)
)

print(
    "\nNote: This report does not contain per-customer predictions, "
    "so exact confusion matrices cannot be reconstructed from these "
    "aggregate metrics alone."
)
print(
    "Next, we will update the evaluation script to save customer-level "
    "test predictions and calculate TP, FP, TN, FN."
)
