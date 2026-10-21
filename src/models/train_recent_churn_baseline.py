from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "churn_dataset_recent.csv"
OUTPUT = ROOT / "reports" / "churn_recent_baseline.csv"

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]
TARGET = "churn_label"


def main():
    df = pd.read_csv(INPUT)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results.append({
            "model": name,
            "test_customers": len(y_test),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        })

    result_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT, index=False)

    print("\n--- Recent Churn Dataset: Diagnostic Baseline ---")
    print(result_df.round(4).to_string(index=False))
    print(f"\nSaved: {OUTPUT}")
    print("Warning: random split is diagnostic, not final temporal validation.")


if __name__ == "__main__":
    main()
