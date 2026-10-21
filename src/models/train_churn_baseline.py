from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "churn_dataset.csv"
MODEL_DIR = ROOT / "models_artifacts"
REPORT_DIR = ROOT / "reports"

FEATURES = ["frequency", "monetary", "purchase_rows", "recency_days"]
TARGET = "churn_label"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    print("\n--- Churn Model Baseline ---")
    print(f"Total customers: {len(df):,}")
    print(f"Training customers: {len(X_train):,}")
    print(f"Test customers: {len(X_test):,}")
    print(f"Churn rate: {y.mean():.2%}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }

        results.append(metrics)
        fitted_models[name] = model

        print(f"\n--- {name} ---")
        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.4f}")

        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print(classification_report(
            y_test, predictions, zero_division=0
        ))

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc", ascending=False
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "churn_baseline_metrics.csv"
    model_path = MODEL_DIR / "churn_baseline_model.joblib"

    results_df.to_csv(metrics_path, index=False)

    best_name = results_df.iloc[0]["model"]
    joblib.dump(fitted_models[best_name], model_path)

    print("\n--- Model Comparison ---")
    print(results_df.round(4).to_string(index=False))
    print(f"\nBest baseline by ROC-AUC: {best_name}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Model saved: {model_path}")
    print("\nNote: This random split is a baseline, not final time-based validation.")


if __name__ == "__main__":
    main()
