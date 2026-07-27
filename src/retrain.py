"""
Retrain, validate, and conditionally promote the churn model.

This is the script Airflow's DAG (Phase 6) will call after drift_check.py
signals drift. It:
  1. Retrains all model variants on the current reference data (reuses
     train.py's logic — no duplicated training code)
  2. Registers the best one as a new "challenger" version
  3. Validates it against a minimum ROC-AUC threshold
  4. Compares it against the current "champion" (if one exists)
  5. Promotes challenger -> champion ONLY if it passes the threshold AND
     beats (or ties) the current champion. Otherwise leaves champion
     untouched and the new version sits as "challenger" for manual review.

This gate — never silently replacing a good production model with a worse
one — is the actual point of the exercise, not just "retrain on a schedule."

Usage:
    python src/retrain.py --path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --min-roc-auc 0.70
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train import (
    CATEGORICAL_COLS,
    EXPERIMENT_NAME,
    NUMERIC_COLS,
    REGISTERED_MODEL_NAME,
    build_preprocessor,
    evaluate,
    get_models,
    load_data,
)


def get_champion_metric(client: mlflow.tracking.MlflowClient, metric: str):
    """Return the champion's metric value, or None if no champion exists yet."""
    try:
        champion_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    except Exception:
        return None, None
    run = client.get_run(champion_version.run_id)
    return run.data.metrics.get(metric), champion_version.version


def main():
    parser = argparse.ArgumentParser(description="Retrain, validate, and conditionally promote")
    parser.add_argument("--path", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-roc-auc", type=float, default=0.70)
    parser.add_argument("--metric", default="roc_auc")
    args = parser.parse_args()

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    df = load_data(args.path)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()
    results = {}

    print("Retraining all model variants...")
    for model_name, model in get_models().items():
        with mlflow.start_run(run_name=f"retrain-{model_name}"):
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("trigger", "retrain")
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(pipeline, "pipeline", serialization_format="pickle")

            results[model_name] = {"run_id": mlflow.active_run().info.run_id, "metrics": metrics}
            print(f"  {model_name}: {args.metric}={metrics[args.metric]:.4f}")

    best_model_name = max(results, key=lambda k: results[k]["metrics"][args.metric])
    best_run_id = results[best_model_name]["run_id"]
    best_score = results[best_model_name]["metrics"][args.metric]
    print(f"\nBest retrained model: {best_model_name} ({args.metric}={best_score:.4f})")

    # Register as a new challenger version
    model_uri = f"runs:/{best_run_id}/pipeline"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", registered.version)
    print(f"Registered v{registered.version} -> alias 'challenger'")

    # Gate 1: absolute threshold
    if best_score < args.min_roc_auc:
        print(f"\nREJECTED: {args.metric}={best_score:.4f} < threshold {args.min_roc_auc}")
        print("Champion unchanged. New version remains 'challenger' for manual review.")
        sys.exit(1)

    # Gate 2: must not regress vs current champion
    champion_score, champion_version = get_champion_metric(client, args.metric)
    if champion_score is not None and best_score < champion_score:
        print(f"\nREJECTED: challenger {args.metric}={best_score:.4f} "
              f"< champion v{champion_version} {args.metric}={champion_score:.4f}")
        print("Champion unchanged. New version remains 'challenger' for manual review.")
        sys.exit(1)

    # Promote
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", registered.version)
    if champion_score is not None:
        print(f"\nPROMOTED: v{registered.version} -> 'champion' "
              f"(beat v{champion_version}: {best_score:.4f} >= {champion_score:.4f})")
    else:
        print(f"\nPROMOTED: v{registered.version} -> 'champion' (first champion)")
    sys.exit(0)


if __name__ == "__main__":
    main()