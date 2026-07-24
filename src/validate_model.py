"""
Model validation gate for CI/CD.

Checks the most recently registered model version's ROC-AUC against a
minimum threshold. Writes `passed=true|false` to $GITHUB_OUTPUT so the
GitHub Actions workflow can block deployment on failure.

Usage:
    python src/validate_model.py --min-roc-auc 0.70
"""

import argparse
import os
import sys

import mlflow

REGISTERED_MODEL_NAME = "churn-classifier"


def get_latest_version_metrics(client: mlflow.tracking.MlflowClient):
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No versions found for model '{REGISTERED_MODEL_NAME}'")

    latest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)
    return latest.version, run.data.metrics


def write_github_output(passed: bool):
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as f:
            f.write(f"passed={'true' if passed else 'false'}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate latest model against a metric gate")
    parser.add_argument("--min-roc-auc", type=float, default=0.70)
    parser.add_argument("--metric", default="roc_auc")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = mlflow.tracking.MlflowClient()

    version, metrics = get_latest_version_metrics(client)
    score = metrics.get(args.metric)

    if score is None:
        print(f"Metric '{args.metric}' not found on model version {version}")
        write_github_output(False)
        sys.exit(1)

    passed = score >= args.min_roc_auc
    status = "PASSED" if passed else "FAILED"
    print(f"Model v{version} — {args.metric}={score:.4f} (threshold={args.min_roc_auc}) — {status}")

    write_github_output(passed)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()