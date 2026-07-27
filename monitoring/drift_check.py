"""
Data drift detection for the churn model — scipy-only implementation.

Same behavior/CLI/output contract as the Evidently-based drift_check.py,
but implemented directly with scipy.stats so it has no dependency on the
`evidently` package. Use this if `evidently` won't import in your
environment (e.g. the Python 3.11.0 typing bug — see gh-98852).

  - Numeric features: two-sample Kolmogorov-Smirnov test
  - Categorical features: chi-square test of independence

Usage:
    python monitoring/drift_check.py \
        --reference data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
        --current monitoring/predictions.csv \
        --drift-share-threshold 0.5
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
P_VALUE_THRESHOLD = 0.05  # below this, a feature is considered drifted

DRIFT_HISTORY_PATH = "monitoring/drift_history.csv"


def load_reference(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    return df[FEATURE_COLS]


def load_current(path: str, min_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} live predictions logged; need at least {min_rows} "
            "before running a meaningful drift check."
        )
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    return df[FEATURE_COLS]


def numeric_drift_p_value(ref_col: pd.Series, cur_col: pd.Series) -> float:
    """Two-sample K-S test — same test Evidently's DataDriftPreset uses for numerics."""
    result = ks_2samp(ref_col.dropna(), cur_col.dropna())
    return float(result.pvalue)


def categorical_drift_p_value(ref_col: pd.Series, cur_col: pd.Series) -> float:
    """Chi-square test on value-count contingency table — matches Evidently's categorical method."""
    ref_counts = ref_col.value_counts()
    cur_counts = cur_col.value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))

    if len(categories) < 2:
        return 1.0  # can't test drift on a single-category column

    contingency = pd.DataFrame(
        {
            "reference": [ref_counts.get(c, 0) for c in categories],
            "current": [cur_counts.get(c, 0) for c in categories],
        }
    )
    _, p_value, _, _ = chi2_contingency(contingency.T)
    return float(p_value)


def run_drift_check(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    p_values = {}
    for col in NUMERIC_COLS:
        p_values[col] = numeric_drift_p_value(reference[col], current[col])
    for col in CATEGORICAL_COLS:
        p_values[col] = categorical_drift_p_value(reference[col], current[col])

    drifted = {col: p for col, p in p_values.items() if p < P_VALUE_THRESHOLD}

    return {
        "drifted_columns_count": float(len(drifted)),
        "drifted_columns_share": len(drifted) / len(p_values),
        "per_column_p_values": p_values,
    }


def append_to_history(summary: dict, n_current_rows: int):
    os.makedirs(os.path.dirname(DRIFT_HISTORY_PATH), exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_current_rows": n_current_rows,
        "drifted_columns_count": summary["drifted_columns_count"],
        "drifted_columns_share": summary["drifted_columns_share"],
    }
    df_row = pd.DataFrame([row])
    file_exists = os.path.isfile(DRIFT_HISTORY_PATH)
    df_row.to_csv(DRIFT_HISTORY_PATH, mode="a", header=not file_exists, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run data drift check (scipy-only)")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", default="monitoring/predictions.csv")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument(
        "--drift-share-threshold",
        type=float,
        default=0.5,
        help="Fraction of features that must drift to flag the run",
    )
    args = parser.parse_args()

    reference = load_reference(args.reference)

    try:
        current = load_current(args.current, args.min_rows)
    except (FileNotFoundError, ValueError) as e:
        print(f"Skipping drift check: {e}")
        sys.exit(0)

    summary = run_drift_check(reference, current)
    append_to_history(summary, len(current))

    print(f"Drifted columns: {int(summary['drifted_columns_count'])} "
          f"({summary['drifted_columns_share']:.1%} of features)")
    for col, p_value in summary["per_column_p_values"].items():
        flag = "DRIFTED" if p_value < P_VALUE_THRESHOLD else "stable"
        print(f"  {col}: p={p_value:.4g} [{flag}]")

    if summary["drifted_columns_share"] >= args.drift_share_threshold:
        print(f"\nDRIFT DETECTED — {summary['drifted_columns_share']:.1%} "
              f">= threshold {args.drift_share_threshold:.1%}")
        sys.exit(1)

    print("\nNo significant drift detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
