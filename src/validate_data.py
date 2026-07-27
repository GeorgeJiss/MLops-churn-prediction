"""
Data validation for the Telco Customer Churn dataset using Pandera.

Usage:
    python src/validate_data.py --path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------
# Reflects the raw Kaggle Telco Churn columns. Adjust names/types if your
# CSV headers differ.

schema = DataFrameSchema(
    {
        "customerID": Column(str, unique=True, nullable=False),
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.in_range(0, 100), nullable=False),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(str, nullable=False),
        "InternetService": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"])
        ),
        "OnlineSecurity": Column(str, nullable=False),
        "OnlineBackup": Column(str, nullable=False),
        "DeviceProtection": Column(str, nullable=False),
        "TechSupport": Column(str, nullable=False),
        "StreamingTV": Column(str, nullable=False),
        "StreamingMovies": Column(str, nullable=False),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(str, nullable=False),
        "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0)),
        # TotalCharges arrives as object/string in the raw CSV (has blanks
        # for customers with 0 tenure) — coerce happens in the loader below.
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "Churn": Column(str, Check.isin(["Yes", "No"]), nullable=False),
    },
    checks=[
        # Cross-column sanity check: TotalCharges should roughly track
        # tenure * MonthlyCharges (loose bound to allow billing variance).
        Check(
            lambda df: (
                df["TotalCharges"].isna()
                | (df["TotalCharges"] >= df["MonthlyCharges"] * df["tenure"] * 0.5)
            ),
            error="TotalCharges is implausibly low relative to tenure and MonthlyCharges",
        )
    ],
    strict=False,  # allow extra columns without failing (e.g. future feature adds)
    coerce=True,
)


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw CSV and apply the minimal cleaning validation expects."""
    df = pd.read_csv(path)

    # TotalCharges has blank strings for new customers (tenure=0) in the raw file.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def validate(path: str) -> pd.DataFrame:
    """Load, clean, and validate the dataset. Raises SchemaError on failure."""
    df = load_and_clean(path)
    validated_df = schema.validate(df, lazy=True)
    return validated_df


def main():
    parser = argparse.ArgumentParser(description="Validate Telco Churn raw data")
    parser.add_argument(
        "--path",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the raw CSV file",
    )
    args = parser.parse_args()

    try:
        df = validate(args.path)
    except pa.errors.SchemaErrors as e:
        print("❌ Validation FAILED\n")
        print(e.failure_cases.to_string(index=False))
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ File not found: {args.path}")
        sys.exit(1)

    print(f"✅ Validation PASSED — {len(df)} rows, {len(df.columns)} columns")
    print(f"   Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print(f"   Missing TotalCharges: {df['TotalCharges'].isna().sum()} rows")


if __name__ == "__main__":
    main()
