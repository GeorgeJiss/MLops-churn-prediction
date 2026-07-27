"""
Simulate live traffic against the running serving API by replaying real
rows from the dataset through /predict. Populates monitoring/predictions.csv
with enough volume for drift_check.py to produce a meaningful result.

No new dataset needed — this reuses your existing Telco Churn CSV.

Usage:
    python monitoring/generate_traffic.py --path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --n 50
    python monitoring/generate_traffic.py --path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --n 50 --drift
"""

import argparse

import pandas as pd
import requests

FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def load_rows(path: str, n: int, inject_drift: bool) -> list[dict]:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    sample = df.sample(n=min(n, len(df)), random_state=None).copy()

    if inject_drift:
        # Simulate a realistic shift: newer, shorter-tenure, higher-paying
        # customers — e.g. after a pricing change or a new acquisition channel.
        sample["tenure"] = (sample["tenure"] * 0.2).astype(int)
        sample["MonthlyCharges"] = sample["MonthlyCharges"] + 40

    return sample[FEATURE_COLS].to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(description="Replay dataset rows through /predict")
    parser.add_argument("--path", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--drift", action="store_true", help="Inject synthetic drift into the traffic")
    args = parser.parse_args()

    rows = load_rows(args.path, args.n, args.drift)

    ok, failed = 0, 0
    for row in rows:
        try:
            resp = requests.post(args.url, json=row, timeout=5)
            resp.raise_for_status()
            ok += 1
        except requests.RequestException as e:
            failed += 1
            print(f"Request failed: {e}")

    print(f"Sent {ok} predictions successfully, {failed} failed.")
    if args.drift:
        print("Drift injected: tenure compressed, MonthlyCharges shifted +40.")


if __name__ == "__main__":
    main()