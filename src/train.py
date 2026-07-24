
import argparse
 
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


EXPERIMENT_NAME = "churn-prediction"
REGISTERED_MODEL_NAME = "churn-classifier"
 
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
 
 
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)  # treat as categorical
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df
 
 
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
 
 
def get_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
        ),
    }
 
 
def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
 
 
def main():
    parser = argparse.ArgumentParser(description="Train churn models with MLflow tracking")
    parser.add_argument(
        "--path", default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
 
    mlflow.set_experiment(EXPERIMENT_NAME)
 
    df = load_data(args.path)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df["Churn"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
 
    preprocessor = build_preprocessor()
    results = {}
 
    for model_name, model in get_models().items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
            pipeline.fit(X_train, y_train)
 
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)
 
            # Log params
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", args.test_size)
            for k, v in model.get_params().items():
                mlflow.log_param(k, v)
 
            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
 
            # Log the full pipeline (preprocessor + model) as one artifact.
            # serialization_format="pickle" is required here because the
            # default "skops" format doesn't trust xgboost's Booster type.
            mlflow.sklearn.log_model(
                pipeline, "pipeline", serialization_format="pickle"
            )
 
            results[model_name] = {
                "run_id": mlflow.active_run().info.run_id,
                "metrics": metrics,
            }
            print(f"{model_name}: {metrics}")
 
    # Pick best by ROC-AUC
    best_model_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_run_id = results[best_model_name]["run_id"]
    print(f"\nBest model: {best_model_name} (run_id={best_run_id})")
 
    # Register best model to the Model Registry, staged as "Staging"
    model_uri = f"runs:/{best_run_id}/pipeline"
    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
 
    # MLflow deprecated stage-based transitions (Staging/Production) in favor
    # of aliases. "challenger" == candidate awaiting promotion to "champion".
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="challenger",
        version=registered.version,
    )
    print(f"Registered {REGISTERED_MODEL_NAME} v{registered.version} -> alias 'challenger'")
 
 
if __name__ == "__main__":
    main()
    