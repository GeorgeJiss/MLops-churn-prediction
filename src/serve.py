"""
FastAPI serving layer for the churn prediction model.

Loads the model tagged with the "champion" alias from the MLflow Model
Registry (falls back to "challenger" if no champion has been promoted yet),
and exposes a /predict endpoint.

Run:
    uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-serving")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL_NAME = "churn-classifier"
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "monitoring/predictions.csv")

model_store = {}  # holds the loaded model + metadata at runtime


def load_champion_model():
    """Load the model tagged 'champion', falling back to 'challenger'."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    for alias in ("champion", "challenger"):
        try:
            print(f"\nTrying alias: {alias}")

            version = client.get_model_version_by_alias(
                REGISTERED_MODEL_NAME,
                alias,
            )

            print(version)

            model_uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"

            print(model_uri)

            model = mlflow.sklearn.load_model(model_uri)

            print("MODEL LOADED")

            return model, version.version, alias

        except Exception as e:
            print(f"\nAlias {alias} failed")
            import traceback
            traceback.print_exc()

    raise RuntimeError(
        f"No model found for '{REGISTERED_MODEL_NAME}' with alias 'champion' or 'challenger'. "
        "Run src/train.py first."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, version, alias = load_champion_model()
    model_store["model"] = model
    model_store["version"] = version
    model_store["alias"] = alias
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    yield
    model_store.clear()


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


class ChurnFeatures(BaseModel):
    gender: str
    SeniorCitizen: str = Field(..., description="'0' or '1' as string")
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1020.0,
            }
        }
    )


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    model_version: str
    model_alias: str


def log_prediction(features: dict, prediction: int, probability: float):
    """Append the request + prediction to a CSV for later drift analysis."""
    row = {
        **features,
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    df_row = pd.DataFrame([row])
    file_exists = os.path.isfile(PREDICTION_LOG_PATH)
    df_row.to_csv(PREDICTION_LOG_PATH, mode="a", header=not file_exists, index=False)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": model_store.get("version"),
        "model_alias": model_store.get("alias"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ChurnFeatures):
    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([features.model_dump()])

    try:
        proba = model_store["model"].predict_proba(input_df)[0, 1]
        prediction = int(proba >= 0.5)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    log_prediction(features.model_dump(), prediction, float(proba))

    return PredictionResponse(
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(float(proba), 4),
        model_version=str(model_store["version"]),
        model_alias=model_store["alias"],
    )
