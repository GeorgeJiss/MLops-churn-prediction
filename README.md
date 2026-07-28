# MLOps Churn Prediction Pipeline

An end-to-end MLOps project built to understand how Machine Learning models are developed, versioned, deployed, monitored, and maintained in production environments. The project focuses on replicating an industry-style workflow by integrating data versioning, experiment tracking, model serving, CI/CD, Docker, and monitoring into a single pipeline.

---

## Objective

The primary goal of this project is to gain hands-on experience with the complete Machine Learning lifecycle beyond model training. It explores how production ML systems handle data, model management, deployment, monitoring, and automated workflows using modern MLOps tools.

---

## Live Dashboard

**Streamlit Dashboard**

https://mlops-churn-pred.streamlit.app

The deployed Streamlit application currently serves as the monitoring dashboard, displaying model performance, prediction statistics, and drift-related information. Model inference is handled separately through the FastAPI service.

---

## Project Architecture

```
                Data Source
                     │
                     ▼
             Data Versioning (DVC)
                     │
                     ▼
             Data Validation
                     │
                     ▼
              Model Training
                     │
                     ▼
        Experiment Tracking (MLflow)
                     │
                     ▼
          Model Registry & Versioning
                     │
                     ▼
             Model Validation
                     │
                     ▼
            Dockerized FastAPI API
                     │
                     ▼
         Prediction Logging
                     │
                     ▼
         Monitoring & Drift Detection
                     │
                     ▼
          Streamlit Dashboard
```

---

## Project Structure

```
mlops-churn-pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── train.py
│   ├── retrain.py
│   ├── serve.py
│   ├── validate_data.py
│   └── validate_model.py
│
├── monitoring/
│   ├── dashboard.py
│   ├── drift_check.py
│   ├── predictions.csv
│   └── drift_history.csv
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Model training and preprocessing pipelines |
| XGBoost | Gradient boosting classifier |
| MLflow | Experiment tracking, model registry, and versioning |
| DVC | Dataset version control and remote data storage |
| FastAPI | REST API for model inference |
| Streamlit | Monitoring dashboard |
| Docker | Containerization of the inference service |
| GitHub Actions | Continuous Integration and Continuous Deployment |
| Pytest | Unit testing |
| Pandera | Data schema validation |
| Uvicorn | ASGI server for FastAPI |

---

# Features

- Data versioning using DVC
- Data validation before training
- Multiple model training (Logistic Regression, Random Forest, XGBoost)
- Automatic best model selection
- MLflow experiment tracking
- Model Registry with versioning and aliases
- FastAPI prediction service
- Dockerized deployment
- Prediction logging
- Data drift detection
- Monitoring dashboard
- Automated CI/CD pipeline
- Automated model validation before deployment

---

# Machine Learning Pipeline

```
Raw Data
    │
    ▼
Validation
    │
    ▼
Preprocessing
    │
    ▼
Train Multiple Models
    │
    ▼
Evaluate
    │
    ▼
Select Best Model
    │
    ▼
Register in MLflow
    │
    ▼
Serve using FastAPI
    │
    ▼
Log Predictions
    │
    ▼
Monitor Drift
    │
    ▼
Retrain (Future Extension)
```

---

# CI/CD Workflow

On every push to the `main` branch:

1. Install dependencies
2. Validate dataset
3. Run unit tests
4. Train the model
5. Validate model performance
6. Build Docker image
7. Push Docker image to Docker Hub
8. Trigger deployment (configurable)

---

# API Endpoints

### Health Check

```
GET /health
```

Returns the API status and currently loaded model version.

---

### Prediction

```
POST /predict
```

Returns:

- Churn Prediction
- Prediction Probability
- Model Version
- Model Alias

---

# Model Lifecycle

```
Training
     │
     ▼
MLflow Experiment
     │
     ▼
Model Registry
     │
     ▼
Alias Assignment
(challenger/champion)
     │
     ▼
FastAPI Loading
     │
     ▼
Production Predictions
```

---

# Monitoring

The monitoring module tracks:

- Prediction history
- Prediction confidence
- Feature drift
- Drift history
- Model statistics

The Streamlit dashboard visualizes these metrics for easier monitoring and analysis.

---

# Dataset

**IBM Telco Customer Churn Dataset**

Features include:

- Customer demographics
- Services subscribed
- Contract information
- Billing information
- Monthly charges
- Customer tenure
- Churn status

---

# Running the Project

Clone the repository

```bash
git clone https://github.com/GeorgeJiss/MLops-churn-prediction.git
cd MLops-churn-prediction
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python src/train.py
```

Run the API

```bash
uvicorn src.serve:app --reload
```

Launch the monitoring dashboard

```bash
streamlit run monitoring/dashboard.py
```

---

# Future Improvements

- Automated retraining based on drift threshold
- Scheduled pipeline execution
- Kubernetes deployment
- MLflow Tracking Server
- Prometheus and Grafana integration
- Feature Store integration
- Cloud deployment (AWS/GCP/Azure)

---

## Author

George Jiss

B.Tech Artificial Intelligence and Machine Learning

This project was developed as a personal learning initiative to understand the complete MLOps lifecycle and gain practical experience with production-oriented Machine Learning workflows.