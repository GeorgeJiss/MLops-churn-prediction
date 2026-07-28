# ---- Stage 1: builder ----
FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: runtime ----
FROM python:3.12-slim
WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /root/.local /home/appuser/.local

COPY src/ ./src/
COPY data/ ./data/

ENV PATH=/home/appuser/.local/bin:$PATH
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV PREDICTION_LOG_PATH=/app/monitoring/predictions.csv

RUN mkdir -p /app/monitoring

# Train inside the image -> mlflow.db + mlruns paths match the container FS
RUN python src/train.py --path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
