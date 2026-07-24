# ---- Stage 1: build dependencies in an isolated layer ----
FROM python:3.12-slim AS builder
 
WORKDIR /build
 
# Only serving-time dependencies go into the final image; dvc/pytest stay
# out (they're dev/CI tools), keeping the runtime image smaller.
RUN pip install --no-cache-dir --upgrade pip
 
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    pandas numpy scikit-learn xgboost mlflow fastapi "uvicorn[standard]"
 
# ---- Stage 2: minimal runtime image ----
FROM python:3.12-slim
 
WORKDIR /app
 
# Create a non-root user for the container process
RUN useradd --create-home --shell /bin/bash appuser
 
# Bring in only the installed packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local
 
COPY src/serve.py ./src/serve.py
COPY mlflow.db ./mlflow.db
COPY mlruns ./mlruns
 
RUN mkdir -p /app/monitoring && chown -R appuser:appuser /app
 
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
ENV PREDICTION_LOG_PATH=/app/monitoring/predictions.csv
 
EXPOSE 8000
 
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
 
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]