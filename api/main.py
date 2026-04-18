# ============================================================
# Credit Card Propensity – FastAPI Model Serving
# Exposes /predict, /health, /metrics (Prometheus)
# ============================================================

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ============================================================
# CONFIG
# ============================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit_card_propensity")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

FEATURE_NAMES = [
    "age", "gender", "tenure_to_bank", "avg_casa_this_m",
    "avg_bal_amt_6mtd_fcy_casa", "avg_bal_amt_ytd_fcy_casa",
    "cr_amt_mtd_fcy_casa", "dr_amt_mtd_fcy_casa",
    "cr_amt_qtd_fcy_casa", "dr_amt_qtd_fcy_casa",
    "avg_loan_lmt", "max_loan_lmt", "max_loan_dsbr_amt",
    "avg_loan_duration", "avg_td_last_2m", "sum_td_this_m",
    "debit_credit_ratio", "cash_pressure", "pressure_score",
    "spend_velocity", "txn_velocity", "momentum_score",
    "casa_trend", "balance_change_ratio", "utilization_ratio",
    "loan_gap", "loan_pressure_ratio", "tenure_x_util",
    "age_x_spend", "eligible_by_age", "high_spend_flag",
    "low_balance_flag", "active_txn_flag",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# PROMETHEUS METRICS
# ============================================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions made",
    ["model_name", "model_version"],
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Model prediction latency",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)
PREDICTION_SCORE = Histogram(
    "model_propensity_score",
    "Distribution of propensity scores (credit-card open probability)",
    ["model_name"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total prediction errors",
    ["model_name", "error_type"],
)
MODEL_VERSION_INFO = Gauge(
    "model_version_info",
    "Current model version",
    ["model_name", "version"],
)
MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time taken to load the model",
    ["model_name"],
)
FEATURE_VALUE = Histogram(
    "model_feature_value",
    "Distribution of feature values (drift detection)",
    ["feature_name"],
    buckets=np.linspace(-3, 3, 20).tolist(),
)

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ..., description="33 feature values in the canonical order"
    )
    feature_names: Optional[List[str]] = Field(
        None, description="Optional feature names (must match length of features)"
    )
    customer_id: Optional[str] = Field(None, description="Optional customer ID for logging")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [46, 1, 8.32, 3310613.69, 3367367.83, 2348056.57,
                              3053211.37, 3847988.53, 7404976.06, 13258901.81,
                              78920973.24, 119569150.41, 48365930.36, 70.93,
                              2950061.74, 1766109.95, 1.26, 0.61, 5853925.75,
                              0.29, 0.41, 0.35, 0.98, 0.70, 1.64,
                              40648177.18, 0.61, 13.64, 13.35, 1, 1, 0, 1],
                "feature_names": ["age", "gender", "tenure_to_bank", "..."],
            }
        }


class PredictionResponse(BaseModel):
    propensity_score: float
    will_open_card: bool
    threshold: float
    model_name: str
    model_version: str
    customer_id: Optional[str]
    timestamp: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float


# ============================================================
# MODEL MANAGER
# ============================================================


class ModelManager:
    def __init__(self):
        self.model = None
        self.model_name = MODEL_NAME
        self.model_version = "unknown"
        self.model_uri = None
        self.load_time = None

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("MLFlow tracking URI: %s", MLFLOW_TRACKING_URI)

    def load_model(self) -> bool:
        try:
            t0 = time.time()
            self.model_uri = f"models:/{self.model_name}/{MODEL_STAGE}"
            logger.info("Loading model: %s", self.model_uri)

            self.model = mlflow.pyfunc.load_model(self.model_uri)

            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(
                self.model_name, stages=[MODEL_STAGE]
            )
            self.model_version = versions[0].version if versions else "unknown"
            self.load_time = time.time() - t0

            MODEL_VERSION_INFO.labels(
                model_name=self.model_name, version=self.model_version
            ).set(int(self.model_version) if str(self.model_version).isdigit() else 0)
            MODEL_LOAD_TIME.labels(model_name=self.model_name).set(self.load_time)

            logger.info(
                "Model loaded | version=%s | time=%.2fs",
                self.model_version,
                self.load_time,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            PREDICTION_ERRORS.labels(
                model_name=self.model_name, error_type="model_load_error"
            ).inc()
            return False

    def predict(self, features: List[float]):
        if self.model is None:
            raise ValueError("Model not loaded")

        import pandas as pd
        df = pd.DataFrame([features], columns=FEATURE_NAMES)

        t0 = time.time()
        raw = self.model.predict(df)
        latency = time.time() - t0

        score = float(raw[0]) if np.ndim(raw) == 1 else float(raw[0][1])

        PREDICTION_COUNT.labels(
            model_name=self.model_name, model_version=self.model_version
        ).inc()
        PREDICTION_LATENCY.labels(model_name=self.model_name).observe(latency)
        PREDICTION_SCORE.labels(model_name=self.model_name).observe(score)

        return score, latency


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Card Propensity API",
    description="Serving LightGBM propensity model for credit-card open prediction",
    version="1.0.0",
)

model_manager = ModelManager()
app_start_time = time.time()

# ============================================================
# MIDDLEWARE
# ============================================================


@app.middleware("http")
async def track_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    latency = time.time() - t0

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method, endpoint=request.url.path
    ).observe(latency)

    logger.info(
        "%s %s status=%s latency=%.3fs",
        request.method, request.url.path, response.status_code, latency,
    )
    return response


# ============================================================
# LIFECYCLE
# ============================================================


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Credit Card Propensity API...")
    success = model_manager.load_model()
    if not success:
        logger.error("Model not loaded on startup – /predict will return 503")
    else:
        logger.info("API ready!")


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/")
async def root():
    return {
        "service": "Credit Card Propensity API",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE,
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "metrics": "GET /metrics",
            "model_info": "GET /model/info",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.model is not None else "unhealthy",
        model_loaded=model_manager.model is not None,
        model_name=model_manager.model_name,
        model_version=model_manager.model_version,
        uptime_seconds=time.time() - app_start_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model_manager.model is None:
        PREDICTION_ERRORS.labels(
            model_name=model_manager.model_name, error_type="model_not_loaded"
        ).inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.features) != len(FEATURE_NAMES):
        PREDICTION_ERRORS.labels(
            model_name=model_manager.model_name, error_type="bad_input"
        ).inc()
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURE_NAMES)} features, got {len(request.features)}",
        )

    names = request.feature_names or FEATURE_NAMES
    for fname, fval in zip(names, request.features):
        try:
            FEATURE_VALUE.labels(feature_name=fname).observe(fval)
        except Exception:
            pass

    try:
        score, pred_latency = model_manager.predict(request.features)
    except Exception as exc:
        PREDICTION_ERRORS.labels(
            model_name=model_manager.model_name, error_type="inference_error"
        ).inc()
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed")

    threshold = 0.5
    return PredictionResponse(
        propensity_score=round(score, 6),
        will_open_card=score >= threshold,
        threshold=threshold,
        model_name=model_manager.model_name,
        model_version=model_manager.model_version,
        customer_id=request.customer_id,
        timestamp=datetime.utcnow().isoformat(),
        latency_ms=round(pred_latency * 1000, 3),
    )


@app.get("/model/info")
async def model_info():
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name": model_manager.model_name,
        "model_version": model_manager.model_version,
        "model_stage": MODEL_STAGE,
        "model_uri": model_manager.model_uri,
        "load_time_seconds": model_manager.load_time,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/model/reload")
async def reload_model():
    logger.info("Reloading model...")
    success = model_manager.load_model()
    if not success:
        raise HTTPException(status_code=500, detail="Model reload failed")
    return {
        "status": "success",
        "model_version": model_manager.model_version,
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
