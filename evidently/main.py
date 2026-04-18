# ============================================================
# Credit Card MLOps – Evidently Drift Detection Service
# Endpoints: /capture, /capture/batch, /analyze, /reference,
#            /reports, /health, /metrics (Prometheus)
# ============================================================

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.responses import Response

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("/app/reports")
DATA_DIR = Path("/app/data")
REFERENCE_DIR = Path("/app/reference")

for d in [REPORTS_DIR, DATA_DIR, REFERENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = float(os.getenv("EVIDENTLY_DRIFT_THRESHOLD", "0.1"))
MIN_SAMPLES = int(os.getenv("EVIDENTLY_MIN_SAMPLES", "100"))

# ============================================================
# PROMETHEUS METRICS
# ============================================================

DRIFT_DETECTED = Gauge(
    "evidently_data_drift_detected",
    "Whether data drift is detected (1=yes, 0=no)",
)
DRIFT_SCORE = Gauge("evidently_drift_score", "Overall drift score (share of drifted columns)")
DRIFTED_FEATURES_COUNT = Gauge(
    "evidently_drifted_features_count", "Number of features with detected drift"
)
FEATURE_DRIFT = Gauge(
    "evidently_feature_drift", "Drift detected per feature", ["feature_name"]
)
MISSING_VALUES = Gauge(
    "evidently_missing_values_ratio", "Ratio of missing values", ["feature_name"]
)
ANALYSIS_COUNT = Counter(
    "evidently_analysis_total", "Total drift analyses performed"
)
ANALYSIS_DURATION = Histogram(
    "evidently_analysis_duration_seconds",
    "Duration of drift analysis",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
CAPTURE_COUNT = Counter(
    "evidently_captures_total", "Total prediction samples captured"
)

# ============================================================
# SCHEMAS
# ============================================================


class PredictionData(BaseModel):
    features: Dict[str, float]
    prediction: Optional[float] = None
    propensity_score: Optional[float] = None
    timestamp: Optional[str] = None
    model_version: Optional[str] = None


class BatchPredictionData(BaseModel):
    data: List[Dict[str, Any]]


class DriftAnalysisRequest(BaseModel):
    window_size: Optional[int] = Field(500, description="Recent samples to analyse")
    threshold: Optional[float] = Field(None, description="Override default drift threshold")


class ReferenceDataRequest(BaseModel):
    data: List[Dict[str, Any]]
    feature_names: Optional[List[str]] = None
    description: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    reference_data_loaded: bool
    production_data_count: int
    last_analysis: Optional[str]
    reports_count: int


# ============================================================
# DATA STORE
# ============================================================


class DataStore:
    def __init__(self):
        self.reference_data: Optional[pd.DataFrame] = None
        self.production_data: List[Dict] = []
        self.last_analysis_time: Optional[datetime] = None
        self.reference_metadata: Dict = {}
        self._load_reference_data()

    def _load_reference_data(self):
        ref_file = REFERENCE_DIR / "reference_data.csv"
        meta_file = REFERENCE_DIR / "metadata.json"
        if ref_file.exists():
            try:
                self.reference_data = pd.read_csv(ref_file)
                logger.info("Loaded reference data: %d samples", len(self.reference_data))
                if meta_file.exists():
                    with open(meta_file) as f:
                        self.reference_metadata = json.load(f)
            except Exception as exc:
                logger.error("Failed to load reference data: %s", exc)

    def save_reference_data(self, df: pd.DataFrame, metadata: Dict = None):
        ref_file = REFERENCE_DIR / "reference_data.csv"
        df.to_csv(ref_file, index=False)
        if metadata:
            with open(REFERENCE_DIR / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        self.reference_data = df
        self.reference_metadata = metadata or {}
        logger.info("Saved reference data: %d samples", len(df))

    def add_production_data(self, record: Dict):
        if not record.get("timestamp"):
            record["timestamp"] = datetime.utcnow().isoformat()
        self.production_data.append(record)
        if len(self.production_data) > 20000:
            self.production_data = self.production_data[-20000:]
        CAPTURE_COUNT.inc()

    def get_production_df(self, window: Optional[int] = None) -> pd.DataFrame:
        if not self.production_data:
            return pd.DataFrame()
        data = self.production_data[-window:] if window else self.production_data
        return pd.DataFrame(data)

    def clear_production_data(self):
        self.production_data = []
        logger.info("Production data cleared")


data_store = DataStore()

# ============================================================
# DRIFT ANALYSIS
# ============================================================


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold: float = DRIFT_THRESHOLD,
) -> Dict[str, Any]:
    EXCLUDE = {"prediction", "propensity_score", "timestamp", "model_version"}
    common_cols = [
        c for c in reference_df.columns
        if c in current_df.columns and c not in EXCLUDE
    ]

    numeric_cols = [
        c for c in common_cols
        if pd.api.types.is_numeric_dtype(reference_df[c])
    ]

    if not numeric_cols:
        raise ValueError("No common numeric feature columns found")

    ref = reference_df[numeric_cols].copy()
    cur = current_df[numeric_cols].copy()

    logger.info("Drift analysis | ref=%d | cur=%d | features=%d", len(ref), len(cur), len(numeric_cols))

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)
    result_dict = report.as_dict()

    drift_detected = False
    drift_score = 0.0
    drifted_features: List[str] = []
    drift_scores: Dict[str, float] = {}

    for metric in result_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            res = metric.get("result", {})
            drift_detected = res.get("dataset_drift", False)
            drift_score = res.get("share_of_drifted_columns", 0.0)
            for feat, info in res.get("drift_by_columns", {}).items():
                is_drifted = info.get("drift_detected", False)
                score = info.get("drift_score", 0.0)
                drift_scores[feat] = score
                if is_drifted:
                    drifted_features.append(feat)
                FEATURE_DRIFT.labels(feature_name=feat).set(1 if is_drifted else 0)

        if metric.get("metric") == "DataQualityPreset":
            res = metric.get("result", {})
            for feat, stats in res.get("current", {}).get("columns", {}).items():
                missing = stats.get("missing_percentage", 0.0) or 0.0
                MISSING_VALUES.labels(feature_name=feat).set(missing / 100.0)

    DRIFT_DETECTED.set(1 if drift_detected else 0)
    DRIFT_SCORE.set(drift_score)
    DRIFTED_FEATURES_COUNT.set(len(drifted_features))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = f"drift_report_{ts}.html"
    report.save_html(str(REPORTS_DIR / report_file))
    logger.info("Report saved: %s", report_file)

    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "drifted_features": drifted_features,
        "drift_scores": drift_scores,
        "total_features": len(numeric_cols),
        "drifted_count": len(drifted_features),
        "reference_samples": len(ref),
        "current_samples": len(cur),
        "report_url": f"/reports/{report_file}",
    }


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Credit Card MLOps – Evidently Drift Service",
    description="Data drift & data quality monitoring for credit-card propensity model",
    version="1.0.0",
)

app_start_time = datetime.utcnow()


@app.get("/")
async def root():
    return {
        "service": "Evidently Drift Detection",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "metrics": "GET /metrics",
            "capture": "POST /capture",
            "capture_batch": "POST /capture/batch",
            "analyze": "POST /analyze",
            "reference_get": "GET /reference",
            "reference_upload": "POST /reference",
            "reports": "GET /reports",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    reports = list(REPORTS_DIR.glob("*.html"))
    return HealthResponse(
        status="healthy",
        reference_data_loaded=data_store.reference_data is not None,
        production_data_count=len(data_store.production_data),
        last_analysis=(
            data_store.last_analysis_time.isoformat()
            if data_store.last_analysis_time
            else None
        ),
        reports_count=len(reports),
    )


@app.post("/capture")
async def capture(data: PredictionData):
    record = {**data.features}
    if data.prediction is not None:
        record["prediction"] = data.prediction
    if data.propensity_score is not None:
        record["propensity_score"] = data.propensity_score
    record["timestamp"] = data.timestamp or datetime.utcnow().isoformat()
    record["model_version"] = data.model_version
    data_store.add_production_data(record)
    return {
        "status": "captured",
        "total_samples": len(data_store.production_data),
    }


@app.post("/capture/batch")
async def capture_batch(payload: BatchPredictionData):
    for item in payload.data:
        if "timestamp" not in item:
            item["timestamp"] = datetime.utcnow().isoformat()
        data_store.add_production_data(item)
    return {
        "status": "captured",
        "added": len(payload.data),
        "total_samples": len(data_store.production_data),
    }


@app.post("/analyze")
async def analyze(req: DriftAnalysisRequest = DriftAnalysisRequest()):
    if data_store.reference_data is None:
        raise HTTPException(status_code=400, detail="No reference data loaded")

    current_df = data_store.get_production_df(req.window_size)
    if len(current_df) < MIN_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_SAMPLES} samples (have {len(current_df)})",
        )

    t0 = time.time()
    try:
        result = run_drift_analysis(
            reference_df=data_store.reference_data,
            current_df=current_df,
            threshold=req.threshold or DRIFT_THRESHOLD,
        )
    except Exception as exc:
        logger.error("Drift analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    duration = time.time() - t0
    ANALYSIS_COUNT.inc()
    ANALYSIS_DURATION.observe(duration)
    data_store.last_analysis_time = datetime.utcnow()

    result["analysis_duration_seconds"] = round(duration, 3)
    return result


@app.get("/reference")
async def get_reference():
    if data_store.reference_data is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "samples": len(data_store.reference_data),
        "features": list(data_store.reference_data.columns),
        "metadata": data_store.reference_metadata,
    }


@app.post("/reference")
async def upload_reference(req: ReferenceDataRequest):
    df = pd.DataFrame(req.data)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Empty dataset")

    metadata = {
        "description": req.description or "Credit card training reference data",
        "uploaded_at": datetime.utcnow().isoformat(),
        "samples": len(df),
        "features": req.feature_names or list(df.columns),
    }
    data_store.save_reference_data(df, metadata)
    return {
        "status": "success",
        "samples": len(df),
        "features": list(df.columns),
    }


@app.get("/reports")
async def list_reports():
    reports = sorted(
        REPORTS_DIR.glob("*.html"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return {
        "count": len(reports),
        "reports": [
            {
                "filename": r.name,
                "created": datetime.fromtimestamp(r.stat().st_mtime).isoformat(),
                "size_kb": round(r.stat().st_size / 1024, 2),
                "url": f"/reports/{r.name}",
            }
            for r in reports
        ],
    }


@app.get("/reports/{report_name}", response_class=HTMLResponse)
async def get_report(report_name: str):
    path = REPORTS_DIR / report_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return path.read_text()


@app.delete("/production-data")
async def clear():
    data_store.clear_production_data()
    return {"status": "cleared"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================
# STARTUP
# ============================================================


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("Evidently Drift Service – Credit Card MLOps")
    logger.info("=" * 50)
    if data_store.reference_data is not None:
        logger.info(
            "Reference data loaded: %d samples | %d features",
            len(data_store.reference_data),
            len(data_store.reference_data.columns),
        )
    else:
        logger.warning("No reference data. Upload via POST /reference")
    logger.info("Service ready!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
