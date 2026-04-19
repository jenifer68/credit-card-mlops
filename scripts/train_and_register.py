# ============================================================
# Credit Card MLOps – Train & Register LightGBM Champion
#
# Run AFTER infrastructure is up:
#   python scripts/train_and_register.py
#
# What it does:
#   1. Loads feature-engineered train/valid/test CSVs
#   2. Trains LightGBM with early stopping
#   3. Logs params, metrics (AUC, KS, Lift@10%) to MLflow
#   4. Registers model as 'credit_card_propensity'
#   5. Promotes to Production stage
#   6. Saves reference data for Evidently drift monitoring
# ============================================================

import os
import sys
import time
import json
import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credit_card_propensity")
EXPERIMENT_NAME = "credit_card_propensity_experiment"

# Paths to feature-engineered data
DATA_DIR = os.getenv(
    "DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Dataset"),
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "target"

# MinIO / S3 for artifact storage
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")
os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")

# ============================================================
# HELPER METRICS
# ============================================================


def ks_stat(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False)
    df["cp"] = df["y"].cumsum() / df["y"].sum()
    df["cn"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return float(np.max(np.abs(df["cp"] - df["cn"])))


def lift_at_k(y_true, y_score, k=0.10):
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False)
    top = int(len(df) * k)
    return float(df.iloc[:top]["y"].mean() / df["y"].mean())


# ============================================================
# LOAD DATA
# ============================================================

logger.info("Loading data from: %s", DATA_DIR)

train = pd.read_csv(os.path.join(DATA_DIR, "card_train_fe.csv"))
valid = pd.read_csv(os.path.join(DATA_DIR, "card_valid_fe.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "card_test_fe.csv"))

X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
X_valid, y_valid = valid.drop(columns=[TARGET]), valid[TARGET]
X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]

logger.info("Train : %s  |  target_rate=%.4f", train.shape, y_train.mean())
logger.info("Valid : %s  |  target_rate=%.4f", valid.shape, y_valid.mean())
logger.info("Test  : %s  |  target_rate=%.4f", test.shape, y_test.mean())

FEATURE_NAMES = list(X_train.columns)
logger.info("Features: %d  -> %s", len(FEATURE_NAMES), FEATURE_NAMES[:5])

# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info("MLFlow tracking URI : %s", MLFLOW_TRACKING_URI)
logger.info("Experiment          : %s", EXPERIMENT_NAME)

# ============================================================
# TRAIN
# ============================================================

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 200,
    "verbosity": -1,
    "seed": 42,
}

lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

logger.info("Training LightGBM champion model...")

with mlflow.start_run(run_name="LightGBM_champion") as run:
    run_id = run.info.run_id
    logger.info("MLflow run_id: %s", run_id)

    mlflow.log_params(params)
    mlflow.log_param("num_boost_round", 1000)
    mlflow.log_param("early_stopping_rounds", 100)
    mlflow.log_param("train_rows", len(X_train))
    mlflow.log_param("valid_rows", len(X_valid))
    mlflow.log_param("feature_count", len(FEATURE_NAMES))

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_valid],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    pred_test = model.predict(X_test, num_iteration=model.best_iteration)

    # Validation metrics
    auc_valid = roc_auc_score(y_valid, pred_valid)
    ks_valid = ks_stat(y_valid, pred_valid)
    lift_valid = lift_at_k(y_valid, pred_valid)

    # Test metrics
    auc_test = roc_auc_score(y_test, pred_test)
    ks_test = ks_stat(y_test, pred_test)
    lift_test = lift_at_k(y_test, pred_test)

    mlflow.log_metrics({
        "valid_auc": round(auc_valid, 6),
        "valid_ks": round(ks_valid, 6),
        "valid_lift10": round(lift_valid, 6),
        "test_auc": round(auc_test, 6),
        "test_ks": round(ks_test, 6),
        "test_lift10": round(lift_test, 6),
        "best_iteration": model.best_iteration,
    })

    logger.info("Valid  AUC=%.4f  KS=%.4f  Lift@10%%=%.2f", auc_valid, ks_valid, lift_valid)
    logger.info("Test   AUC=%.4f  KS=%.4f  Lift@10%%=%.2f", auc_test, ks_test, lift_test)

    # Create a sample input for signature inference
    sample_input = X_train.head(5)
    sample_output = pd.DataFrame(
        model.predict(sample_input), columns=["propensity_score"]
    )
    signature = mlflow.models.infer_signature(sample_input, sample_output)

    mlflow.lightgbm.log_model(
        lgb_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
        signature=signature,
        input_example=sample_input,
    )

    # Log feature importance
    fi = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path)

    logger.info("Model logged to MLflow!")

# ============================================================
# PROMOTE TO PRODUCTION
# ============================================================

logger.info("Promoting model to Production stage...")

client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions(MODEL_NAME, stages=["None"])

if versions:
    latest = versions[0].version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info("Model promoted to Production | name=%s | version=%s", MODEL_NAME, latest)
else:
    logger.error("No model versions found in registry")
    sys.exit(1)

# ============================================================
# SAVE REFERENCE DATA FOR EVIDENTLY
# ============================================================

logger.info("Saving reference data for Evidently drift monitoring...")

ref_dir = os.path.join(os.path.dirname(__file__), "..", "data", "reference")
os.makedirs(ref_dir, exist_ok=True)

ref_df = X_valid.copy()
ref_df["propensity_score"] = pred_valid
ref_df.to_csv(os.path.join(ref_dir, "reference_data.csv"), index=False)

meta = {
    "description": "Validation set predictions from LightGBM champion",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "samples": len(ref_df),
    "features": FEATURE_NAMES,
    "model_name": MODEL_NAME,
    "valid_auc": round(auc_valid, 6),
    "valid_ks": round(ks_valid, 6),
}
with open(os.path.join(ref_dir, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

logger.info("Reference data saved: %d rows", len(ref_df))

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Model     : {MODEL_NAME}")
print(f"  Version   : {latest}")
print("  Stage     : Production")
print(f"  Valid AUC : {auc_valid:.4f}")
print(f"  Valid KS  : {ks_valid:.4f}")
print(f"  Test AUC  : {auc_test:.4f}")
print(f"  Test KS   : {ks_test:.4f}")
print()
print("  Next steps:")
print("    docker-compose up -d api")
print("    curl http://localhost:8000/health")
print("    python scripts/simulate_predictions.py")
print("=" * 60)
