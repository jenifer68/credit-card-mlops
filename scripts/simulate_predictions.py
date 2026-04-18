# ============================================================
# Credit Card MLOps – Prediction Simulator
#
# Sends synthetic traffic to the API and captures data to
# Evidently for drift monitoring.
#
# Usage:
#   python scripts/simulate_predictions.py [--n 200] [--drift]
# ============================================================

import argparse
import json
import os
import random
import sys
import time
import logging

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "Dataset"
)

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


# ============================================================
# LOAD REAL DATA (for realistic simulation)
# ============================================================

def load_data(drift: bool = False) -> pd.DataFrame:
    if drift:
        path = os.path.join(DATA_DIR, "card_test_fe.csv")
        logger.info("Drift mode: loading test data")
    else:
        path = os.path.join(DATA_DIR, "card_valid_fe.csv")
        logger.info("Normal mode: loading validation data")

    if not os.path.exists(path):
        logger.warning("Data file not found – using synthetic data: %s", path)
        return None

    df = pd.read_csv(path)
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    return df[FEATURE_NAMES]


def apply_drift(df: pd.DataFrame, strength: float = 2.0) -> pd.DataFrame:
    """Apply artificial feature drift for demonstration."""
    drifted = df.copy()
    numeric_cols = [c for c in FEATURE_NAMES if c not in
                    ["gender", "eligible_by_age", "high_spend_flag",
                     "low_balance_flag", "active_txn_flag"]]
    for col in numeric_cols[:10]:
        drifted[col] = drifted[col] * strength + drifted[col].std() * np.random.randn(len(drifted))
    return drifted


def make_synthetic_row() -> list:
    return [
        random.randint(22, 60),               # age
        random.randint(0, 1),                  # gender
        random.uniform(1, 20),                 # tenure_to_bank
        random.uniform(100_000, 5_000_000),    # avg_casa_this_m
        random.uniform(100_000, 4_000_000),    # avg_bal_amt_6mtd_fcy_casa
        random.uniform(100_000, 3_500_000),    # avg_bal_amt_ytd_fcy_casa
        random.uniform(500_000, 8_000_000),    # cr_amt_mtd_fcy_casa
        random.uniform(500_000, 9_000_000),    # dr_amt_mtd_fcy_casa
        random.uniform(1_000_000, 15_000_000), # cr_amt_qtd_fcy_casa
        random.uniform(1_000_000, 18_000_000), # dr_amt_qtd_fcy_casa
        random.uniform(20_000_000, 200_000_000), # avg_loan_lmt
        random.uniform(30_000_000, 300_000_000), # max_loan_lmt
        random.uniform(10_000_000, 100_000_000), # max_loan_dsbr_amt
        random.uniform(12, 120),               # avg_loan_duration
        random.uniform(100_000, 4_000_000),    # avg_td_last_2m
        random.uniform(100_000, 3_000_000),    # sum_td_this_m
        random.uniform(0.5, 3.5),              # debit_credit_ratio
        random.uniform(0.1, 1.5),              # cash_pressure
        random.uniform(100_000, 8_000_000),    # pressure_score
        random.uniform(0.1, 0.6),              # spend_velocity
        random.uniform(0.1, 0.6),              # txn_velocity
        random.uniform(0.2, 0.5),              # momentum_score
        random.uniform(0.5, 1.5),              # casa_trend
        random.uniform(0.5, 2.0),              # balance_change_ratio
        random.uniform(0.5, 2.0),              # utilization_ratio
        random.uniform(5_000_000, 150_000_000), # loan_gap
        random.uniform(0.3, 0.8),              # loan_pressure_ratio
        random.uniform(5, 30),                 # tenure_x_util
        random.uniform(5, 30),                 # age_x_spend
        random.randint(0, 1),                  # eligible_by_age
        random.randint(0, 1),                  # high_spend_flag
        random.randint(0, 1),                  # low_balance_flag
        random.randint(0, 1),                  # active_txn_flag
    ]


# ============================================================
# SEND REFERENCE DATA TO EVIDENTLY
# ============================================================

def upload_reference_to_evidently(df: pd.DataFrame):
    try:
        sample = df.sample(min(500, len(df)), random_state=42)
        payload = {
            "data": sample.to_dict(orient="records"),
            "feature_names": FEATURE_NAMES,
            "description": "Validation set from LightGBM training",
        }
        r = requests.post(f"{EVIDENTLY_URL}/reference", json=payload, timeout=30)
        r.raise_for_status()
        logger.info("Reference data uploaded to Evidently: %d rows", len(sample))
    except Exception as exc:
        logger.warning("Could not upload reference to Evidently: %s", exc)


# ============================================================
# CAPTURE TO EVIDENTLY
# ============================================================

def capture_to_evidently(features: list, score: float, version: str):
    try:
        record = dict(zip(FEATURE_NAMES, features))
        payload = {
            "features": record,
            "propensity_score": score,
            "model_version": version,
        }
        requests.post(f"{EVIDENTLY_URL}/capture", json=payload, timeout=5)
    except Exception:
        pass


# ============================================================
# SIMULATE TRAFFIC
# ============================================================

def simulate(n: int = 100, drift: bool = False, delay: float = 0.05):
    logger.info("Checking API health...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        r.raise_for_status()
        info = r.json()
        model_version = info.get("model_version", "unknown")
        logger.info("API ready | model_version=%s", model_version)
    except Exception as exc:
        logger.error("API not reachable at %s: %s", API_URL, exc)
        sys.exit(1)

    df = load_data(drift)

    if df is not None and drift:
        df = apply_drift(df, strength=2.5)

    logger.info("Sending %d predictions (drift=%s)...", n, drift)

    success = 0
    errors = 0
    scores = []

    for i in range(n):
        if df is not None:
            row = df.sample(1).iloc[0].tolist()
        else:
            row = make_synthetic_row()

        payload = {"features": row, "feature_names": FEATURE_NAMES}

        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            r.raise_for_status()
            resp = r.json()
            score = resp.get("propensity_score", 0)
            scores.append(score)
            capture_to_evidently(row, score, model_version)
            success += 1
        except Exception as exc:
            logger.warning("Request %d failed: %s", i + 1, exc)
            errors += 1

        if (i + 1) % 50 == 0:
            logger.info("  Progress: %d/%d | success=%d errors=%d", i + 1, n, success, errors)

        time.sleep(delay)

    logger.info("Done! success=%d errors=%d avg_score=%.4f",
                success, errors, np.mean(scores) if scores else 0)

    # Trigger drift analysis if we have enough data
    if success >= 100:
        try:
            r = requests.post(f"{EVIDENTLY_URL}/analyze", json={"window_size": success}, timeout=60)
            if r.status_code == 200:
                result = r.json()
                logger.info(
                    "Drift analysis | detected=%s | score=%.4f | drifted_features=%s",
                    result.get("drift_detected"),
                    result.get("drift_score", 0),
                    result.get("drifted_features", []),
                )
            else:
                logger.warning("Drift analysis returned %d: %s", r.status_code, r.text[:200])
        except Exception as exc:
            logger.warning("Could not run drift analysis: %s", exc)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Credit Card MLOps – Prediction Simulator")
    parser.add_argument("-n", "--n", type=int, default=100, help="Number of requests")
    parser.add_argument("--drift", action="store_true", help="Apply artificial feature drift")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between requests")
    parser.add_argument("--upload-reference", action="store_true",
                        help="Upload reference data to Evidently before simulation")
    args = parser.parse_args()

    if args.upload_reference:
        df = load_data(drift=False)
        if df is not None:
            upload_reference_to_evidently(df)

    simulate(n=args.n, drift=args.drift, delay=args.delay)


if __name__ == "__main__":
    main()
