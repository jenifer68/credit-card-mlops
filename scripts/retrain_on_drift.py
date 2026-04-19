# ============================================================
# Credit Card MLOps – Retrain on Drift
#
# Kiểm tra drift từ Evidently, nếu vượt ngưỡng thì retrain.
#
# Usage:
#   python scripts/retrain_on_drift.py [--force] [--dry-run]
#
# Flow:
#   1. Gọi POST /analyze trên Evidently service
#   2. Nếu drift_detected=True hoặc drift_score > threshold
#      → chạy train_and_register.py
#      → gọi POST /model/reload trên API
#   3. In kết quả và exit code
#        0 = không cần retrain / retrain thành công
#        1 = lỗi
# ============================================================

import argparse
import logging
import os
import subprocess
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8001")
API_URL = os.getenv("API_URL", "http://localhost:8000")
DRIFT_SCORE_THRESHOLD = float(os.getenv("DRIFT_SCORE_THRESHOLD", "0.1"))
MIN_SAMPLES = int(os.getenv("EVIDENTLY_MIN_SAMPLES", "100"))

SCRIPTS_DIR = os.path.dirname(__file__)
TRAIN_SCRIPT = os.path.join(SCRIPTS_DIR, "train_and_register.py")


# ============================================================
# HELPERS
# ============================================================

def check_evidently_health() -> dict:
    try:
        r = requests.get(f"{EVIDENTLY_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.error("Cannot reach Evidently at %s: %s", EVIDENTLY_URL, exc)
        sys.exit(1)


def run_drift_analysis(window_size: int = 500) -> dict:
    try:
        r = requests.post(
            f"{EVIDENTLY_URL}/analyze",
            json={"window_size": window_size},
            timeout=120,
        )
        if r.status_code == 400:
            logger.error("Drift analysis failed: %s", r.json().get("detail"))
            sys.exit(1)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.error("Drift analysis error: %s", exc)
        sys.exit(1)


def retrain() -> bool:
    logger.info("=" * 55)
    logger.info("Starting retraining pipeline...")
    logger.info("=" * 55)
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("train_and_register.py failed with exit code %d", result.returncode)
        return False
    logger.info("Retraining completed successfully.")
    return True


def reload_api_model() -> bool:
    try:
        r = requests.post(f"{API_URL}/model/reload", timeout=60)
        r.raise_for_status()
        info = r.json()
        logger.info("API reloaded model | version=%s", info.get("model_version"))
        return True
    except Exception as exc:
        logger.error("Failed to reload API model: %s", exc)
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Retrain model when data drift is detected")
    parser.add_argument("--force", action="store_true", help="Force retrain regardless of drift")
    parser.add_argument("--dry-run", action="store_true", help="Check drift only, do not retrain")
    parser.add_argument("--window", type=int, default=500, help="Production samples to analyse (default 500)")
    parser.add_argument("--threshold", type=float, default=DRIFT_SCORE_THRESHOLD,
                        help=f"Drift score threshold (default {DRIFT_SCORE_THRESHOLD})")
    args = parser.parse_args()

    # ── 1. Kiểm tra Evidently health ──────────────────────────
    logger.info("Checking Evidently health...")
    health = check_evidently_health()
    logger.info(
        "Evidently | reference_loaded=%s | production_samples=%d",
        health.get("reference_data_loaded"),
        health.get("production_data_count", 0),
    )

    if not health.get("reference_data_loaded"):
        logger.error("No reference data in Evidently. Upload via: "
                     "python scripts/simulate_predictions.py --upload-reference")
        sys.exit(1)

    prod_count = health.get("production_data_count", 0)
    if prod_count < MIN_SAMPLES:
        logger.warning(
            "Not enough production samples (%d < %d). Run simulate_predictions.py first.",
            prod_count, MIN_SAMPLES,
        )
        sys.exit(1)

    # ── 2. Chạy drift analysis ─────────────────────────────────
    logger.info("Running drift analysis (window=%d)...", args.window)
    result = run_drift_analysis(window_size=args.window)

    drift_detected = result.get("drift_detected", False)
    drift_score = result.get("drift_score", 0.0)
    drifted_features = result.get("drifted_features", [])

    logger.info("=" * 55)
    logger.info("DRIFT ANALYSIS RESULT")
    logger.info("  drift_detected : %s", drift_detected)
    logger.info("  drift_score    : %.4f  (threshold=%.4f)", drift_score, args.threshold)
    logger.info("  drifted_count  : %d / %d features", result.get("drifted_count", 0), result.get("total_features", 0))
    if drifted_features:
        logger.info("  drifted_features: %s", drifted_features)
    logger.info("  report_url     : %s%s", EVIDENTLY_URL, result.get("report_url", ""))
    logger.info("=" * 55)

    # ── 3. Quyết định retrain ──────────────────────────────────
    should_retrain = args.force or drift_detected or (drift_score >= args.threshold)

    if not should_retrain:
        logger.info("No drift detected. No retraining needed.")
        sys.exit(0)

    if args.dry_run:
        logger.info("[DRY-RUN] Drift detected but skipping retrain (--dry-run flag).")
        sys.exit(0)

    logger.info("Drift detected! Triggering retrain...")

    # ── 4. Retrain ─────────────────────────────────────────────
    if not retrain():
        sys.exit(1)

    # ── 5. Reload API ──────────────────────────────────────────
    logger.info("Reloading model in API...")
    time.sleep(3)
    reload_ok = reload_api_model()

    if not reload_ok:
        logger.warning("Retrain succeeded but API reload failed. Run manually:")
        logger.warning("  curl.exe -X POST %s/model/reload", API_URL)
        sys.exit(1)

    # ── 6. Xác nhận version mới ────────────────────────────────
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        info = r.json()
        logger.info("=" * 55)
        logger.info("RETRAIN COMPLETE")
        logger.info("  New model version : %s", info.get("model_version"))
        logger.info("  API status        : %s", info.get("status"))
        logger.info("=" * 55)
    except Exception:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
