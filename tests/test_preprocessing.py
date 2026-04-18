# ============================================================
# Preprocessing / Feature Engineering Tests
# Run: pytest tests/test_preprocessing.py -v
# ============================================================

import os
import pytest
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "Dataset"
)

ENGINEERED_FEATURES = [
    "debit_credit_ratio",
    "cash_pressure",
    "pressure_score",
    "spend_velocity",
    "txn_velocity",
    "momentum_score",
    "casa_trend",
    "balance_change_ratio",
    "utilization_ratio",
    "loan_gap",
    "loan_pressure_ratio",
    "tenure_x_util",
    "age_x_spend",
    "eligible_by_age",
    "high_spend_flag",
    "low_balance_flag",
    "active_txn_flag",
]

RAW_BASE_FEATURES = [
    "age", "gender", "tenure_to_bank", "avg_casa_this_m",
    "avg_loan_lmt", "max_loan_lmt", "max_loan_dsbr_amt",
]


@pytest.fixture(scope="module")
def sample():
    path = os.path.join(
        os.path.dirname(__file__), "..", "src", "card_train_fe - sample.csv"
    )
    if not os.path.exists(path):
        pytest.skip(f"Sample file not found: {path}")
    return pd.read_csv(path)


class TestEngineeredFeatures:
    def test_all_engineered_cols_present(self, sample):
        missing = [c for c in ENGINEERED_FEATURES if c not in sample.columns]
        assert missing == [], f"Missing engineered columns: {missing}"

    def test_debit_credit_ratio_positive(self, sample):
        assert (sample["debit_credit_ratio"] >= 0).all(), \
            "debit_credit_ratio contains negative values"

    def test_utilization_ratio_positive(self, sample):
        assert (sample["utilization_ratio"] >= 0).all(), \
            "utilization_ratio contains negative values"

    def test_spend_velocity_bounded(self, sample):
        assert sample["spend_velocity"].between(0, 10).all(), \
            "spend_velocity out of expected range [0, 10]"

    def test_momentum_score_bounded(self, sample):
        assert sample["momentum_score"].between(0, 5).all(), \
            "momentum_score out of expected range [0, 5]"

    def test_flag_columns_binary(self, sample):
        flag_cols = ["eligible_by_age", "high_spend_flag", "low_balance_flag", "active_txn_flag"]
        for col in flag_cols:
            vals = set(sample[col].unique())
            assert vals.issubset({0, 1}), f"{col} is not binary: {vals}"

    def test_loan_gap_non_negative(self, sample):
        assert (sample["loan_gap"] >= 0).all(), "loan_gap contains negative values"


class TestFeatureConsistency:
    def test_max_loan_gte_avg(self, sample):
        assert (sample["max_loan_lmt"] >= sample["avg_loan_lmt"]).all(), \
            "max_loan_lmt should be >= avg_loan_lmt"

    def test_tenure_positive(self, sample):
        assert (sample["tenure_to_bank"] > 0).all(), "tenure_to_bank must be positive"

    def test_interaction_terms_nonzero(self, sample):
        for col in ["tenure_x_util", "age_x_spend"]:
            non_zero = (sample[col] != 0).any()
            assert non_zero, f"{col} is all zeros – interaction term may be broken"


class TestFeatureDistributions:
    def test_no_constant_features(self, sample):
        feature_cols = [c for c in sample.columns if c != "target"]
        constant = [c for c in feature_cols if sample[c].nunique() == 1]
        assert constant == [], f"Constant features (zero variance): {constant}"

    def test_no_extreme_skew(self, sample):
        numeric_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            skew = abs(sample[col].skew())
            assert skew < 1000, f"{col} has extremely high skew: {skew:.1f}"


class TestFeatureVsTarget:
    def test_positive_correlation_casa(self, sample):
        corr = sample["avg_casa_this_m"].corr(sample["target"])
        assert corr > -1.0, f"Unexpected perfect negative correlation for avg_casa_this_m"

    def test_eligible_by_age_with_target(self, sample):
        rate_eligible = sample.loc[sample["eligible_by_age"] == 1, "target"].mean()
        assert 0.0 <= rate_eligible <= 1.0, "Target rate for eligible customers out of [0,1]"


class TestSampleCoversPipeline:
    def test_sample_has_rows(self, sample):
        assert len(sample) > 0, "Sample CSV is empty"

    def test_all_required_base_features(self, sample):
        missing = [c for c in RAW_BASE_FEATURES if c not in sample.columns]
        assert missing == [], f"Missing base features: {missing}"
