# ============================================================
# Data Quality Tests for Credit Card Dataset
# Run: pytest tests/test_data_quality.py -v
# ============================================================

import os
import pytest
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "Dataset"
)

REQUIRED_FEATURE_COLS = [
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

BINARY_COLS = ["gender", "eligible_by_age", "high_spend_flag", "low_balance_flag", "active_txn_flag"]
POSITIVE_COLS = ["age", "tenure_to_bank", "avg_loan_lmt", "max_loan_lmt"]


@pytest.fixture(scope="module")
def train_fe():
    path = os.path.join(DATA_DIR, "card_train_fe.csv")
    if not os.path.exists(path):
        pytest.skip(f"Dataset not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def valid_fe():
    path = os.path.join(DATA_DIR, "card_valid_fe.csv")
    if not os.path.exists(path):
        pytest.skip(f"Dataset not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def sample_fe():
    path = os.path.join(
        os.path.dirname(__file__), "..", "src", "card_train_fe - sample.csv"
    )
    if not os.path.exists(path):
        pytest.skip(f"Sample file not found: {path}")
    return pd.read_csv(path)


class TestSchemaValidation:
    def test_required_columns_present(self, sample_fe):
        missing = [c for c in REQUIRED_FEATURE_COLS if c not in sample_fe.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_target_column_present(self, sample_fe):
        assert "target" in sample_fe.columns

    def test_feature_count(self, sample_fe):
        feature_cols = [c for c in sample_fe.columns if c != "target"]
        assert len(feature_cols) == 33, f"Expected 33 features, got {len(feature_cols)}"


class TestMissingValues:
    def test_no_missing_in_sample(self, sample_fe):
        nulls = sample_fe.isnull().sum()
        cols_with_nulls = nulls[nulls > 0].index.tolist()
        assert cols_with_nulls == [], f"Columns with nulls: {cols_with_nulls}"

    def test_missing_rate_below_threshold(self, sample_fe):
        threshold = 0.10
        missing_rate = sample_fe.isnull().mean()
        high_missing = missing_rate[missing_rate > threshold].index.tolist()
        assert high_missing == [], f"High missing rate (>{threshold*100}%): {high_missing}"


class TestValueRanges:
    def test_binary_columns(self, sample_fe):
        for col in BINARY_COLS:
            if col in sample_fe.columns:
                unique_vals = set(sample_fe[col].dropna().unique())
                assert unique_vals.issubset({0, 1}), f"{col} has non-binary values: {unique_vals}"

    def test_age_range(self, sample_fe):
        assert sample_fe["age"].min() >= 18, "Age below 18 detected"
        assert sample_fe["age"].max() <= 100, "Age above 100 detected"

    def test_positive_cols(self, sample_fe):
        for col in POSITIVE_COLS:
            if col in sample_fe.columns:
                assert sample_fe[col].min() >= 0, f"{col} has negative values"

    def test_target_binary(self, sample_fe):
        unique_targets = set(sample_fe["target"].unique())
        assert unique_targets.issubset({0, 1}), f"target has non-binary values: {unique_targets}"


class TestClassBalance:
    def test_target_rate_reasonable(self, sample_fe):
        rate = sample_fe["target"].mean()
        assert 0.01 <= rate <= 0.99, f"Target rate extreme: {rate:.4f}"


class TestDataTypes:
    def test_numeric_features(self, sample_fe):
        for col in REQUIRED_FEATURE_COLS:
            if col in sample_fe.columns:
                assert pd.api.types.is_numeric_dtype(sample_fe[col]), \
                    f"Column {col} is not numeric: {sample_fe[col].dtype}"

    def test_no_object_columns_in_features(self, sample_fe):
        obj_cols = [
            c for c in REQUIRED_FEATURE_COLS
            if c in sample_fe.columns and sample_fe[c].dtype == object
        ]
        assert obj_cols == [], f"Object-type feature columns: {obj_cols}"


class TestDuplicates:
    def test_no_duplicate_rows(self, sample_fe):
        n_dupes = sample_fe.duplicated().sum()
        assert n_dupes == 0, f"Found {n_dupes} duplicate rows"


class TestTrainValidConsistency:
    def test_same_columns(self, train_fe, valid_fe):
        assert set(train_fe.columns) == set(valid_fe.columns), \
            "Train and valid have different columns"

    def test_target_rate_similar(self, train_fe, valid_fe):
        rate_diff = abs(train_fe["target"].mean() - valid_fe["target"].mean())
        assert rate_diff < 0.05, f"Target rate differs by {rate_diff:.4f} between train/valid"
