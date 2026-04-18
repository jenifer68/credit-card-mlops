# Credit Card MLOps – System Architecture

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      CREDIT CARD MLOPS STACK                            │
│                                                                          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐   │
│  │   Grafana    │─────▶│  Prometheus  │◀─────│   FastAPI (API)      │   │
│  │   :3000      │      │   :9090      │      │   :8000              │   │
│  └──────────────┘      └──────────────┘      └──────────┬───────────┘   │
│         ▲                      ▲                         │               │
│         │                      │                         │ /predict      │
│         │             ┌────────┴──────────┐              │               │
│         │             │  Evidently        │◀─────────────┘               │
│         │             │  :8001            │  capture predictions          │
│         │             │  /capture /analyze│  for drift analysis           │
│         │             └───────────────────┘                               │
│         │                                                                  │
│         │             ┌──────────────────┐                                │
│         └────────────▶│  MLflow          │  model registry &              │
│                        │  :5000           │  experiment tracking           │
│                        └──────┬──────────┘                                │
│                               │                                           │
│                  ┌────────────┴──────────┐                                │
│                  │                       │                                │
│           ┌──────▼──────┐        ┌───────▼────┐                          │
│           │  PostgreSQL │        │   MinIO    │  (S3-compatible)          │
│           │  :5432      │        │   :9000    │  artifact storage         │
│           └─────────────┘        └────────────┘                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Services

| Service      | Port  | Purpose                                      |
|-------------|-------|----------------------------------------------|
| FastAPI      | 8000  | Model serving – POST /predict                |
| Evidently    | 8001  | Drift & data quality monitoring              |
| MLflow       | 5000  | Experiment tracking + Model Registry         |
| Prometheus   | 9090  | Metrics collection                           |
| Grafana      | 3000  | Dashboards & visualization                   |
| MinIO        | 9000  | Artifact storage (S3-compatible)             |
| MinIO Console| 9001  | MinIO web UI                                 |
| PostgreSQL   | 5432  | MLflow backend database                      |

## ML Pipeline Stages

```
Raw Data (card_train.csv)
         │
         ▼
Feature Engineering (card_train_fe.csv)
    - Ratio features (debit_credit_ratio, utilization_ratio, …)
    - Velocity features (spend_velocity, txn_velocity, momentum_score)
    - Interaction terms (tenure_x_util, age_x_spend)
    - Flag features (eligible_by_age, high_spend_flag, …)
         │
         ▼
Challenger Comparison (Step 3)
    - Logistic Regression (baseline)
    - XGBoost (challenger)
         │
         ▼
SHAP Feature Selection (Step 4)
    - Top 33 features selected by mean |SHAP|
         │
         ▼
Champion Training – LightGBM (Step 5)
    - Early stopping on validation AUC
    - Metrics: AUC, KS statistic, Lift@10%
         │
         ├──▶ MLflow Model Registry (Production)
         │
         ▼
Prediction & Customer Profiling (Step 6)
    - Top 10% propensity customers
    - Gender / Age / CASA / Loan profiles
```

## Monitoring Flow

```
1. API serves predictions → increments Prometheus counters
2. Simulator sends traffic → captures to Evidently /capture
3. Evidently /analyze compares production vs reference data
4. Drift metrics → Prometheus → Grafana dashboards
5. Alerts fire if drift_score > 0.1 or drifted_features > 5
```

## Data Flow

```
card_train_fe.csv ──▶ train_and_register.py ──▶ MLflow Registry
card_valid_fe.csv ──▶ reference_data.csv   ──▶ Evidently /reference
card_test_fe.csv  ──▶ evaluation metrics
```

## Key Metrics (Banking Standard)

| Metric     | Description                                          | Target  |
|-----------|------------------------------------------------------|---------|
| AUC       | Area Under ROC Curve                                 | > 0.75  |
| KS        | Kolmogorov–Smirnov separation statistic              | > 0.35  |
| Lift@10%  | Lift in top 10% score bucket vs overall rate         | > 2.0x  |
