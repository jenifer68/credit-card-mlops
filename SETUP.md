# Credit Card MLOps – Hướng Dẫn Setup & Chạy Đầy Đủ

> **Dự án:** Dự đoán mở thẻ tín dụng (Credit Card Propensity)  
> **Student:** FSB32-Minh Ha  
> **Stack:** LightGBM · MLflow · FastAPI · Evidently · Prometheus · Grafana · MinIO · PostgreSQL

---

## Mục Lục

1. [Yêu Cầu Hệ Thống](#1-yêu-cầu-hệ-thống)
2. [Cấu Trúc Dự Án](#2-cấu-trúc-dự-án)
3. [Bước 1 – Chuẩn Bị Môi Trường](#3-bước-1--chuẩn-bị-môi-trường)
4. [Bước 2 – Khởi Động Infrastructure](#4-bước-2--khởi-động-infrastructure)
5. [Bước 3 – Train & Register Model](#5-bước-3--train--register-model)
6. [Bước 4 – Khởi Động API & Monitoring](#6-bước-4--khởi-động-api--monitoring)
7. [Bước 5 – Upload Reference Data cho Evidently](#7-bước-5--upload-reference-data-cho-evidently)
8. [Bước 6 – Simulate Traffic & Drift](#8-bước-6--simulate-traffic--drift)
9. [Truy Cập Dashboards](#9-truy-cập-dashboards)
10. [Chạy Unit Tests](#10-chạy-unit-tests)
11. [API Reference](#11-api-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Reset Toàn Bộ](#13-reset-toàn-bộ)

---

## 1. Yêu Cầu Hệ Thống

| Tool | Phiên bản tối thiểu | Ghi chú |
|------|-------------------|---------|
| **Docker** | ≥ 20.10 | `docker --version` |
| **Docker Compose** | ≥ 2.0 | `docker compose version` |
| **Python** | ≥ 3.10 | Để chạy training script và tests |
| **RAM** | ≥ 8 GB | MLflow + Grafana + API + Evidently |
| **Disk** | ≥ 5 GB | Data CSV + Docker images |

---

## 2. Cấu Trúc Dự Án

```
credit-card-mlops/
├── .env                          # Biến môi trường (tạo từ .env.example)
├── .env.example                  # Template env
├── docker-compose.yml            # Toàn bộ stack (8 services)
├── requirements.txt              # Python dependencies (dev/test)
│
├── api/                          # FastAPI model serving
│   ├── Dockerfile
│   ├── main.py                   # /predict /health /metrics /model/info
│   └── requirements.txt
│
├── evidently/                    # Drift detection service
│   ├── Dockerfile
│   ├── main.py                   # /capture /analyze /reports /reference
│   └── requirements.txt
│
├── mlflow/                       # MLflow server
│   └── Dockerfile
│
├── scripts/                      # Scripts chạy ngoài Docker
│   ├── train_and_register.py     # Train LightGBM → MLflow Registry
│   ├── simulate_predictions.py   # Gửi traffic đến API + Evidently
│   └── requirements.txt
│
├── src/                          # Code ML gốc (phân tích, so sánh)
│   ├── 3_train_challengers.py    # LogReg vs XGBoost
│   ├── 4_shap_feature_selection.py
│   ├── 5_train_champion_lightgbm_full.py
│   └── 6_predict_and_profile.py
│
├── data/
│   └── raw/Dataset/              # CSV data files
│       ├── card_train_fe.csv
│       ├── card_valid_fe.csv
│       └── card_test_fe.csv
│
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml
│   │   └── dashboards/dashboards.yml
│   └── dashboards/
│       └── credit_card_mlops.json  # Pre-built Grafana dashboard
│
├── docker/
│   └── prometheus/
│       └── prometheus.yml
│
├── tests/
│   ├── test_api.py               # Unit tests cho API
│   ├── test_data_quality.py      # Kiểm tra chất lượng data
│   └── test_preprocessing.py     # Kiểm tra feature engineering
│
├── .github/workflows/
│   └── ci_cd.yml                 # GitHub Actions CI/CD
│
├── SETUP.md                      # File này
└── ARCHITECTURE.md               # Kiến trúc hệ thống
```

---

## 3. Bước 1 – Chuẩn Bị Môi Trường

### 3.1 Clone / Unzip dự án

```bash
# Nếu dùng git
cd credit-card-mlops

# Hoặc unzip
unzip credit-card-mlops.zip
cd credit-card-mlops
```

### 3.2 Tạo file .env

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Kiểm tra nội dung `.env` (mặc định đã dùng được):

```env
USER=student
MODEL_NAME=credit_card_propensity
MODEL_STAGE=Production
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

### 3.3 Cài Python dependencies (cho scripts local)

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Cài packages
pip install -r scripts/requirements.txt
```

---

## 4. Bước 2 – Khởi Động Infrastructure

### 4.1 Start database + storage + MLflow

```bash
docker-compose up -d postgres minio minio-init mlflow
```

### 4.2 Kiểm tra trạng thái

```bash
docker-compose ps
```

Chờ tất cả services trạng thái **healthy** hoặc **Up** (khoảng 30-60 giây):

```
NAME                  STATUS
student_cc-postgres   Up (healthy)
student_cc-minio      Up (healthy)
student_cc-minio-init Exited (0)      ← OK, đây là init container
student_cc-mlflow     Up (healthy)
```

### 4.3 Kiểm tra MLflow

```bash
curl http://localhost:5000/health
# Hoặc mở trình duyệt: http://localhost:5000
```

---

## 5. Bước 3 – Train & Register Model

> ⚠️ **QUAN TRỌNG:** Phải chạy bước này TRƯỚC khi start API.

### 5.1 Chạy training script

```bash
python scripts/train_and_register.py
```

**Script sẽ tự động:**
- Load `data/raw/Dataset/card_train_fe.csv` và `card_valid_fe.csv`
- Train LightGBM champion với early stopping
- Log params + metrics (AUC, KS, Lift@10%) vào MLflow
- Register model `credit_card_propensity` vào MLflow Registry
- Promote lên stage **Production**
- Lưu reference data vào `data/reference/reference_data.csv` cho Evidently

**Output mong đợi:**
```
Train : (N, 34)  |  target_rate=0.XXXX
Valid : (N, 34)  |  target_rate=0.XXXX
MLflow run_id: abc123...
[LightGBM] Training with early stopping...
Valid  AUC=0.7800  KS=0.4200  Lift@10%=2.50
Model promoted to Production | name=credit_card_propensity | version=1

==============================================================
  TRAINING COMPLETE
  Model     : credit_card_propensity
  Version   : 1
  Valid AUC : 0.7800
  Test  AUC : 0.7750
==============================================================
```

### 5.2 Kiểm tra trên MLflow UI

Mở http://localhost:5000 → **Models** → `credit_card_propensity` → Stage: **Production**

---

## 6. Bước 4 – Khởi Động API & Monitoring

### 6.1 Start toàn bộ stack còn lại

```bash
docker-compose up -d api evidently prometheus grafana
```

### 6.2 Kiểm tra API

```bash
curl http://localhost:8000/health
```

Response mong đợi:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_card_propensity",
  "model_version": "1",
  "uptime_seconds": 12.3
}
```

### 6.3 Kiểm tra Evidently

```bash
curl http://localhost:8001/health
```

### 6.4 Test prediction thủ công

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [46, 1, 8.32, 3310613.69, 3367367.83, 2348056.57,
                 3053211.37, 3847988.53, 7404976.06, 13258901.81,
                 78920973.24, 119569150.41, 48365930.36, 70.93,
                 2950061.74, 1766109.95, 1.26, 0.61, 5853925.75,
                 0.29, 0.41, 0.35, 0.98, 0.70, 1.64,
                 40648177.18, 0.61, 13.64, 13.35, 1, 1, 0, 1]
  }'
```

Response:
```json
{
  "propensity_score": 0.712345,
  "will_open_card": true,
  "threshold": 0.5,
  "model_name": "credit_card_propensity",
  "model_version": "1",
  "latency_ms": 8.5
}
```

---

## 7. Bước 5 – Upload Reference Data cho Evidently

Reference data (validation set) cần được upload vào Evidently để làm baseline cho drift detection.

### Cách 1 – Qua script (tự động)

```bash
python scripts/simulate_predictions.py --upload-reference --n 0
```

### Cách 2 – Thủ công qua curl

```bash
python -c "
import pandas as pd, json, requests
df = pd.read_csv('data/raw/Dataset/card_valid_fe.csv').drop(columns=['target'])
sample = df.sample(500, random_state=42)
payload = {
    'data': sample.to_dict(orient='records'),
    'description': 'Validation set reference data'
}
r = requests.post('http://localhost:8001/reference', json=payload)
print(r.json())
"
```

### 7.3 Kiểm tra reference đã load

```bash
curl http://localhost:8001/reference
# => {"loaded": true, "samples": 500, "features": [...]}
```

---

## 8. Bước 6 – Simulate Traffic & Drift

### 8.1 Normal traffic (không drift)

```bash
python scripts/simulate_predictions.py --n 200
```

### 8.2 Traffic với drift nhân tạo

```bash
python scripts/simulate_predictions.py --n 200 --drift
```

### 8.3 Trigger drift analysis thủ công

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"window_size": 200}'
```

Response:
```json
{
  "drift_detected": true,
  "drift_score": 0.45,
  "drifted_features": ["avg_casa_this_m", "cr_amt_mtd_fcy_casa", "..."],
  "drifted_count": 8,
  "report_url": "/reports/drift_report_20260418_143000.html"
}
```

### 8.4 Xem HTML drift report

Mở http://localhost:8001/reports để xem danh sách reports, hoặc truy cập URL từ response trên.

---

## 9. Truy Cập Dashboards

| Service | URL | Thông tin đăng nhập |
|---------|-----|---------------------|
| **Grafana** | http://localhost:3000 | admin / admin |
| **MLflow** | http://localhost:5000 | – |
| **Prometheus** | http://localhost:9090 | – |
| **MinIO Console** | http://localhost:9001 | minio / minio123 |
| **API Swagger** | http://localhost:8000/docs | – |
| **Evidently Reports** | http://localhost:8001/reports | – |

### Grafana Dashboard

1. Mở http://localhost:3000
2. Login: `admin` / `admin`
3. Vào **Dashboards** → **Credit Card MLOps** → **Credit Card MLOps – Model Monitoring**

Dashboard gồm các panels:

| Panel | Mô tả |
|-------|-------|
| Requests / 5m | Tổng requests mỗi 5 phút |
| Prediction Latency p95 | Độ trễ dự đoán (percentile 95) |
| Prediction Error Rate | Tỉ lệ lỗi |
| Model Version | Version đang phục vụ |
| Request Rate | Biểu đồ requests theo thời gian |
| Latency Percentiles | p50 / p95 / p99 |
| Propensity Score Histogram | Phân phối score |
| Drift Status | Có drift hay không |
| Overall Drift Score | Score drift (0–1) |
| Drifted Features Count | Số features bị drift |
| Feature Drift Timeline | Drift từng feature theo thời gian |

### Prometheus Queries hữu ích

```promql
# Requests per giây
rate(api_requests_total[5m])

# Latency p95
histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))

# Tỉ lệ lỗi
rate(model_prediction_errors_total[5m]) / rate(model_predictions_total[5m])

# Drift score hiện tại
evidently_drift_score

# Số features bị drift
evidently_drifted_features_count

# Propensity score trung bình
rate(model_propensity_score_sum[5m]) / rate(model_propensity_score_count[5m])
```

---

## 10. Chạy Unit Tests

### Cài test dependencies

```bash
pip install pytest pytest-cov httpx fastapi
pip install -r requirements.txt
```

### Chạy tất cả tests

```bash
pytest tests/ -v
```

### Chạy từng nhóm test

```bash
# Test API endpoints (mock model)
pytest tests/test_api.py -v

# Test data quality (dùng sample CSV)
pytest tests/test_data_quality.py -v

# Test feature engineering
pytest tests/test_preprocessing.py -v
```

### Chạy với coverage

```bash
pytest tests/ -v --cov=api --cov-report=term-missing
```

**Output mong đợi:**
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_200 PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_success PASSED
tests/test_data_quality.py::TestSchemaValidation::test_required_columns_present PASSED
tests/test_preprocessing.py::TestEngineeredFeatures::test_all_engineered_cols_present PASSED
...
```

---

## 11. API Reference

### POST /predict

Dự đoán propensity score cho 1 khách hàng.

**Request body:**
```json
{
  "features": [46, 1, 8.32, ...],
  "feature_names": ["age", "gender", ...],
  "customer_id": "CUST_001"
}
```

**33 features theo thứ tự:**
```
age, gender, tenure_to_bank, avg_casa_this_m,
avg_bal_amt_6mtd_fcy_casa, avg_bal_amt_ytd_fcy_casa,
cr_amt_mtd_fcy_casa, dr_amt_mtd_fcy_casa,
cr_amt_qtd_fcy_casa, dr_amt_qtd_fcy_casa,
avg_loan_lmt, max_loan_lmt, max_loan_dsbr_amt,
avg_loan_duration, avg_td_last_2m, sum_td_this_m,
debit_credit_ratio, cash_pressure, pressure_score,
spend_velocity, txn_velocity, momentum_score,
casa_trend, balance_change_ratio, utilization_ratio,
loan_gap, loan_pressure_ratio, tenure_x_util,
age_x_spend, eligible_by_age, high_spend_flag,
low_balance_flag, active_txn_flag
```

**Response:**
```json
{
  "propensity_score": 0.712345,
  "will_open_card": true,
  "threshold": 0.5,
  "model_name": "credit_card_propensity",
  "model_version": "1",
  "customer_id": "CUST_001",
  "timestamp": "2026-04-18T07:30:00",
  "latency_ms": 8.5
}
```

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_card_propensity",
  "model_version": "1",
  "uptime_seconds": 3600.0
}
```

### GET /model/info

Thông tin chi tiết model, danh sách features, MLflow URI.

### POST /model/reload

Reload model mới nhất từ MLflow Registry (không cần restart container).

### GET /metrics

Prometheus metrics endpoint – được scrape tự động bởi Prometheus.

---

## 12. Troubleshooting

### API không start được: "Model not loaded"

```bash
# Kiểm tra model đã đăng ký chưa
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=credit_card_propensity

# Nếu chưa có → chạy lại training
python scripts/train_and_register.py

# Restart API
docker-compose restart api
docker-compose logs -f api
```

### MLflow không healthy

```bash
docker-compose logs mlflow
# Thường do postgres chưa sẵn sàng → đợi thêm hoặc restart
docker-compose restart mlflow
```

### MinIO bucket không tạo được

```bash
docker-compose logs minio-init

# Tạo thủ công
docker-compose exec minio sh -c "
  mc alias set local http://localhost:9000 minio minio123 &&
  mc mb local/mlflow-artifacts --ignore-existing
"
```

### Grafana không thấy metrics

```bash
# Kiểm tra Prometheus scrape
curl http://localhost:9090/-/healthy
curl http://localhost:9090/api/v1/targets

# Test datasource trong Grafana
# Vào: http://localhost:3000 → Configuration → Data Sources → Prometheus → Test
```

### Evidently: "Need at least 100 samples"

```bash
# Gửi thêm traffic
python scripts/simulate_predictions.py --n 150

# Sau đó trigger analyze
curl -X POST http://localhost:8001/analyze -H "Content-Type: application/json" -d '{}'
```

### Kiểm tra logs tất cả services

```bash
docker-compose logs -f api
docker-compose logs -f evidently
docker-compose logs -f mlflow
docker-compose logs -f prometheus
docker-compose logs -f grafana
```

---

## 13. Reset Toàn Bộ

```bash
# Dừng tất cả
docker-compose down

# Xóa volumes (XÓA TOÀN BỘ DATA – cẩn thận!)
docker-compose down -v

# Build lại và start fresh
docker-compose build --no-cache
docker-compose up -d postgres minio minio-init mlflow

# Train lại model
python scripts/train_and_register.py

# Start services còn lại
docker-compose up -d api evidently prometheus grafana

# Upload reference data và simulate
python scripts/simulate_predictions.py --upload-reference --n 200
```

---

## Tóm Tắt Quy Trình (Quick Reference)

```bash
# 1. Chuẩn bị
cp .env.example .env
pip install -r scripts/requirements.txt

# 2. Start infrastructure
docker-compose up -d postgres minio minio-init mlflow
# Đợi ~60s cho services healthy

# 3. Train model
python scripts/train_and_register.py

# 4. Start toàn bộ stack
docker-compose up -d api evidently prometheus grafana

# 5. Upload reference + simulate
python scripts/simulate_predictions.py --upload-reference --n 200

# 6. Xem kết quả
# Grafana:    http://localhost:3000   (admin/admin)
# MLflow:     http://localhost:5000
# API Docs:   http://localhost:8000/docs
# Evidently:  http://localhost:8001/reports
# Prometheus: http://localhost:9090
```
