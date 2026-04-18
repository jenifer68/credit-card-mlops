# 🚀 Hướng Dẫn Chi Tiết: Setup & Chạy Dự Án Credit Card MLOps

> **Mục tiêu:** Hướng dẫn từng bước một – từ lúc clone repo cho đến khi
> Grafana hiển thị metrics và GitHub Actions chạy xanh.  
> Mỗi lệnh đều có giải thích **tại sao** làm vậy.

---

## Mục Lục

- [0. Kiến trúc tổng quan – hiểu trước khi làm](#0-kiến-trúc-tổng-quan)
- [1. Cài đặt Prerequisites](#1-cài-đặt-prerequisites)
- [2. Clone & cấu hình môi trường](#2-clone--cấu-hình-môi-trường)
- [3. Hiểu cấu trúc Docker Compose](#3-hiểu-cấu-trúc-docker-compose)
- [4. Khởi động Infrastructure (Database + Storage)](#4-khởi-động-infrastructure)
- [5. Train & Register Model vào MLflow](#5-train--register-model-vào-mlflow)
- [6. Khởi động API + Monitoring Stack](#6-khởi-động-api--monitoring-stack)
- [7. Upload Reference Data & Test Drift Detection](#7-upload-reference-data--test-drift-detection)
- [8. Xem kết quả trên Grafana & Prometheus](#8-xem-kết-quả-trên-grafana--prometheus)
- [9. Chạy Tests & Kiểm tra Coverage](#9-chạy-tests--kiểm-tra-coverage)
- [10. GitHub CI/CD Pipeline – Giải thích từng bước](#10-github-cicd-pipeline)
- [11. Workflow vận hành hằng ngày](#11-workflow-vận-hành-hằng-ngày)
- [12. Troubleshooting chi tiết](#12-troubleshooting-chi-tiết)

---

## 0. Kiến Trúc Tổng Quan

Trước khi bắt tay vào làm, hãy hiểu **luồng dữ liệu** trong hệ thống:

```
[CSV Data]
    │
    ▼ python scripts/train_and_register.py
[MLflow Registry]  ◄──── Lưu model artifact vào MinIO (S3)
    │                     Lưu metadata vào PostgreSQL
    ▼ API khởi động, load model từ Registry
[FastAPI :8000]  ◄──── Nhận request /predict từ client
    │                   Trả về propensity_score [0–1]
    │
    ├──► [Prometheus :9090]  ◄──── Scrape metrics từ /metrics endpoint
    │        │
    │        ▼
    │    [Grafana :3000]  ──── Hiển thị dashboard
    │
    └──► [Evidently :8001]  ◄──── Nhận prediction data qua /capture
              │                   Phân tích drift qua /analyze
              └──► [Prometheus]   Expose drift metrics
```

**8 services trong Docker Compose:**

| Service      | Vai trò                           | Port  |
|-------------|-----------------------------------|-------|
| `postgres`  | Database cho MLflow               | 5432  |
| `minio`     | Lưu model artifacts (S3)          | 9000  |
| `minio-init`| Tạo bucket lần đầu (rồi tắt)     | –     |
| `mlflow`    | Experiment tracking + Registry    | 5000  |
| `api`       | FastAPI – serve predictions       | 8000  |
| `evidently` | Drift & data quality monitoring   | 8001  |
| `prometheus`| Thu thập và lưu metrics           | 9090  |
| `grafana`   | Visualize metrics trên dashboard  | 3000  |

---

## 1. Cài Đặt Prerequisites

### 1.1 Docker Desktop

Docker là nền tảng chạy containers – mỗi service (API, MLflow, Grafana...)
chạy trong container riêng biệt, không ảnh hưởng nhau.

**Windows:**
```
https://www.docker.com/products/docker-desktop/
```
Sau khi cài, mở Docker Desktop và đợi biểu tượng whale ở taskbar chuyển xanh.

**Kiểm tra:**
```bash
docker --version
# Docker version 24.0.x, build xxxxx

docker compose version
# Docker Compose version v2.x.x
```

> ⚠️ **Lưu ý:** Phải dùng `docker compose` (v2, không có dấu gạch ngang),
> không phải `docker-compose` (v1 cũ). Nếu máy bạn chỉ có v1 thì install lại
> Docker Desktop mới nhất.

### 1.2 Python 3.10+

Python cần để chạy training scripts và tests trên máy local
(không phải trong Docker).

```bash
# Kiểm tra
python --version
# Python 3.10.x

# Nếu chưa có, tải tại:
# https://www.python.org/downloads/
```

### 1.3 Git

```bash
git --version
# git version 2.x.x
```

---

## 2. Clone & Cấu Hình Môi Trường

### 2.1 Clone repository

```bash
git clone https://github.com/jenifer68/credit-card-mlops.git
cd credit-card-mlops
```

### 2.2 Tạo file `.env`

File `.env` chứa tất cả **biến môi trường** – mật khẩu, tên database,
tên model, v.v. Docker Compose đọc file này để điền vào các service.

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

**Giải thích từng biến trong `.env`:**

```env
# ── Tên user (dùng để đặt tên container, volume)
USER=student
# → Container sẽ có tên: student_cc-api, student_cc-mlflow, ...

# ── MLflow
MLFLOW_PORT=5000          # Port MLflow server lắng nghe
MODEL_NAME=credit_card_propensity  # Tên model trong Registry
MODEL_STAGE=Production    # Stage để API load (Production/Staging)

# ── MinIO (S3-compatible storage cho model artifacts)
MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123
MINIO_BUCKET_NAME=mlflow-artifacts  # Bucket chứa model files

# ── PostgreSQL (backend database của MLflow)
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow

# ── Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# ── API
API_PORT=8000

# ── Evidently
EVIDENTLY_PORT=8001
EVIDENTLY_DRIFT_THRESHOLD=0.1   # Ngưỡng drift (10% features = alert)
EVIDENTLY_MIN_SAMPLES=100        # Cần ít nhất 100 samples mới analyze
```

> **Tại sao không hardcode thẳng vào docker-compose.yml?**  
> Vì `.env` được thêm vào `.gitignore` – mật khẩu production không bị
> commit lên GitHub. `.env.example` là template an toàn để share.

### 2.3 Tạo virtual environment cho Python scripts

```bash
# Tạo môi trường Python riêng biệt
python -m venv .venv

# Kích hoạt
# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# Cài dependencies
pip install -r scripts/requirements.txt

# Bạn sẽ thấy prefix (.venv) ở terminal sau khi activate thành công
```

> **Tại sao cần venv?**  
> Tránh xung đột package giữa các project. Khi `deactivate` thì môi trường
> Python global không bị ảnh hưởng.

---

## 3. Hiểu Cấu Trúc Docker Compose

File `docker-compose.yml` định nghĩa toàn bộ stack. Hãy đọc từng phần:

### 3.1 Phần `services`

Mỗi service là 1 container. Ví dụ phân tích service `mlflow`:

```yaml
mlflow:
  build:
    context: ./mlflow        # Đường dẫn tới Dockerfile
    dockerfile: Dockerfile
  container_name: ${USER}_cc-mlflow  # Tên container (dùng biến .env)
  environment:
    - AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER:-minio}
    # ↑ Biến env truyền vào container. Cú pháp ${VAR:-default}
    #   nghĩa là: dùng VAR nếu có, không thì dùng "minio"
    - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
    # ↑ "minio" ở đây là tên service trong docker-compose,
    #   Docker tự resolve thành IP của container minio
  depends_on:
    postgres:
      condition: service_healthy  # Chỉ start SAU KHI postgres healthy
    minio:
      condition: service_healthy
  healthcheck:
    test: ["CMD-SHELL", "python -c \"import urllib.request; ...\""]
    interval: 30s    # Kiểm tra mỗi 30s
    timeout: 10s     # Timeout mỗi lần check
    retries: 5       # Thử lại 5 lần trước khi báo unhealthy
    start_period: 40s # Cho phép 40s để container khởi động trước khi check
  networks:
    - cc-mlops       # Tất cả service trong cùng 1 network
                     # → có thể gọi nhau qua tên service
```

### 3.2 Phần `volumes`

```yaml
volumes:
  postgres_data:
    driver: local
    name: ${USER}_cc-postgres_data
```

Volume là **persistent storage** – data không mất khi container restart.
PostgreSQL lưu database vào volume này.

### 3.3 Phần `networks`

```yaml
networks:
  cc-mlops:
    driver: bridge
```

Tất cả services trong cùng network `cc-mlops` có thể gọi nhau qua
tên service (VD: `http://mlflow:5000`, `http://minio:9000`).
Từ bên ngoài, chỉ truy cập qua `localhost:PORT` đã expose.

### 3.4 Service phụ đặc biệt: `minio-init`

```yaml
minio-init:
  image: minio/mc:latest   # MinIO Client – công cụ CLI
  depends_on:
    minio:
      condition: service_healthy
  entrypoint: >
    /bin/sh -c "
    mc alias set myminio http://minio:9000 minio minio123;
    mc mb myminio/mlflow-artifacts --ignore-existing;
    exit 0;
    "
```

Container này chỉ chạy **1 lần** để tạo bucket `mlflow-artifacts`,
sau đó tự tắt (`exit 0`). Đây là pattern "init container" phổ biến.

---

## 4. Khởi Động Infrastructure

### 4.1 Start database + storage layer trước

```bash
docker compose up -d postgres minio minio-init mlflow
```

**Giải thích lệnh:**
- `docker compose up` – tạo và start containers
- `-d` (detached) – chạy ngầm, không chiếm terminal
- `postgres minio minio-init mlflow` – chỉ start 4 services này trước
  (API cần model trước khi start)

**Xem quá trình khởi động:**

```bash
# Xem log realtime (Ctrl+C để thoát nhưng container vẫn chạy)
docker compose logs -f

# Hoặc xem từng service
docker compose logs -f mlflow
```

### 4.2 Kiểm tra trạng thái

```bash
docker compose ps
```

Output mong đợi sau ~60 giây:

```
NAME                    STATUS
student_cc-postgres     Up (healthy)    ✅
student_cc-minio        Up (healthy)    ✅
student_cc-minio-init   Exited (0)      ✅ (init xong rồi tắt – bình thường)
student_cc-mlflow       Up (healthy)    ✅
```

> ⚠️ Nếu thấy `(unhealthy)` hoặc `Restarting` – xem phần
> [Troubleshooting](#12-troubleshooting-chi-tiết)

### 4.3 Verify MLflow đang chạy

```bash
# Test health endpoint
curl http://localhost:5000/health
# Hoặc mở trình duyệt: http://localhost:5000
```

Bạn sẽ thấy MLflow UI với tab **Experiments** và **Models** (đang trống).

---

## 5. Train & Register Model Vào MLflow

### 5.1 Hiểu script `train_and_register.py`

Script này làm **5 việc** theo thứ tự:

```
1. Load data   → data/raw/Dataset/card_train_fe.csv + card_valid_fe.csv
       │
       ▼
2. Train LightGBM champion model
   (hyperparams: learning_rate=0.05, num_leaves=31, early_stopping=100)
       │
       ▼
3. Log vào MLflow:
   - Parameters (hyperparams)
   - Metrics   (AUC, KS, Lift@10%)
   - Artifacts (model file, feature importance CSV)
       │
       ▼
4. Register model "credit_card_propensity" → MLflow Registry
   Promote sang stage "Production"
       │
       ▼
5. Lưu reference data (validation set 1000 rows)
   → data/reference/reference_data.csv  (để Evidently dùng sau)
```

### 5.2 Cấu hình biến môi trường cho script

Script cần biết MLflow và MinIO ở đâu. Trên Windows PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID = "minio"
$env:AWS_SECRET_ACCESS_KEY = "minio123"
$env:DATA_DIR = "data/raw/Dataset"
```

Trên Linux/macOS:
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minio"
export AWS_SECRET_ACCESS_KEY="minio123"
export DATA_DIR="data/raw/Dataset"
```

> **Tại sao cần `MLFLOW_S3_ENDPOINT_URL`?**  
> MLflow dùng boto3 (AWS SDK) để upload artifacts. Boto3 mặc định kết nối
> AWS S3 thật. Ta phải override để nó kết nối MinIO local thay vì AWS.

### 5.3 Chạy training

```bash
python scripts/train_and_register.py
```

**Output mong đợi:**
```
Loading data from data/raw/Dataset
Train : (400000, 34) | target_rate=0.1823
Valid : (100000, 34) | target_rate=0.1819
Setting up MLflow... uri=http://localhost:5000
Starting run: run_id=abc123def456...
[LightGBM] [Info] Number of positive: 72920, number of negative: 327080
Training until validation scores don't improve for 100 rounds
[100] valid_0's auc: 0.743201
[200] valid_0's auc: 0.761432
...
[547] valid_0's auc: 0.782341
Early stopping, best iteration is: 447

=== Validation Metrics ===
  AUC   : 0.7823
  KS    : 0.4215
  Lift@10%: 2.53x

=== Test Metrics ===
  AUC   : 0.7789
  KS    : 0.4187
  Lift@10%: 2.47x

Registering model: credit_card_propensity
✅ Model promoted to Production | version=1

Reference data saved: data/reference/reference_data.csv (1000 rows)

==============================================================
  TRAINING COMPLETE
  Model   : credit_card_propensity
  Version : 1
  Run ID  : abc123def456...
  Valid AUC: 0.7823
==============================================================
```

### 5.4 Kiểm tra trên MLflow UI

1. Mở http://localhost:5000
2. Click **Experiments** → thấy run vừa train
3. Click vào run → xem params, metrics, artifacts
4. Click **Models** → thấy `credit_card_propensity` → Version 1 → **Production**

---

## 6. Khởi Động API + Monitoring Stack

### 6.1 Start toàn bộ stack còn lại

```bash
docker compose up -d api evidently prometheus grafana
```

Lần đầu chạy sẽ mất 2–5 phút vì Docker phải **build images** từ Dockerfile.
Các lần sau sẽ nhanh hơn vì image đã được cache.

**Theo dõi build progress:**
```bash
# Xem build log (có thanh progress)
docker compose logs -f api
```

### 6.2 Kiểm tra từng service

```bash
# API
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "credit_card_propensity",
  "model_version": "1",
  "uptime_seconds": 45.2
}
```

```bash
# Evidently
curl http://localhost:8001/health
```
```json
{
  "status": "healthy",
  "reference_loaded": false,
  "current_samples": 0
}
```

```bash
# Prometheus
curl http://localhost:9090/-/healthy
# Prometheus is Healthy.
```

```bash
# Grafana
curl http://localhost:3000/api/health
# {"commit":"...","database":"ok","version":"..."}
```

### 6.3 Test prediction đầu tiên

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [46, 1, 8.32, 3310613.69, 3367367.83, 2348056.57,
                 3053211.37, 3847988.53, 7404976.06, 13258901.81,
                 78920973.24, 119569150.41, 48365930.36, 70.93,
                 2950061.74, 1766109.95, 1.26, 0.61, 5853925.75,
                 0.29, 0.41, 0.35, 0.98, 0.70, 1.64,
                 40648177.18, 0.61, 13.64, 13.35, 1, 1, 0, 1],
    "customer_id": "TEST_001"
  }'
```

**Response:**
```json
{
  "propensity_score": 0.7123,
  "will_open_card": true,
  "threshold": 0.5,
  "model_name": "credit_card_propensity",
  "model_version": "1",
  "customer_id": "TEST_001",
  "latency_ms": 8.3
}
```

### 6.4 Xem raw Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

Bạn sẽ thấy output dạng text như:
```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 1.0
# HELP model_prediction_latency_seconds Prediction latency
...
```

Đây là format **Prometheus exposition format** – Prometheus sẽ scrape
endpoint này mỗi 10 giây.

---

## 7. Upload Reference Data & Test Drift Detection

### 7.1 Upload reference data

Reference data là **baseline** để Evidently so sánh với production data.
Ta dùng validation set (1000 rows đã lưu từ bước training).

```bash
python scripts/simulate_predictions.py --upload-reference --n 0
```

**Output:**
```
Uploading reference data...
POST http://localhost:8001/reference
✅ Reference uploaded: 1000 samples, 33 features
```

**Verify:**
```bash
curl http://localhost:8001/reference
```
```json
{
  "loaded": true,
  "samples": 1000,
  "features": ["age", "gender", "tenure_to_bank", ...]
}
```

### 7.2 Gửi traffic bình thường (không drift)

```bash
python scripts/simulate_predictions.py --n 200
```

Script này làm 2 việc:
1. **Gửi 200 requests** đến `POST /predict` (API)
2. **Capture** mỗi prediction vào `POST /capture` (Evidently)

### 7.3 Trigger drift analysis

```bash
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"window_size": 200}'
```

**Response (không có drift):**
```json
{
  "drift_detected": false,
  "drift_score": 0.06,
  "drifted_features": [],
  "drifted_count": 2,
  "total_features": 33,
  "samples_analyzed": 200,
  "report_url": "/reports/drift_report_20260418_150000.html"
}
```

### 7.4 Mô phỏng drift nhân tạo

```bash
# Gửi data có drift (các features bị shift về phân phối khác)
python scripts/simulate_predictions.py --n 300 --drift

# Trigger analyze lại
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"window_size": 300}'
```

**Response (có drift):**
```json
{
  "drift_detected": true,
  "drift_score": 0.45,
  "drifted_features": ["avg_casa_this_m", "cr_amt_mtd_fcy_casa", ...],
  "drifted_count": 15,
  "report_url": "/reports/drift_report_20260418_151200.html"
}
```

### 7.5 Xem HTML drift report

Mở trình duyệt: `http://localhost:8001/reports`

Click vào report mới nhất → Evidently render HTML report với charts
cho thấy distribution shift của từng feature.

---

## 8. Xem Kết Quả Trên Grafana & Prometheus

### 8.1 Grafana Dashboard

1. Mở http://localhost:3000
2. Đăng nhập: `admin` / `admin`
3. Menu trái → **Dashboards** → **Credit Card MLOps** →
   **Credit Card MLOps – Model Monitoring**

**Giải thích từng panel:**

| Panel | Query Prometheus | Ý nghĩa |
|-------|-----------------|---------|
| Requests / 5m | `sum(increase(api_requests_total[5m]))` | Tổng requests trong 5 phút |
| Latency p95 | `histogram_quantile(0.95, rate(model_prediction_latency_seconds_bucket[5m]))` | 95% requests hoàn thành trong bao nhiêu giây |
| Error Rate | `rate(model_prediction_errors_total[5m]) / rate(model_predictions_total[5m])` | Tỉ lệ requests lỗi |
| Drift Score | `evidently_drift_score` | Score drift hiện tại (0=không drift, 1=full drift) |
| Drifted Features | `evidently_drifted_features_count` | Số features bị drift |

### 8.2 Prometheus UI

Mở http://localhost:9090 → Vào tab **Graph** → thử các queries:

```promql
# Kiểm tra targets đang được scrape
# → Menu: Status → Targets
# Phải thấy: api:8000 (UP), evidently:8001 (UP)

# Prediction rate per second
rate(model_predictions_total[1m])

# Score distribution histogram
rate(model_propensity_score_bucket[5m])

# Drift status (1 = có drift)
evidently_data_drift_detected
```

### 8.3 Thêm Alert rule trong Grafana

1. Menu trái → **Alerting** → **Alert rules** → **New alert rule**
2. Condition: `evidently_drift_score > 0.3`
3. Evaluate every: `1m`
4. Message: "Data drift detected! Check Evidently report."

---

## 9. Chạy Tests & Kiểm Tra Coverage

### 9.1 Cài test dependencies

```bash
pip install pytest pytest-cov httpx
```

### 9.2 Chạy toàn bộ test suite

```bash
pytest tests/ -v --cov=api --cov-report=term-missing
```

**Giải thích flags:**
- `-v` (verbose) – hiển thị tên từng test case
- `--cov=api` – đo coverage cho package `api`
- `--cov-report=term-missing` – hiển thị dòng code chưa được test

**Output mong đợi:**
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_200 PASSED
tests/test_api.py::TestHealthEndpoint::test_health_model_loaded PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_success PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_wrong_feature_count PASSED
...
tests/test_data_quality.py::TestSchemaValidation::test_required_columns_present PASSED
tests/test_preprocessing.py::TestEngineeredFeatures::test_flag_columns_binary PASSED
...
==================== 30 passed in 5.23s ====================

Name         Stmts   Miss  Cover
--------------------------------
api/main.py    185     28    85%
```

### 9.3 Chạy từng nhóm test riêng

```bash
# Chỉ test API (nhanh, mock model)
pytest tests/test_api.py -v

# Chỉ test data quality (cần sample CSV)
pytest tests/test_data_quality.py -v

# Chỉ test feature engineering
pytest tests/test_preprocessing.py -v

# Test với keyword filter
pytest tests/ -k "predict" -v   # Chỉ test có chữ "predict" trong tên
```

### 9.4 Giải thích `tests/conftest.py`

```python
# tests/conftest.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

File này được pytest **tự động load** trước khi chạy tests.
Dòng `sys.path.insert` thêm root folder vào Python path → cho phép
`from api.main import app` hoạt động từ trong thư mục `tests/`.

---

## 10. GitHub CI/CD Pipeline

### 10.1 Tổng quan pipeline

File `.github/workflows/ci_cd.yml` định nghĩa 3 jobs chạy tuần tự:

```
Push code → GitHub
     │
     ▼
[Job 1: test]           ← Chạy mọi branch
  ├─ Setup Python 3.10
  ├─ pip install
  ├─ flake8 lint
  └─ pytest --cov

     │ (chỉ tiếp nếu test PASS)
     ▼
[Job 2: build]          ← Chạy mọi branch
  ├─ docker build api
  ├─ docker build evidently
  └─ docker build mlflow

     │ (chỉ tiếp nếu build OK VÀ đang ở branch main)
     ▼
[Job 3: integration]    ← CHỈ chạy khi push vào main
  ├─ docker compose up infrastructure
  ├─ python train_and_register.py (smoke test)
  ├─ docker compose up api
  ├─ curl /health
  └─ docker compose down -v
```

### 10.2 Cấu hình GitHub repository

#### Bước 1: Tạo GitHub repository

Đã có: `https://github.com/jenifer68/credit-card-mlops`

#### Bước 2: Push code lên (đã làm)

```bash
git add .
git commit -m "final project"
git push origin main
```

#### Bước 3: Xem pipeline chạy

1. Vào GitHub → repository → tab **Actions**
2. Bạn sẽ thấy workflow **"Credit Card MLOps – CI/CD"** đang chạy
3. Click vào để xem từng job

### 10.3 Giải thích từng phần của `ci_cd.yml`

```yaml
# Trigger: khi nào pipeline chạy
on:
  push:
    branches: [main, develop]   # Push vào main hoặc develop
  pull_request:
    branches: [main]            # Tạo PR vào main
```

```yaml
# Job 1: test
jobs:
  test:
    runs-on: ubuntu-latest      # Runner: máy ảo Ubuntu miễn phí

    steps:
      # Bước 1: Checkout code
      - uses: actions/checkout@v4
      # actions/checkout là action tải code từ repo về runner

      # Bước 2: Cài Python
      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      # Bước 3: Cài packages
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install flake8 pytest pytest-cov httpx
          pip install -r requirements.txt

      # Bước 4: Lint
      - name: Lint with flake8
        run: |
          flake8 api/main.py evidently/main.py scripts/ \
            --max-line-length=120 --ignore=E501,W503
          # E501: line too long → bỏ qua
          # W503: line break before binary operator → bỏ qua

      # Bước 5: Tests
      - name: Run unit tests
        run: |
          pytest tests/ -v --cov=api --cov-report=xml
        env:
          PYTHONPATH: .   # Để import api.main hoạt động
```

```yaml
  # Job 2: build (chỉ chạy sau khi test pass)
  build:
    needs: test    # ← Key quan trọng: dependency giữa jobs

    steps:
      - name: Build API image
        run: docker build -t credit-card-api:${{ github.sha }} ./api
        # github.sha = commit hash → mỗi build có tag unique
```

```yaml
  # Job 3: integration (chỉ khi push vào main)
  integration:
    needs: build
    if: github.ref == 'refs/heads/main'   # ← Điều kiện

    steps:
      - name: Copy env file
        run: cp .env.example .env

      - name: Start infrastructure
        run: |
          docker-compose up -d postgres minio minio-init mlflow
          sleep 30   # Đợi services healthy

      - name: Train & register model (smoke)
        run: python scripts/train_and_register.py
        env:
          MLFLOW_TRACKING_URI: http://localhost:5000
          # ... các env vars khác

      - name: Check API health
        run: curl --retry 5 --retry-delay 5 http://localhost:8000/health
        # --retry 5: thử lại 5 lần nếu fail
        # --retry-delay 5: đợi 5 giây giữa các lần retry

      - name: Tear down
        if: always()   # Luôn chạy dù pass hay fail
        run: docker-compose down -v
```

### 10.4 Xem badge status trong README

Thêm badge CI vào README.md để thấy trạng thái build:

```markdown
[![CI/CD](https://github.com/jenifer68/credit-card-mlops/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/jenifer68/credit-card-mlops/actions/workflows/ci_cd.yml)
```

### 10.5 GitHub Secrets (cho production)

Nếu deploy lên cloud (không phải local), các credentials cần được lưu
vào **GitHub Secrets** (không phải `.env`):

1. GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Thêm: `MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Trong `ci_cd.yml`:
```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
```

---

## 11. Workflow Vận Hành Hằng Ngày

### 11.1 Kiểm tra sức khỏe hệ thống buổi sáng

```bash
# Kiểm tra tất cả containers đang chạy
docker compose ps

# Kiểm tra API
curl http://localhost:8000/health

# Kiểm tra drift có xảy ra không
curl http://localhost:8001/health | python -m json.tool
```

### 11.2 Deploy model mới (không cần restart container)

```bash
# 1. Train và register version mới
python scripts/train_and_register.py

# 2. API tự động load version Production mới nhất
curl -X POST http://localhost:8000/model/reload

# 3. Verify
curl http://localhost:8000/model/info
# Kết quả: "model_version": "2"
```

### 11.3 Xem và rollback model version

Trên MLflow UI (http://localhost:5000 → Models → credit_card_propensity):
- Để rollback: Click vào version cũ → Change stage → "Production"
- Gọi API reload: `curl -X POST http://localhost:8000/model/reload`

### 11.4 Scale API khi cần

```bash
# Chạy 3 instances của API (load balancing)
docker compose up -d --scale api=3
```

---

## 12. Troubleshooting Chi Tiết

### ❌ Lỗi: `model_loaded: false` hoặc API trả về 503

**Nguyên nhân:** API start nhưng chưa load được model từ MLflow.

```bash
# Xem log API
docker compose logs api --tail=50

# Thường thấy:
# "No model found in stage Production"
# → Chưa train hoặc chưa promote model
```

**Fix:**
```bash
python scripts/train_and_register.py
docker compose restart api
```

---

### ❌ Lỗi: MLflow `Exited (1)` hoặc `unhealthy`

```bash
docker compose logs mlflow --tail=30

# Thường thấy:
# "could not connect to server: Connection refused"
# → PostgreSQL chưa sẵn sàng
```

**Fix:**
```bash
# Đợi PostgreSQL healthy trước
docker compose ps postgres
# Sau đó restart mlflow
docker compose restart mlflow
```

---

### ❌ Lỗi: `minio-init Exited (1)` (không phải Exited 0)

```bash
docker compose logs minio-init

# Thường thấy:
# "mc: <ERROR> Unable to connect to minio"
```

**Fix:**
```bash
# MinIO chưa healthy, chạy lại
docker compose restart minio-init
```

---

### ❌ Lỗi Evidently: `{"detail": "Need at least 100 samples"}`

**Nguyên nhân:** Chưa đủ data để analyze drift.

```bash
python scripts/simulate_predictions.py --n 150
curl -X POST http://localhost:8001/analyze -H "Content-Type: application/json" -d '{}'
```

---

### ❌ Lỗi: Grafana không hiển thị data

```bash
# Kiểm tra Prometheus có scrape được không
# Mở: http://localhost:9090/targets
# Tất cả targets phải ở trạng thái "UP"

# Nếu api:8000 là "DOWN":
docker compose ps api  # Kiểm tra api có running không
curl http://localhost:8000/metrics  # Test metrics endpoint
```

---

### ❌ Lỗi: `docker build` fails

```bash
# Xóa cache và build lại
docker compose build --no-cache api
docker compose build --no-cache evidently

# Nếu lỗi network trong Docker (Windows):
# Docker Desktop → Settings → Docker Engine
# Thêm: "dns": ["8.8.8.8"]
```

---

### ❌ Lỗi GitHub Actions: `pytest: command not found`

Xảy ra khi `requirements.txt` thiếu hoặc pip install fail.

```yaml
# Sửa trong ci_cd.yml:
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install pytest pytest-cov httpx  # Cài rõ ràng trước
    pip install -r requirements.txt
```

---

### 🧹 Reset toàn bộ (khi mọi thứ hỏng)

```bash
# Dừng tất cả
docker compose down

# Xóa volumes (mất toàn bộ data – cẩn thận!)
docker compose down -v

# Xóa cả images để build lại
docker compose down --rmi local -v

# Start fresh
docker compose up -d postgres minio minio-init mlflow
# Đợi 60s
python scripts/train_and_register.py
docker compose up -d api evidently prometheus grafana
python scripts/simulate_predictions.py --upload-reference --n 200
```

---

## Tóm Tắt Nhanh (Quick Reference Card)

```bash
# ── SETUP LẦN ĐẦU ──────────────────────────────────────────
cp .env.example .env
python -m venv .venv && .venv\Scripts\activate
pip install -r scripts/requirements.txt

# ── START STACK ─────────────────────────────────────────────
docker compose up -d postgres minio minio-init mlflow   # Đợi 60s
python scripts/train_and_register.py                    # Train model
docker compose up -d api evidently prometheus grafana   # Start tất cả

# ── GENERATE TRAFFIC ────────────────────────────────────────
python scripts/simulate_predictions.py --upload-reference --n 200

# ── URLS ────────────────────────────────────────────────────
# Grafana:    http://localhost:3000    (admin/admin)
# MLflow:     http://localhost:5000
# API Docs:   http://localhost:8000/docs
# Evidently:  http://localhost:8001
# Prometheus: http://localhost:9090
# MinIO:      http://localhost:9001    (minio/minio123)

# ── TESTS ───────────────────────────────────────────────────
pytest tests/ -v --cov=api

# ── DAILY CHECK ─────────────────────────────────────────────
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8001/health

# ── STOP ────────────────────────────────────────────────────
docker compose down           # Dừng, giữ volumes
docker compose down -v        # Dừng + xóa volumes
```
