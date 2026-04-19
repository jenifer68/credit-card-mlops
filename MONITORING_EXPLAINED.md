# Giai Thich: Prometheus - Grafana - MLflow trong Du An

> Tai lieu nay giai thich **vai tro, cach hoat dong, va luong du lieu**
> cua 3 cong cu cot loi trong he thong MLOps Credit Card Propensity.

---

## Muc Luc

1. [Tai sao can 3 cong cu nay?](#1-tai-sao-can)
2. [MLflow - Quan ly vong doi mo hinh](#2-mlflow)
3. [Prometheus - Thu thap metrics](#3-prometheus)
4. [Grafana - Hien thi va canh bao](#4-grafana)
5. [3 cong cu phoi hop voi nhau](#5-phoi-hop)
6. [Giai thich loi No data tren Grafana](#6-no-data)
7. [Checklist dam bao data hien thi dung](#7-checklist)

---

## 1. Tai sao can 3 cong cu nay?

Mot mo hinh ML khi dua vao production gap 3 van de khac nhau:

```
Van de 1: Model thay doi → can quan ly version
          Giai phap: MLflow

Van de 2: API cham? Model loi? → can do luong lien tuc
          Giai phap: Prometheus

Van de 3: Du lieu tho qua nhieu, can nhin truc quan
          Giai phap: Grafana
```

Moi cong cu giai quyet **1 bai toan rieng biet**:

| Cong cu | Bai toan giai quyet | Du lieu luu tru |
|---------|---------------------|-----------------|
| **MLflow** | Da train model gi? Ket qua bao nhieu? Dang dung version nao? | Experiments, params, metrics, model files |
| **Prometheus** | API dang hoat dong the nao? Latency? Error rate? | Time-series metrics |
| **Grafana** | Hien thi toan bo trang thai he thong truc quan | Khong luu gi - chi doc tu Prometheus |

---

## 2. MLflow - Quan Ly Vong Doi Mo Hinh

### 2.1 MLflow la gi?

MLflow la nen tang ma nguon mo quan ly **toan bo lifecycle cua ML model**:
tu training, evaluation, den deployment va versioning.

### 2.2 Cau truc cua MLflow

```
MLflow Server (port 5000)
|
+-- Tracking Server  <--- Luu ket qua experiments
|   +-- Experiments  (nhom cac runs)
|   +-- Runs         (moi lan train = 1 run)
|       +-- Parameters  (hyperparams: learning_rate, num_leaves...)
|       +-- Metrics     (AUC=0.782, KS=0.421, Lift@10%=2.53x)
|       +-- Artifacts   (model file, feature_importance.csv)
|
+-- Model Registry  <--- Quan ly model versions
|   +-- credit_card_propensity
|       +-- Version 1  [Production]  <-- API dang dung
|       +-- Version 2  [Staging]
|       +-- Version 3  [None]
|
+-- Backend Storage
    +-- PostgreSQL  <-- Luu metadata (params, metrics, tags)
    +-- MinIO (S3)  <-- Luu artifact files (model binary, CSVs)
```

### 2.3 Luong du lieu khi training

```python
# scripts/train_and_register.py lam 5 buoc:

# 1. Tao run moi
with mlflow.start_run(run_name="lightgbm_champion"):

    # 2. Log hyperparameters
    mlflow.log_params({
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
    })

    # 3. Train model...

    # 4. Log ket qua
    mlflow.log_metrics({
        "valid_auc": 0.782,
        "valid_ks": 0.421,
        "valid_lift10": 2.53,
    })

    # 5. Upload model len MinIO qua MLflow
    mlflow.lightgbm.log_model(model, "model")

# 6. Dang ky va promote Production
client.transition_model_version_stage(
    name="credit_card_propensity",
    version="1",
    stage="Production"
)
```

### 2.4 Luong du lieu khi API khoi dong

```python
# api/main.py - ModelManager.load_model()

# 1. API goi MLflow Registry
model_uri = "models:/credit_card_propensity/Production"

# 2. MLflow tra ve duong dan artifact tren MinIO
# → s3://mlflow-artifacts/1/abc123.../artifacts/model

# 3. MLflow download model file ve RAM
model = mlflow.pyfunc.load_model(model_uri)

# 4. API set Prometheus Gauge
MODEL_VERSION_INFO.labels(
    model_name="credit_card_propensity",
    version="1"
).set(1)
# → Prometheus biet: dang serve version 1
```

### 2.5 Cac trang quan trong trong MLflow UI

| URL | Noi dung |
|-----|---------|
| `http://localhost:5000` | Trang chu - danh sach experiments |
| `http://localhost:5000/#/experiments/1` | Chi tiet experiment, so sanh runs |
| `http://localhost:5000/#/models` | Model Registry |
| `http://localhost:5000/#/models/credit_card_propensity` | Cac versions cua model |

### 2.6 MLflow Backend: PostgreSQL + MinIO

```
Tai sao can 2 storage backend?

PostgreSQL: luu metadata nhe (params, metrics, tags)
  → truy van nhanh, co the SQL query
  → VD: "Show me all runs with AUC > 0.75"

MinIO (S3): luu files nang (model binary, CSV, PNG)
  → model file LightGBM co the 50-200 MB
  → PostgreSQL khong phu hop luu files lon

Cung giong nhu: database luu thong tin san pham,
S3 luu hinh anh san pham.
```

### 2.7 MLflow va API: khi nao tuong tac?

```
Chi anh huong 2 thoi diem:
  1. Luc API khoi dong → load model tu Registry
  2. Luc goi POST /model/reload → load version moi

Trong qua trinh serving binh thuong:
  API KHONG goi MLflow nua
  → MLflow down KHONG anh huong prediction dang chay
```

---

## 3. Prometheus - Thu Thap Metrics

### 3.1 Prometheus la gi?

Prometheus la he thong **thu thap va luu tru time-series metrics**.
Dac diem quan trong nhat: Prometheus **chu dong den hoi** (pull model),
khong phai service day data den Prometheus.

### 3.2 Co che Pull - Prometheus scrape metrics

```
                   Moi 10 giay
                        |
                        v
Prometheus ---GET /metrics---> FastAPI (api:8000)
                        <--- tra ve text metrics

Moi 30 giay:
Prometheus ---GET /metrics---> Evidently (evidently:8001)
                        <--- tra ve drift metrics
```

Cau hinh scrape trong `docker/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: credit-card-api
    metrics_path: /metrics      # URL endpoint tra metrics
    scrape_interval: 10s        # Hoi moi 10 giay
    static_configs:
      - targets: ["api:8000"]   # Dia chi scrape
```

### 3.3 Format metrics (Prometheus Exposition Format)

Khi goi `curl http://localhost:8000/metrics`:

```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 145.0

# HELP model_prediction_latency_seconds Model prediction latency
# TYPE model_prediction_latency_seconds histogram
model_prediction_latency_seconds_bucket{le="0.001",...} 0.0
model_prediction_latency_seconds_bucket{le="0.005",...} 12.0
model_prediction_latency_seconds_bucket{le="0.01",...}  130.0
model_prediction_latency_seconds_bucket{le="+Inf",...}  145.0
model_prediction_latency_seconds_sum{...}   0.893
model_prediction_latency_seconds_count{...} 145.0

# HELP model_version_info Current model version
# TYPE model_version_info gauge
model_version_info{model_name="credit_card_propensity",version="1"} 1.0
```

### 3.4 Cac loai Metric Types

| Type | Dac diem | Vi du trong du an |
|------|----------|-------------------|
| **Counter** | Chi tang, khong bao gio giam. Reset ve 0 khi restart | `api_requests_total`, `model_predictions_total` |
| **Histogram** | Phan phoi gia tri. Tu dong tao `_bucket`, `_sum`, `_count` | `model_prediction_latency_seconds`, `model_propensity_score` |
| **Gauge** | Co the tang hoac giam bat ky luc nao | `model_version_info`, `evidently_drift_score` |

**Dieu quan trong:**

```
Counter va Histogram chi xuat hien trong Prometheus
SAU KHI duoc goi lan dau tien.

Vi du:
- model_prediction_latency_seconds: chi co data sau prediction dau tien
- model_prediction_errors_total: chi co data sau loi dau tien

Neu chua co prediction nao → metric nay chua ton tai
→ Grafana hien "No data" (khac voi gia tri = 0)
```

### 3.5 Prometheus luu tru nhu the nao?

```
Prometheus luu vao Docker Volume: prometheus_data
Giu trong: 30 ngay (config trong docker-compose.yml)

Moi data point = (timestamp, value, labels)

Vi du - Counter:
  (10:00:00, 145, {endpoint="/predict", status="200"})
  (10:00:10, 147, {endpoint="/predict", status="200"})
  (10:00:20, 150, {endpoint="/predict", status="200"})

Prometheus luu raw data, khong tinh rate.
PromQL tinh rate() luc query: (150-145) / 20s = 0.25 req/s
```

### 3.6 PromQL - Ngon ngu Query

Grafana dung PromQL de lay du lieu tu Prometheus:

```promql
-- So requests moi giay (1 phut gan nhat)
rate(api_requests_total[1m])

-- Latency percentile 95 (5 phut gan nhat)
histogram_quantile(0.95,
  sum(rate(model_prediction_latency_seconds_bucket[5m])) by (le)
)
-- Giai thich:
-- rate(...bucket[5m])  → ty le tang cua tung bucket trong 5 phut
-- sum(...) by (le)     → gop tat ca labels, giu lai bucket boundary
-- histogram_quantile() → tinh gia tri ma 95% request nho hon

-- Error rate (%)
(sum(rate(model_prediction_errors_total[5m])) or vector(0))
/ (sum(rate(model_predictions_total[5m])) + 0.0001) * 100
-- "or vector(0)" → tra ve 0 thay vi "No data" neu chua co errors
```

### 3.7 Kiem tra Prometheus dang hoat dong

Truy cap: `http://localhost:9090/targets`

```
Ket qua mong doi:
  credit-card-api  →  State: UP   (scrape OK)
  evidently        →  State: UP   (scrape OK)
  prometheus       →  State: UP   (tu scrape chinh no)

Neu thay "DOWN":
  → Service do khong chay
  → Hoac /metrics endpoint bi loi
  → Chay: docker compose ps de kiem tra
```

---

## 4. Grafana - Hien Thi va Canh Bao

### 4.1 Grafana la gi?

Grafana la cong cu **visualization** - no **KHONG** luu tru bat ky du lieu nao.
Grafana chi doc tu Prometheus va ve do thi.

```
Prometheus ----- luu time-series data
     |
     | PromQL query moi khi refresh dashboard (30 giay)
     v
Grafana -------- chi visualize, khong luu gi
     |
     v
Browser -------- hien thi charts, stat panels, alerts
```

### 4.2 Cac thanh phan chinh cua Grafana

```
Grafana
|
+-- Data Sources: noi Grafana doc du lieu
|   +-- Prometheus (uid: prometheus-datasource)
|       → url: http://prometheus:9090
|
+-- Dashboards: tap hop cac panels
|   +-- Credit Card MLOps - Model Monitoring
|       +-- Row 1: API Overview
|       |   +-- [Stat] Requests / 5m
|       |   +-- [Stat] Prediction Latency p95
|       |   +-- [Stat] Prediction Error Rate
|       |   +-- [Stat] Model Version
|       |
|       +-- Row 2: Request Monitoring
|       |   +-- [Timeseries] Request Rate (1m)
|       |   +-- [Timeseries] Latency Percentiles
|       |
|       +-- Row 3: Propensity Score Distribution
|       |   +-- [Histogram] Score Distribution
|       |   +-- [Timeseries] Predictions per Second
|       |
|       +-- Row 4: Drift Monitoring (Evidently)
|           +-- [Stat] Drift Status
|           +-- [Stat] Overall Drift Score
|           +-- [Stat] Drifted Features Count
|           +-- [Timeseries] Feature Drift Timeline
|
+-- Alerting: canh bao khi metric vuot nguong
    +-- Alert Rule: drift_score > 0.3
```

### 4.3 Loai panels va y nghia

| Panel type | Dung khi | Vi du trong du an |
|-----------|---------|-------------------|
| **Stat** | Hien 1 gia tri lon, mau sac canh bao | Latency p95, Model Version, Drift Status |
| **Timeseries** | Xu huong theo thoi gian | Request Rate, Latency Percentiles |
| **Histogram** | Phan phoi gia tri | Propensity Score Distribution |

### 4.4 Grafana tu dong provisioning

Khi Docker Compose khoi dong Grafana, no tu dong load cau hinh tu:

```
grafana/provisioning/datasources/prometheus.yml
→ Tu dong them Prometheus la Data Source
→ uid: "prometheus-datasource" (phai trung voi dashboard JSON)

grafana/provisioning/dashboards/dashboards.yml
→ Cau hinh thu muc chua dashboard JSON

grafana/dashboards/credit_card_mlops.json
→ Dashboard duoc load tu dong, khong can import tay
```

### 4.5 Cach doc Stat panel

```
┌──────────────────────────────┐
│    Prediction Latency p95    │
│                              │
│          8.3ms               │  ← Gia tri (lastNotNull)
│                              │
└──────────────────────────────┘
  xanh = OK (<100ms)
  vang  = WARN (100-500ms)
  do    = ALERT (>500ms)
```

Nguong mau duoc dinh nghia trong `fieldConfig.thresholds` cua dashboard JSON.

---

## 5. 3 Cong Cu Phoi Hop Voi Nhau

### 5.1 Luong day du tu Training den Dashboard

```
BUOC 1: Train model
  python scripts/train_and_register.py
  |
  +---> MLflow luu: params, metrics, model file
  +---> MLflow Registry: Version 1 → Production

BUOC 2: API khoi dong (docker compose up api)
  |
  +---> API goi MLflow: load model Production
  +---> API dat Gauge: model_version_info{version="1"} = 1
  +---> API san sang nhan request

BUOC 3: Client gui prediction
  POST http://localhost:8000/predict
  |
  +---> API chay inference → tra ve propensity_score
  +---> API tang Counter: model_predictions_total += 1
  +---> API ghi Histogram: model_prediction_latency_seconds
  +---> API ghi Histogram: model_propensity_score

BUOC 4: Prometheus scrape (moi 10 giay)
  GET http://api:8000/metrics
  |
  +---> Prometheus luu data points voi timestamp

BUOC 5: Grafana hien thi (refresh moi 30 giay)
  PromQL → Prometheus → Grafana ve chart
  |
  +---> Latency p95 = 8.3ms (mau xanh)
  +---> Requests/5m = 145
  +---> Model Version = v1
```

### 5.2 Luong drift detection

```
BUOC 1: Sau khi train, luu reference data
  python scripts/train_and_register.py
  → Luu data/reference/reference_data.csv (1000 rows tu valid set)

BUOC 2: Upload reference len Evidently
  python scripts/simulate_predictions.py --upload-reference
  POST http://localhost:8001/reference
  → Evidently luu: phan phoi goc cua 33 features

BUOC 3: Moi prediction duoc "capture" vao Evidently
  POST http://localhost:8001/capture {features, score}
  → Evidently luu production data

BUOC 4: Trigger drift analysis
  POST http://localhost:8001/analyze {window_size: 200}
  → Evidently so sanh production vs reference
  → Tinh drift score cho tung feature
  → Cap nhat Prometheus Gauge:
      evidently_data_drift_detected = 1 (co drift)
      evidently_drift_score = 0.45
      evidently_feature_drift{feature="avg_casa_this_m"} = 1

BUOC 5: Prometheus scrape Evidently (moi 30 giay)
  GET http://evidently:8001/metrics

BUOC 6: Grafana hien thi Drift Status = "DRIFT DETECTED" (mau do)
```

### 5.3 So sanh vai tro

```
                  TRUOC PRODUCTION     TRONG PRODUCTION
                  (Offline)            (Online)
                  
MLflow:           Luu experiments      Cung cap model cho API
                  So sanh models       Version history
                  Promote Production   Rollback

Prometheus:       (Khong lien quan)    Thu thap metrics API
                                       Luu time-series data
                                       Nen tang cho alerting

Grafana:          (Khong lien quan)    Hien thi dashboard
                                       Canh bao drift/latency
                                       On-call monitoring
```

---

## 6. Giai Thich Loi "No Data" Tren Grafana

Day la nguyen nhan **goc re** cho tung panel bi "No data":

### Panel: Prediction Latency p95

**PromQL:** `histogram_quantile(0.95, sum(rate(model_prediction_latency_seconds_bucket[5m])) by (le))`

**Nguyen nhan:**
```
model_prediction_latency_seconds la Histogram.
Histogram chi co data sau khi observe() duoc goi.
observe() duoc goi moi lan POST /predict thanh cong.

Neu chua gui prediction nao → bucket rong
→ histogram_quantile() tra ve NaN
→ Grafana hien "No data"

Fix: Chay simulate_predictions.py de tao traffic
```

### Panel: Prediction Error Rate

**PromQL:** `(sum(rate(model_prediction_errors_total[5m])) or vector(0)) / ...`

**Nguyen nhan:**
```
model_prediction_errors_total la Counter.
Counter chi xuat hien trong Prometheus sau lan .inc() dau tien.
.inc() duoc goi khi co loi (400, 500, model not loaded...).

Neu chua co loi nao → Counter chua ton tai trong Prometheus
→ rate() tra ve "No data" (khac voi 0)

Sau khi fix dashboard voi "or vector(0)":
→ Se hien 0% thay vi "No data"
→ 0% = tot (khong co loi)
```

### Panel: Model Version

**PromQL:** `topk(1, model_version_info)`

**Nguyen nhan:**
```
model_version_info la Gauge.
Gauge duoc set trong ham load_model() cua API.

Neu model load that bai (khong co Production model trong MLflow)
→ .set() chua bao gio duoc goi
→ Gauge chua ton tai trong Prometheus
→ Grafana hien "No data"

Kiem tra:
  curl http://localhost:8000/health
  → "model_loaded": false  → Can chay train_and_register.py truoc

Sau khi model load thanh cong:
  model_version_info{..., version="1"} = 1
  → Grafana hien "v1" (textMode="name" hien legendFormat)
```

### Quy trinh chuan fix "No data"

```bash
# Buoc 1: Dam bao model da duoc train va register
python scripts/train_and_register.py

# Buoc 2: Kiem tra API da load model chua
curl http://localhost:8000/health
# Can thay: "model_loaded": true

# Buoc 3: Kiem tra metric model_version_info da co
curl http://localhost:8000/metrics | grep model_version_info
# Can thay: model_version_info{...,version="1"} 1.0

# Buoc 4: Kiem tra Prometheus da scrape duoc
# Vao: http://localhost:9090/targets
# Can thay: credit-card-api → UP

# Buoc 5: Tao prediction traffic
python scripts/simulate_predictions.py --n 200

# Buoc 6: Doi 30-60 giay roi refresh Grafana
# → Tat ca panels se co data
```

---

## 7. Checklist Dam Bao Data Hien Thi Dung

### Truoc khi xem Grafana

- [ ] `docker compose ps` → tat ca services la `Up (healthy)`
- [ ] `curl http://localhost:8000/health` → `"model_loaded": true`
- [ ] `curl http://localhost:8001/health` → `"status": "healthy"`
- [ ] Vao `http://localhost:9090/targets` → tat ca targets la `UP`

### Sau khi tat ca dich vu chay

- [ ] Chay `python scripts/train_and_register.py` (neu chua co model)
- [ ] Chay `python scripts/simulate_predictions.py --upload-reference --n 200`
- [ ] Doi 30-60 giay (Prometheus can scrape it nhat 1-2 lan)
- [ ] Refresh Grafana dashboard

### Cac metrics se hien thi

| Panel | Dieu kien hien thi |
|-------|-------------------|
| Model Version | Model da load thanh cong (`model_loaded: true`) |
| Requests/5m | Co request den API (bat ky endpoint) |
| Latency p95 | Co request den POST /predict |
| Error Rate | Sau fix: hien 0% ngay ca khi chua co loi |
| Drift Status | Da chay POST /analyze va co >100 samples |
| Feature Drift | Da chay /analyze va co drift |

### Lenh debug huu ich

```bash
# Xem tat ca metrics dang duoc expose
curl http://localhost:8000/metrics | grep "^[^#]" | head -30

# Kiem tra 1 metric cu the trong Prometheus
# Vao: http://localhost:9090/graph
# Nhap: model_predictions_total
# → Neu co data = Prometheus dang scrape OK

# Xem log API de tim nguyen nhan model load fail
docker compose logs api --tail=50 | grep -i "error\|model\|fail"

# Xem log Prometheus de kiem tra scrape
docker compose logs prometheus --tail=20
```
