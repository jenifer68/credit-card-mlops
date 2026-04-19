# Data Drift – Giải thích chi tiết (Evidently trong dự án)

## Mục lục
1. [Data Drift là gì?](#1-data-drift-là-gì)
2. [Tại sao Drift gây hại cho mô hình ML?](#2-tại-sao-drift-gây-hại-cho-mô-hình-ml)
3. [Evidently là gì?](#3-evidently-là-gì)
4. [Kiến trúc Drift Detection trong dự án](#4-kiến-trúc-drift-detection-trong-dự-án)
5. [Luồng dữ liệu (Data Flow)](#5-luồng-dữ-liệu-data-flow)
6. [Reference Data vs Production Data](#6-reference-data-vs-production-data)
7. [Các endpoint của Evidently Service](#7-các-endpoint-của-evidently-service)
8. [Phương pháp phát hiện Drift](#8-phương-pháp-phát-hiện-drift)
9. [Prometheus Metrics từ Evidently](#9-prometheus-metrics-từ-evidently)
10. [Đọc hiểu kết quả Drift](#10-đọc-hiểu-kết-quả-drift)
11. [Simulate Drift trong dự án](#11-simulate-drift-trong-dự-án)
12. [Checklist debug "No drift data"](#12-checklist-debug-no-drift-data)

---

## 1. Data Drift là gì?

**Data Drift** xảy ra khi phân phối thống kê của dữ liệu đầu vào **thực tế** thay đổi so với dữ liệu mà mô hình được huấn luyện.

```
Training time:  age ~ N(35, 10)   ← phân phối gốc
Production:     age ~ N(50, 8)    ← phân phối đã thay đổi → DRIFT!
```

Có 3 loại drift chính:

| Loại | Định nghĩa | Ví dụ trong dự án |
|------|-----------|-------------------|
| **Feature Drift** (Covariate Shift) | Phân phối X thay đổi | `avg_casa_this_m` tăng mạnh vì lạm phát |
| **Label Drift** (Prior Shift) | Phân phối Y thay đổi | Tỉ lệ khách mở thẻ tín dụng tăng mùa cuối năm |
| **Concept Drift** | Mối quan hệ X→Y thay đổi | Feature quan trọng trước đây không còn ý nghĩa nữa |

Dự án này tập trung vào **Feature Drift** — tức là 33 features đầu vào.

---

## 2. Tại sao Drift gây hại cho mô hình ML?

Mô hình `credit_card_propensity` được train trên dữ liệu lịch sử. Nếu khách hàng thực tế khác biệt với tập train:

```
Mô hình huấn luyện trên data năm 2023
→ Dùng để predict năm 2025
→ Hành vi khách hàng, lạm phát, thu nhập đã thay đổi
→ Model đưa ra dự đoán kém chính xác dù AUC từng tốt
```

**Hậu quả:**
- AUC trên production thấp hơn validation
- Propensity score bị lệch → campaign sai đối tượng
- Mô hình cần retrain nhưng không ai biết

Evidently giúp **phát hiện sớm** khi nào cần retrain.

---

## 3. Evidently là gì?

[Evidently AI](https://evidentlyai.com) là thư viện open-source để **monitor và đánh giá ML models** trong production.

Trong dự án này, Evidently chạy như một **microservice độc lập** (port `8001`) với 2 chức năng chính:

- **Data Drift Detection**: So sánh phân phối features của production vs training
- **Data Quality Monitoring**: Theo dõi missing values, outliers, thay đổi kiểu dữ liệu

Evidently **không** phải là database hay dashboard — nó là **engine phân tích thống kê**.

---

## 4. Kiến trúc Drift Detection trong dự án

```
┌─────────────────────────────────────────────────────────────┐
│                      LUỒNG DỮ LIỆU                          │
│                                                             │
│  simulate_predictions.py                                    │
│       │                                                     │
│       │ POST /predict                  POST /capture        │
│       ├──────────────────► API :8000 ──────────────────►   │
│       │                    (FastAPI)                        │
│       │                                                     │
│       │              Evidently Service :8001                │
│       │         ┌────────────────────────────┐              │
│       │         │  reference_data (training) │              │
│       │         │  production_data (online)  │              │
│       │         │                            │              │
│       └────────►│  POST /analyze             │              │
│                 │   → so sánh 2 tập          │              │
│                 │   → tính drift score       │              │
│                 │   → lưu HTML report        │              │
│                 │   → expose Prometheus      │              │
│                 └──────────────┬─────────────┘              │
│                                │                            │
│                    Prometheus :9090 (scrape mỗi 30s)        │
│                                │                            │
│                      Grafana :3000 (visualize)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Luồng dữ liệu (Data Flow)

### Bước 1 — Upload Reference Data

```bash
python scripts/simulate_predictions.py --upload-reference
```

Script lấy 500 dòng từ `card_valid_fe.csv` (tập validation khi train) và gửi tới:
```
POST http://localhost:8001/reference
Body: { "data": [...], "feature_names": [...] }
```

Evidently lưu vào `/app/reference/reference_data.csv` bên trong container.  
Đây là **"baseline"** — phân phối chuẩn để so sánh.

### Bước 2 — Capture Production Data

Mỗi lần script gọi `/predict` thành công, nó gửi song song tới:
```
POST http://localhost:8001/capture
Body: {
  "features": { "age": 35, "gender": 1, ... },
  "propensity_score": 0.72,
  "model_version": "2"
}
```

Evidently lưu từng bản ghi vào bộ nhớ (tối đa 20,000 bản ghi).

### Bước 3 — Chạy phân tích Drift

Khi đủ 100+ samples, script tự động gọi:
```
POST http://localhost:8001/analyze
Body: { "window_size": 200 }
```

Evidently so sánh 200 bản ghi production gần nhất với reference data, trả về:
```json
{
  "drift_detected": false,
  "drift_score": 0.03,
  "drifted_features": [],
  "total_features": 28,
  "report_url": "/reports/drift_report_20260419_112000.html"
}
```

### Bước 4 — Expose Prometheus Metrics

Prometheus scrape `http://evidently:8001/metrics` mỗi **30 giây**.  
Grafana đọc từ Prometheus → hiển thị panel drift trên dashboard.

---

## 6. Reference Data vs Production Data

| | Reference Data | Production Data |
|---|---|---|
| **Nguồn** | `card_valid_fe.csv` (khi train) | Request thực tế từ `/predict` |
| **Upload lúc nào** | Một lần, trước khi simulate | Liên tục theo mỗi request |
| **Mục đích** | Baseline phân phối "chuẩn" | Phân phối thực tế cần so sánh |
| **Kích thước** | 500 dòng (sample từ validation set) | Tối đa 20,000 bản ghi |
| **Lưu ở đâu** | `/app/reference/reference_data.csv` | RAM (in-memory list) |

> **Lưu ý**: Reference data bị **mất** khi restart container Evidently (không có persistent volume). Cần upload lại mỗi lần restart bằng `--upload-reference`.

---

## 7. Các endpoint của Evidently Service

| Endpoint | Method | Chức năng |
|----------|--------|-----------|
| `GET /health` | GET | Trạng thái service, số lượng data đã capture |
| `POST /reference` | POST | Upload reference data (data training) |
| `GET /reference` | GET | Xem thông tin reference data hiện tại |
| `POST /capture` | POST | Lưu 1 prediction vào production buffer |
| `POST /capture/batch` | POST | Lưu nhiều predictions cùng lúc |
| `POST /analyze` | POST | Chạy phân tích drift |
| `GET /reports` | GET | Danh sách báo cáo HTML đã tạo |
| `GET /reports/{name}` | GET | Xem báo cáo drift dạng HTML |
| `DELETE /production-data` | DELETE | Xóa toàn bộ production data buffer |
| `GET /metrics` | GET | Prometheus metrics |

---

## 8. Phương pháp phát hiện Drift

Evidently dùng **kiểm định thống kê** để phát hiện drift trên từng feature:

| Loại feature | Test được dùng | Ngưỡng mặc định |
|-------------|---------------|-----------------|
| Numerical (liên tục) | **Wasserstein Distance** hoặc **KS test** | p-value < 0.05 |
| Categorical (nhị phân) | **Chi-Square test** | p-value < 0.05 |

Sau khi kiểm định từng feature:
- `drift_score` = **tỷ lệ features bị drift** (ví dụ: 5/28 = 0.18 = 18%)
- `drift_detected = True` nếu `drift_score > DRIFT_THRESHOLD` (mặc định `0.1` = 10%)

Trong dự án, các features được chia:
- **Numerical** (25 features): `age`, `tenure_to_bank`, `avg_casa_this_m`, các chỉ số tài chính...
- **Binary/Categorical** (5 features): `gender`, `eligible_by_age`, `high_spend_flag`, `low_balance_flag`, `active_txn_flag`

---

## 9. Prometheus Metrics từ Evidently

Sau mỗi lần chạy `/analyze`, Evidently cập nhật các metrics:

| Metric | Loại | Ý nghĩa |
|--------|------|---------|
| `evidently_data_drift_detected` | Gauge | `1` = có drift, `0` = không drift |
| `evidently_drift_score` | Gauge | Tỷ lệ features bị drift (0.0 – 1.0) |
| `evidently_drifted_features_count` | Gauge | Số feature bị drift |
| `evidently_feature_drift{feature_name}` | Gauge | Drift của từng feature riêng lẻ |
| `evidently_missing_values_ratio{feature_name}` | Gauge | Tỷ lệ missing của từng feature |
| `evidently_captures_total` | Counter | Tổng số bản ghi đã capture |
| `evidently_analysis_total` | Counter | Tổng số lần phân tích drift |
| `evidently_analysis_duration_seconds` | Histogram | Thời gian chạy phân tích |

Grafana panel **"Data Drift Score"** dùng query:
```promql
evidently_drift_score
```

---

## 10. Đọc hiểu kết quả Drift

### Kết quả JSON từ `/analyze`

```json
{
  "drift_detected": true,
  "drift_score": 0.35,
  "drifted_features": ["age", "avg_casa_this_m", "spend_velocity"],
  "drift_scores": {
    "age": 0.82,
    "avg_casa_this_m": 0.71,
    "spend_velocity": 0.63,
    "tenure_to_bank": 0.04
  },
  "total_features": 28,
  "drifted_count": 3,
  "reference_samples": 500,
  "current_samples": 200,
  "report_url": "/reports/drift_report_20260419_112000.html"
}
```

### Cách đọc:

| Giá trị | Ý nghĩa |
|---------|---------|
| `drift_detected: true` | Hơn 10% features bị drift → cần xem xét retrain |
| `drift_score: 0.35` | 35% features (≈10/28) có phân phối khác biệt đáng kể |
| `drifted_features: [...]` | Danh sách feature cụ thể bị drift → điều tra nguyên nhân |
| `drift_scores.age: 0.82` | Feature `age` drift mạnh (gần 1.0 = rất drift) |
| `drift_scores.tenure_to_bank: 0.04` | Feature này ổn định (gần 0.0 = không drift) |

### Báo cáo HTML

Truy cập `http://localhost:8001/reports/drift_report_TIMESTAMP.html` để xem:
- Biểu đồ phân phối của từng feature (reference vs current)
- Bảng thống kê tổng hợp
- Heatmap drift score

---

## 11. Simulate Drift trong dự án

Script `simulate_predictions.py` hỗ trợ 2 chế độ:

### Normal mode (không drift)

```bash
python scripts/simulate_predictions.py --n 200
```

Gửi data từ `card_valid_fe.csv` — **cùng phân phối** với reference → drift_score thấp.

### Drift mode (giả lập drift)

```bash
python scripts/simulate_predictions.py --n 200 --drift
```

Gửi data từ `card_test_fe.csv` với biến đổi nhân tạo:

```python
# Nhân 2.5 lần + thêm noise cho 10 features đầu
drifted[col] = drifted[col] * 2.5 + drifted[col].std() * np.random.randn(len(drifted))
```

→ Phân phối thay đổi mạnh → Evidently phát hiện drift → Grafana panel chuyển màu đỏ.

### Kết quả mong đợi:

| Chế độ | `drift_detected` | `drift_score` | Grafana |
|--------|----------------|---------------|---------|
| Normal | `false` | < 0.10 | Xanh lá |
| Drift | `true` | > 0.30 | Đỏ |

---

## 12. Checklist debug "No drift data"

Nếu panel Grafana "Data Drift Score" không có data:

```bash
# 1. Kiểm tra Evidently service
curl.exe http://localhost:8001/health

# Kết quả mong đợi:
# "reference_data_loaded": true
# "production_data_count": > 100

# 2. Nếu reference_data_loaded = false → upload lại
python scripts/simulate_predictions.py --upload-reference --n 200

# 3. Nếu production_data_count < 100 → cần gửi thêm requests
python scripts/simulate_predictions.py --n 200

# 4. Chạy drift analysis thủ công
curl.exe -X POST http://localhost:8001/analyze -H "Content-Type: application/json" -d "{\"window_size\": 100}"

# 5. Xem metrics Prometheus có data chưa
curl.exe http://localhost:8001/metrics | grep evidently_drift

# 6. Kiểm tra Prometheus có scrape thành công không
# Mở: http://localhost:9090/targets
# Tìm job "evidently" → Status phải là UP

# 7. Xem báo cáo HTML mới nhất
curl.exe http://localhost:8001/reports
```

### Các lỗi thường gặp:

| Triệu chứng | Nguyên nhân | Cách fix |
|------------|-------------|----------|
| `reference_data_loaded: false` | Chưa upload hoặc container restart | `--upload-reference` |
| `Need at least 100 samples` | Chưa đủ production data | Gửi thêm `--n 200` |
| `No common numeric feature columns` | Feature names khác nhau giữa ref và production | Kiểm tra `FEATURE_NAMES` khớp |
| Grafana không có data | Prometheus chưa scrape | Chờ 30s hoặc xem `/targets` |
| `drift_score = 0` mãi | Không khi nào chạy `/analyze` tự động | Script chỉ analyze khi `success >= 100` |
