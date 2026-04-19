# Hướng dẫn Retrain khi Data Drift

## Mục lục
1. [Khi nào cần Retrain?](#1-khi-nào-cần-retrain)
2. [Quy trình tổng quan](#2-quy-trình-tổng-quan)
3. [Bước 1 – Phát hiện Drift](#3-bước-1--phát-hiện-drift)
4. [Bước 2 – Retrain tự động (1 lệnh)](#4-bước-2--retrain-tự-động-1-lệnh)
5. [Bước 3 – Xác nhận model mới](#5-bước-3--xác-nhận-model-mới)
6. [Rollback nếu model mới kém hơn](#6-rollback-nếu-model-mới-kém-hơn)
7. [Lịch retrain định kỳ](#7-lịch-retrain-định-kỳ)
8. [Tất cả tùy chọn CLI](#8-tất-cả-tùy-chọn-cli)

---

## 1. Khi nào cần Retrain?

| Tín hiệu | Ngưỡng | Hành động |
|----------|--------|-----------|
| `drift_score` > 10% | Mặc định `0.10` | Retrain ngay |
| `drift_score` > 30% | Drift nghiêm trọng | Retrain khẩn cấp + điều tra data |
| AUC production giảm > 5% | So với validation gốc | Retrain + review features |
| Hơn 3 features drift cùng lúc | `drifted_count >= 3` | Retrain + kiểm tra pipeline ETL |
| Sau 30 ngày | Định kỳ | Retrain phòng ngừa |

```
drift_score = số features bị drift / tổng features
Ví dụ: 5/28 features drift → drift_score = 0.178 → VƯỢt ngưỡng 0.10 → RETRAIN
```

---

## 2. Quy trình tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                    RETRAIN-ON-DRIFT WORKFLOW                    │
│                                                                 │
│  Production traffic                                             │
│       │                                                         │
│       ▼ POST /capture                                           │
│  Evidently (buffer 500+ samples)                                │
│       │                                                         │
│       ▼ POST /analyze                                           │
│  Drift Analysis                                                 │
│       │                                                         │
│   drift?                                                        │
│   ├── NO  → Không làm gì, tiếp tục monitor                     │
│   └── YES ─────────────────────────────────────────────────►   │
│                                                                 │
│       ▼                                                         │
│  train_and_register.py                                          │
│  (train LightGBM mới, log MLflow, promote Production)          │
│       │                                                         │
│       ▼ POST /model/reload                                      │
│  API load model version mới                                     │
│       │                                                         │
│       ▼                                                         │
│  Xác nhận AUC mới ≥ AUC cũ → Done                              │
│  Nếu AUC giảm → Rollback version cũ                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Bước 1 – Phát hiện Drift

### Kiểm tra thủ công

```bash
# Xem trạng thái Evidently
curl.exe http://localhost:8001/health

# Kết quả mong đợi:
# {
#   "reference_data_loaded": true,
#   "production_data_count": 523,   ← cần >= 100
#   "last_analysis": "2026-04-19T..."
# }
```

```bash
# Chạy phân tích drift
curl.exe -X POST http://localhost:8001/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"window_size\": 500}"

# Kết quả mẫu KHI CÓ DRIFT:
# {
#   "drift_detected": true,
#   "drift_score": 0.357,
#   "drifted_features": ["age", "avg_casa_this_m", "spend_velocity"],
#   "report_url": "/reports/drift_report_20260419_112000.html"
# }
```

```bash
# Xem báo cáo HTML chi tiết
# Mở trình duyệt: http://localhost:8001/reports
```

### Kiểm tra từ Grafana

Mở Grafana (`http://localhost:3000`), xem panel **"Data Drift Score"**:
- **Xanh lá** (< 0.10): ổn định, không cần retrain
- **Vàng** (0.10 – 0.30): drift nhẹ, theo dõi thêm
- **Đỏ** (> 0.30): drift nghiêm trọng, retrain ngay

---

## 4. Bước 2 – Retrain tự động (1 lệnh)

### Chạy script tự động

```bash
python scripts/retrain_on_drift.py
```

Script tự động thực hiện:
1. Kiểm tra Evidently health
2. Gọi `/analyze` để lấy drift score
3. Nếu `drift_detected=True` → chạy `train_and_register.py`
4. Sau khi train xong → gọi `POST /model/reload` để API dùng model mới
5. In kết quả (version mới, AUC)

**Output mẫu khi có drift:**
```
2026-04-19 14:30:00 - INFO - Checking Evidently health...
2026-04-19 14:30:01 - INFO - Evidently | reference_loaded=True | production_samples=523
2026-04-19 14:30:01 - INFO - Running drift analysis (window=500)...
2026-04-19 14:30:03 - INFO - ═══════════════════════════════════════════════════════
2026-04-19 14:30:03 - INFO - DRIFT ANALYSIS RESULT
2026-04-19 14:30:03 - INFO -   drift_detected : True
2026-04-19 14:30:03 - INFO -   drift_score    : 0.3571  (threshold=0.1000)
2026-04-19 14:30:03 - INFO -   drifted_count  : 10 / 28 features
2026-04-19 14:30:03 - INFO -   drifted_features: ['age', 'avg_casa_this_m', ...]
2026-04-19 14:30:03 - INFO - Drift detected! Triggering retrain...
...
2026-04-19 14:35:21 - INFO - ═══════════════════════════════════════════════════════
2026-04-19 14:35:21 - INFO - RETRAIN COMPLETE
2026-04-19 14:35:21 - INFO -   New model version : 4
2026-04-19 14:35:21 - INFO -   API status        : healthy
2026-04-19 14:35:21 - INFO - ═══════════════════════════════════════════════════════
```

**Output mẫu khi KHÔNG có drift:**
```
2026-04-19 14:30:03 - INFO - drift_score: 0.0357 (threshold=0.1000)
2026-04-19 14:30:03 - INFO - No drift detected. No retraining needed.
```

---

## 5. Bước 3 – Xác nhận model mới

```bash
# 1. Kiểm tra version mới đã load
curl.exe http://localhost:8000/health
# → "model_version": "4"  ← phải tăng so với trước

# 2. Kiểm tra MLflow Registry
# Mở: http://localhost:5000 → Models → credit_card_propensity
# Version mới nhất phải ở stage "Production"

# 3. So sánh metrics (MLflow UI)
# Mở: http://localhost:5000/#/experiments/1
# So sánh valid_auc, valid_ks của run mới vs run cũ

# 4. Gửi vài request test thử
python scripts/simulate_predictions.py --n 20
# → success=20 errors=0
```

### Checklist sau retrain

- [ ] `model_version` trong `/health` đã tăng
- [ ] MLflow Registry: version mới ở stage `Production`
- [ ] AUC mới ≥ AUC cũ (so sánh trong MLflow Experiments)
- [ ] 20 test predictions không lỗi
- [ ] Grafana panel `Model Version` đã cập nhật

---

## 6. Rollback nếu model mới kém hơn

Nếu AUC model mới **thấp hơn** model cũ, rollback:

```bash
python -c "
import mlflow
client = mlflow.tracking.MlflowClient('http://localhost:5000')
versions = client.search_model_versions(\"name='credit_card_propensity'\")
for v in sorted(versions, key=lambda x: int(x.version)):
    print(f'Version {v.version}: stage={v.current_stage}, run_id={v.run_id[:8]}')
"
```

Sau khi xác định version cũ muốn khôi phục (ví dụ version 3):

```bash
python -c "
import mlflow
client = mlflow.tracking.MlflowClient('http://localhost:5000')
# Promote version cũ lại Production
client.transition_model_version_stage(
    name='credit_card_propensity',
    version='3',
    stage='Production',
    archive_existing_versions=True,
)
print('Rolled back to version 3')
"

# Reload API
curl.exe -X POST http://localhost:8000/model/reload
curl.exe http://localhost:8000/health
# → "model_version": "3"
```

---

## 7. Lịch retrain định kỳ

Thêm vào crontab (Linux/macOS) hoặc Task Scheduler (Windows) để chạy hàng ngày:

```bash
# Chạy mỗi ngày lúc 2:00 sáng (Linux crontab)
0 2 * * * cd /path/to/credit-card-mlops && python scripts/retrain_on_drift.py >> logs/retrain.log 2>&1
```

Hoặc thêm vào GitHub Actions (chạy theo lịch hàng tuần):

```yaml
# Thêm vào .github/workflows/ci_cd.yml
on:
  schedule:
    - cron: '0 2 * * 1'  # Mỗi thứ Hai lúc 2:00 UTC
  workflow_dispatch:      # Cho phép trigger thủ công từ GitHub UI
```

---

## 8. Tất cả tùy chọn CLI

```bash
python scripts/retrain_on_drift.py --help

usage: retrain_on_drift.py [-h] [--force] [--dry-run] [--window WINDOW] [--threshold THRESHOLD]

options:
  --force               Retrain ngay dù không có drift (dùng để test)
  --dry-run             Chỉ phân tích drift, KHÔNG retrain (kiểm tra an toàn)
  --window WINDOW       Số production samples dùng để phân tích (default: 500)
  --threshold THRESHOLD Ngưỡng drift_score để trigger retrain (default: 0.10)
```

### Ví dụ sử dụng

```bash
# Kiểm tra drift mà không retrain
python scripts/retrain_on_drift.py --dry-run

# Hạ ngưỡng xuống 5% (nhạy hơn)
python scripts/retrain_on_drift.py --threshold 0.05

# Retrain bắt buộc dù không có drift (demo / test pipeline)
python scripts/retrain_on_drift.py --force

# Phân tích 1000 samples gần nhất, ngưỡng 15%
python scripts/retrain_on_drift.py --window 1000 --threshold 0.15
```

### Exit codes

| Code | Ý nghĩa |
|------|---------|
| `0` | Không có drift **hoặc** retrain thành công |
| `1` | Lỗi (Evidently không reach được, train fail, reload fail) |
