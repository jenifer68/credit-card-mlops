# MLOps: From Model to Productions – Dự đoán khách hàng tiềm năng mở thẻ tín dụng Ngân hàng

**DDM501 Final Project – FSB32-Minh Ha**

Hệ thống MLOps đầy đủ cho bài toán Dự đoán khách hàng tiềm năng mở thẻ tín dụng (Credit Card Propensity Scoring).

---

## Stack

| Layer | Technology |
|-------|-----------|
| ML Model | LightGBM (champion), XGBoost / LogReg (challengers) |
| Experiment Tracking | MLflow + MinIO + PostgreSQL |
| Model Serving | FastAPI |
| Drift Monitoring | Evidently AI |
| Metrics | Prometheus |
| Dashboards | Grafana |
| Orchestration | Docker Compose |
| CI/CD | GitHub Actions |

## Quick Start

```bash
cp .env.example .env
pip install -r scripts/requirements.txt
docker-compose up -d postgres minio minio-init mlflow
python scripts/train_and_register.py
docker-compose up -d api evidently prometheus grafana
python scripts/simulate_predictions.py --upload-reference --n 200
```

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/****** |
| MLflow | http://localhost:5000 | – |
| API Docs | http://localhost:8000/docs | – |
| Evidently | http://localhost:8001 | – |
| Prometheus | http://localhost:9090 | – |
| MinIO | http://localhost:9001 | minio/****** |

## Documentation

- **[SETUP.md](SETUP.md)** – Hướng dẫn setup và chạy đầy đủ (step-by-step)
- **[ARCHITECTURE.md](ARCHITECTURE.md)** – Kiến trúc hệ thống

## Model Performance (LightGBM Champion)

| Metric | Valid | Test |
|--------|-------|------|
| AUC | > 0.78 | > 0.77 |
| KS | > 0.42 | > 0.40 |
| Lift@10% | > 2.5x | > 2.4x |
