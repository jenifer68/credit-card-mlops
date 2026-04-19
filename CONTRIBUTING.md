python scripts/train_and_register.py

docker compose up -d api
curl.exe -X POST http://localhost:8000/model/reload

python scripts/simulate_predictions.py --upload-reference --n 200