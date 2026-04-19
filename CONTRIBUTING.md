python scripts/train_and_register.py

docker compose up -d api
python scripts/simulate_predictions.py --upload-reference --n 200