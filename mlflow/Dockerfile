FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.16.2 \
    psycopg2-binary \
    boto3 \
    gunicorn

WORKDIR /mlflow

EXPOSE 5000

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", \
     "mlflow server \
      --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB} \
      --default-artifact-root s3://${MINIO_BUCKET_NAME} \
      --host 0.0.0.0 \
      --port 5000"]
