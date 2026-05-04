FROM python:3.10-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default model path; mount ./models to /app/models when running
ENV MODEL_PATH=/app/models/knn_final.joblib
ENV APP_HOST=0.0.0.0
ENV APP_PORT=5000

EXPOSE 5000

CMD ["python", "-m", "src.app"]
