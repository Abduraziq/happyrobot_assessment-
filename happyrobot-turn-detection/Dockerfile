FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serving.txt /app/requirements-serving.txt
RUN pip install --no-cache-dir -r /app/requirements-serving.txt

COPY scripts/extract_features.py /app/scripts/extract_features.py
COPY scripts/predict.py /app/scripts/predict.py
COPY scripts/serve.py /app/scripts/serve.py
COPY models/baseline/model_config.json /app/models/baseline/model_config.json
COPY models/baseline/selected_model.joblib /app/models/baseline/selected_model.joblib

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).read()" || exit 1

CMD ["uvicorn", "serve:app", "--app-dir", "/app/scripts", "--host", "0.0.0.0", "--port", "8000"]
