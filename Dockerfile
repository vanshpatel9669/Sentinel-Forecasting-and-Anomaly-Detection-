FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# The trained forecaster is committed to the repo (models/) so the API is
# runnable out of the box; it is not retrained at build time. To retrain,
# run scripts/generate_synthetic_series.py then scripts/train_forecaster.py.
COPY models/ ./models/

EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8020/health')" || exit 1

CMD ["uvicorn", "sentinel.api.main:app", "--host", "0.0.0.0", "--port", "8020"]
