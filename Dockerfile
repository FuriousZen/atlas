FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for psycopg, pdf text extraction, and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libpq-dev build-essential poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
# Keep torch CPU handling portable: if you build with CUDA, override in compose
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu || true \
    && pip install -e . || pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
