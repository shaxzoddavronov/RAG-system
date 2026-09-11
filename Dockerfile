# Pinned to 3.11: the dev host runs 3.14, where several of these wheels do not
# build. The container is what guarantees the PM never meets that problem.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface

WORKDIR /srv

# CPU-only torch. Keeps the image near 1 GB instead of 3 GB; the GPU work that
# matters (token generation) happens in Ollama on the host, not in here.
RUN pip install --no-cache-dir torch==2.9.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/corpus.jsonl ./data/corpus.jsonl
COPY index/ ./index/
COPY eval/ ./eval/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
