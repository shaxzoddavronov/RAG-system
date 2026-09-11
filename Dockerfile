# Pinned to 3.11: the dev host runs 3.14, where several of these wheels do not
# build. The container is what guarantees the PM never meets that problem.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface

WORKDIR /srv

# CPU-only torch by default: keeps the image near 1 GB instead of ~3.5 GB, and
# the GPU work that matters most (token generation) happens in Ollama on the
# host regardless. docker-compose.gpu.yml overrides this with a CUDA index so a
# machine with room to spare can also rerank on the GPU. See "Rerank on the GPU"
# in RUNNING.md.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.9.1 --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/corpus.jsonl ./data/corpus.jsonl
COPY index/ ./index/
COPY eval/ ./eval/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
