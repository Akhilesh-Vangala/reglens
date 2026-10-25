# RegLens service: retrieval, verifier and demo UI work with no credentials; answering needs a model
# configured through environment variables (see .env.example). Secrets are never baked into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 PORT=8000 FASTEMBED_CACHE_PATH=/opt/models REGLENS_ROOT=/app \
    REGLENS_STATE_DIR=/tmp/reglens

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,stores]"

COPY data ./data
COPY evals ./evals
# Embed the pinned corpus once at build time so the container starts warm and offline.
RUN python -c "from reglens.eval.run import make_retriever; from reglens.retrieval import load_chunks; \
from reglens.config import ROOT; make_retriever('dense_bge', load_chunks(ROOT/'data/processed/chunks.jsonl'))"

RUN useradd --create-home --uid 10001 app && chown -R app /app /opt/models
# weights are baked in above; never reach the network (or write hub caches) at runtime
ENV HF_HUB_OFFLINE=1 HF_HOME=/tmp/hf
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["reglens-serve"]
