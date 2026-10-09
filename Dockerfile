# syntax=docker/dockerfile:1

# Build stage: resolve the locked dependencies and install the package into a venv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Runtime stage: the venv only; no build tools, no uv, non-root.
FROM python:3.12-slim
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data /models \
    && chown 10001:10001 /data /models
COPY --from=build --chown=10001:10001 /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CRAG_VERIFIER_DATA_DIR=/data \
    HF_HOME=/models \
    CRAG_VERIFIER_BACKEND=onnx
USER 10001
WORKDIR /home/app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]
CMD ["crag-verifier", "serve", "--host", "0.0.0.0", "--port", "8000"]
