# syntax=docker/dockerfile:1

# deps: a virtualenv at /opt/crag from uv.lock. Nothing is copied into this stage. uv is
# mounted from its image for each RUN, the lock and pyproject are bind-mounted for the
# dependency layer, so only a lock change rebuilds it, and the project install reads the
# build context through a throwaway read-write bind mount. uv's download cache is a
# BuildKit cache mount, never a layer.
FROM python:3.12-slim AS deps
ENV UV_PROJECT_ENVIRONMENT=/opt/crag \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /src
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.5,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.5,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,target=/src,rw \
    uv sync --frozen --no-dev --no-editable

# Service image: the virtualenv and the licence files, running as uid 10001. SciFact,
# policy.json and the ONNX models are not baked in; mount them at /data and /models, or set
# CRAG_VERIFIER_BACKEND=demo to serve the invented corpus that ships inside the package.
FROM python:3.12-slim
RUN useradd --uid 10001 --user-group --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin verifier \
    && install -d -o verifier -g verifier /data /models
COPY --from=deps /opt/crag /opt/crag
COPY LICENSE NOTICE /usr/share/doc/corrective-rag-claim-verifier/
ENV PATH=/opt/crag/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CRAG_VERIFIER_DATA_DIR=/data \
    HF_HOME=/models \
    CRAG_VERIFIER_BACKEND=onnx
USER 10001:10001
WORKDIR /data
EXPOSE 8000
# Generous start period: on SciFact the service loads three models and the dense index.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]
CMD ["crag-verifier", "serve", "--host", "0.0.0.0", "--port", "8000"]
