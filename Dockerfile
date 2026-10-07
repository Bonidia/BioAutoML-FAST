# Fixed interpreter and installer; Python packages are locked in uv.lock.
FROM ghcr.io/astral-sh/uv:0.7.3@sha256:87a04222b228501907f487b338ca6fc1514a93369bfce6930eb06c8d576e58a4 AS uv
FROM python:3.11.12-slim-bookworm@sha256:dbf1de478a55d6763afaa39c2f3d7b54b25230614980276de5cacdde79529d0c AS base

# Resolve native MMseqs2 and its libraries from the same lock as local Pixi runs.
FROM ghcr.io/prefix-dev/pixi:0.81.0@sha256:788ae451641666e2d1f79d3dbe35392dfc7e9b394b16a3acb75c347f3badb2ab AS native
WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install --locked

# 1. Download MathFeature without shipping Git in the runtime image.
FROM base AS mathfeature
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /export
RUN git clone --depth 1 https://github.com/Bonidia/MathFeature.git MathFeature \
    && git -C MathFeature rev-parse HEAD > mathfeature-commit \
    && rm -rf /export/MathFeature/.git

# 2. System dependencies and non-root user.
FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server libgomp1 tini util-linux ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY --from=native /app/.pixi/envs/default /app/.pixi/envs/default
RUN ln -s /app/.pixi/envs/default/bin/mmseqs /usr/local/bin/mmseqs \
    && mmseqs version

RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --groups 0 --create-home appuser

WORKDIR /app

# 3. Locked Python environment, separate from application files.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/tmp/bioautoml-cache \
    MPLCONFIGDIR=/tmp/bioautoml-matplotlib \
    NUMBA_CACHE_DIR=/tmp/bioautoml-numba \
    TASK_RESULTS_DB=/app/App/task-results/task_results.db

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project --no-cache \
    --python /usr/local/bin/python

# 4. Application files, writable directories, and environment records.
COPY --chown=appuser:appuser . .
COPY --from=mathfeature --chown=appuser:appuser /export/MathFeature/ /app/MathFeature/
COPY --from=mathfeature /export/mathfeature-commit /opt/bioautoml/mathfeature-commit

RUN test -f MathFeature/methods/ExtractionTechniques.py \
    && chmod +x start.sh \
    && install -d -o appuser -g 0 -m 2775 App/task-results App/task-results/redis App/jobs \
    && install -d -o appuser -g appuser -m 755 App/datasets \
    && dpkg-query -W > /opt/bioautoml/system-packages.tsv \
    && /opt/venv/bin/python scripts/environment_report.py > /opt/bioautoml/environment.json

# 5. Startup and health checks; start.sh supervises all three services.
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD ["python", "/app/scripts/container_healthcheck.py"]

# Tini reaps orphaned subprocesses.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/start.sh"]
