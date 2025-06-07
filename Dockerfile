# syntax=docker/dockerfile:1.7

# ---- Builder ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl &&
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
ADD . /app

# Use cache mounts for uv, install only production dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --no-dev --frozen

# ---- Final ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
  curl &&
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the installed environment and app code from builder
COPY --from=builder /app /app
COPY --from=builder /app/.venv /app/.venv

EXPOSE 9483
ENV BREEZE_HOST="0.0.0.0"
ENV BREEZE_DATA_ROOT="/data"

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:9483/health || exit 1

ENTRYPOINT ["uv", "run", "--no-dev", "python", "-m", "breeze"]
CMD ["serve"]
