# SAM - Dockerfile for Production Deployment
# Multi-stage build for optimized container size

# Build arguments
ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION=latest

# Stage 1: Build stage with all build dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --only-binary=all -r /tmp/requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Add metadata labels
LABEL org.opencontainers.image.title="SAM - Secure AI Memory"
LABEL org.opencontainers.image.description="Advanced AI assistant with human-like conceptual understanding"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.source="https://github.com/forge-1825/SAM"
LABEL org.opencontainers.image.url="https://github.com/forge-1825/SAM"
LABEL org.opencontainers.image.documentation="https://github.com/forge-1825/SAM/blob/main/DOCKER_DEPLOYMENT_GUIDE.md"
LABEL org.opencontainers.image.vendor="Forge 1825"
LABEL org.opencontainers.image.licenses="MIT"
LABEL maintainer="vin@forge1825.net"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SAM_DOCKER=true \
    SAM_DATA_DIR=/app/data \
    SAM_LOGS_DIR=/app/logs \
    SAM_VERSION="${VERSION}" \
    SAM_BUILD_DATE="${BUILD_DATE}" \
    SAM_GIT_COMMIT="${GIT_COMMIT}"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user for security
RUN groupadd -r sam && useradd -r -g sam sam

# Create application directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/logs /app/memory_store /app/chroma_db /app/uploads \
    /app/cache /app/backups /app/config /app/security && \
    chown -R sam:sam /app

# Copy application code
COPY --chown=sam:sam . /app/

# Create Docker-specific configuration
COPY docker/sam_docker_config.json /app/config/sam_config.json
COPY docker/docker_entrypoint.sh /app/docker_entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/docker_entrypoint.sh

# Switch to non-root user
USER sam

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8502/health || exit 1

# Expose ports
EXPOSE 8502 8501 8503

# Set entrypoint
ENTRYPOINT ["/app/docker_entrypoint.sh"]

# Default command
CMD ["streamlit", "run", "secure_streamlit_app.py", "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true"]
