# CPU + Silero-VAD diagnostic image. CPU-only by design (no CUDA).
FROM python:3.11-slim

# OS-level benchmark/topology tools:
#   sysbench  -> CPU throughput number (events/sec)
#   util-linux -> lscpu (core count + cache + flags)
#   hwloc      -> lstopo-no-graphics (NUMA/cache topology)
#   curl       -> fetch the Silero model at build time
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        sysbench \
        util-linux \
        hwloc \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch ONLY the Silero VAD ONNX model (2.3 MB), pinned to a release tag.
# This avoids the silero-vad pip package, which would drag in torch + CUDA.
ARG SILERO_TAG=v5.1.2
RUN mkdir -p /srv/models \
    && curl -fsSL -o /srv/models/silero_vad.onnx \
       "https://raw.githubusercontent.com/snakers4/silero-vad/${SILERO_TAG}/src/silero_vad/data/silero_vad.onnx"
ENV SILERO_ONNX_PATH=/srv/models/silero_vad.onnx

COPY server.py bench.py /srv/app/

# Persisted run history lives here; mount a volume to keep it across restarts.
ENV DATA_DIR=/data
RUN mkdir -p /data && useradd -m -u 10001 diag && chown -R diag:diag /data /srv
USER diag

# Keep VAD per-stream behaviour predictable; override at runtime if you want.
ENV OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

WORKDIR /srv/app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]