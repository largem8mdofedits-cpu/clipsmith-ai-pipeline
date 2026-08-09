# Clipsmith AI pipeline — cloud version. No GPU/CUDA needed: transcription
# runs on Deepgram, highlight picking on Claude, so this is a plain CPU
# container. Fine for Railway's standard (non-GPU) service tier.
FROM python:3.11-slim

# ffmpeg          — cutting, reframing, caption burn-in
# fonts-liberation — Arial-compatible font so captions render correctly
#                    (the container has no fonts installed by default)
# curl, unzip, ca-certificates — needed to install Deno below
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Deno — yt-dlp needs this to execute YouTube's player JS and derive
# working download URLs (see pipeline.py's download_video for why).
RUN curl -fsSL https://deno.land/install.sh | sh -s -- -y \
    && ln -s /root/.deno/bin/deno /usr/local/bin/deno

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway sets $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn pipeline:app --host 0.0.0.0 --port ${PORT}"]
