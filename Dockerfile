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
    fonts-dejavu-core \
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

# Piper TTS voice models — self-hosted, zero-API-key fallback for
# voice-over (see synthesize_voiceover() in pipeline.py). Pinned to the
# v1.0.0 tag of rhasspy/piper-voices on Hugging Face rather than `main` so
# a rebuild can't silently pull a renamed/moved file. ~50-70MB each; six
# voices keeps the image growth reasonable while still giving the frontend
# voice picker real variety. Placed before COPY . . so editing pipeline.py
# doesn't bust this layer's cache and re-download ~300MB every build.
RUN mkdir -p /app/piper_voices && \
    PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US" && \
    for v in amy ryan kristin joe ljspeech john; do \
      curl -fsSL "$PIPER_BASE/$v/medium/en_US-$v-medium.onnx" -o "/app/piper_voices/en_US-$v-medium.onnx" && \
      curl -fsSL "$PIPER_BASE/$v/medium/en_US-$v-medium.onnx.json" -o "/app/piper_voices/en_US-$v-medium.onnx.json"; \
    done

COPY . .

# Re-pull the newest yt-dlp release on every build. requirements.txt pins
# no version, but Docker still caches the `pip install -r requirements.txt`
# layer above whenever requirements.txt itself hasn't changed — so without
# this, a rebuild can silently keep running a yt-dlp that's days or weeks
# stale. YouTube tweaks its anti-bot checks often enough that this alone
# was a real cause of "Sign in to confirm you're not a bot" failures.
# Placed after COPY . . so it reruns on every code change (this project's
# pipeline.py changes often, which conveniently busts the cache here too).
RUN pip install --no-cache-dir -U yt-dlp

# Railway sets $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn pipeline:app --host 0.0.0.0 --port ${PORT}"]
