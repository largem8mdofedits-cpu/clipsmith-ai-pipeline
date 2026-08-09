# Clipsmith AI pipeline (cloud version)

Downloads a video, transcribes it via **Deepgram** (hosted, no GPU needed),
asks **Claude** to pick the best highlight moments from the transcript,
cuts clips with ffmpeg, reframes to 9:16, and burns in **animated
karaoke-style captions** — words highlight from white to yellow exactly as
they're spoken.

This replaced the earlier local-GPU-only version. No NVIDIA card, no CUDA,
no local Whisper model required — it's a plain CPU service now, so it can
run on Railway (or any container host) instead of your own machine.

## 1. Get your API keys

- **Deepgram** (transcription) — sign up at https://console.deepgram.com/,
  free tier includes $200 of credit. Copy your API key.
- **Anthropic** (highlight picking) — get a key at
  https://console.anthropic.com/. Optional but recommended: without it,
  the pipeline falls back to a simple "most words per window" heuristic
  instead of Claude picking actual hooks/punchlines/emotional beats.

## 2. Run it locally

```bash
cd clipsmith-ai-pipeline
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need on your PATH:
- **ffmpeg** — Windows: `winget install ffmpeg` / Mac: `brew install ffmpeg` / Linux: `sudo apt install ffmpeg`
- **Deno** — yt-dlp needs this to execute YouTube's player JS. Windows: `winget install DenoLand.Deno` / Mac: `brew install deno` / Linux: see deno.land

Set your keys and start the server:
```bash
export DEEPGRAM_API_KEY=your_key_here      # Windows: set DEEPGRAM_API_KEY=...
export ANTHROPIC_API_KEY=your_key_here
uvicorn pipeline:app --reload --port 8000
```

Test it's alive:
```bash
curl http://localhost:8000/health
```

Try a real clip:
```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=SOME_VIDEO_ID","clip_count":2,"clip_seconds":20}'
```

Finished clips land in `clipsmith-ai-pipeline/clips/`, and the response
gives you their filenames, timestamps, and (if Claude picked them) a short
reason each moment was chosen.

## 3. Deploy to Railway

This folder includes a `Dockerfile`, so Railway can build and run it
directly:

1. Push `clipsmith-ai-pipeline/` to its own GitHub repo (or a subfolder of
   your main repo — Railway lets you set a root directory per service).
2. In Railway: New Project → Deploy from GitHub repo → pick this repo/folder.
   Railway will detect the `Dockerfile` and build it automatically.
3. Under Variables, add:
   - `DEEPGRAM_API_KEY`
   - `ANTHROPIC_API_KEY` (optional, but recommended)
4. Generate a public domain for the service (Settings → Networking →
   Generate Domain). You'll get a URL like `https://clipsmith-pipeline.up.railway.app`.
5. Update `frontend/config.js`:
   ```js
   window.CLIPSMITH_AI_PIPELINE_URL = 'https://clipsmith-pipeline.up.railway.app';
   ```

## How it works

1. **Download** — `yt-dlp` pulls the source video.
2. **Extract audio** — ffmpeg strips just the audio track (smaller/faster
   to upload than the whole video).
3. **Transcribe** — the audio goes to Deepgram's `nova-2` model, which
   returns word-level timestamps.
4. **Pick highlights** — the timestamped transcript goes to Claude, which
   picks the `clip_count` best moments (hooks, punchlines, emotional
   beats) rather than just the most talkative windows. If no
   `ANTHROPIC_API_KEY` is set, or the call fails for any reason, it falls
   back automatically to a speech-density heuristic — clip generation
   never hard-fails because of this step.
5. **Cut + caption** — for each highlight, ffmpeg cuts the window, crops
   and scales it to 1080x1920 (9:16), and burns in an `.ass` subtitle file
   with per-word karaoke (`\k`) tags. libass (built into ffmpeg) handles
   the color-sweep timing — no frame-by-frame image generation needed.

## Tuning

| Variable | What it controls |
|---|---|
| `DEEPGRAM_API_KEY` | Required for transcription. |
| `ANTHROPIC_API_KEY` | Optional. Enables smart highlight picking. |
| `ANTHROPIC_MODEL` | Defaults to `claude-sonnet-5`. |

Caption look (font, size, colors) is set in `words_to_ass()` in
`pipeline.py` — the `Style:` line follows standard ASS/SSA styling, so any
ASS reference covers what each field does.

## What this version does NOT do yet

- **No queueing** — if two requests come in at once, the second waits for
  the first to finish. Fine for testing solo; would need a job queue
  (e.g. Redis + a worker) before multiple real users could hit it at once.
- **CORS is wide open** (`allow_origins=["*"]`) — fine for testing, worth
  tightening to your real frontend domain before wide public use.
- **One caption style** — karaoke word-highlight only for now. A
  word-by-word pop-in style (CapCut/Crayo-style) would be a second
  `words_to_ass`-style function plus a way to pick between them per request.
