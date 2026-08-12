"""
Clipsmith AI pipeline — cloud version.

Downloads a video, transcribes it via Deepgram (no local GPU needed),
asks Claude to pick the best highlight moments from the transcript
(falling back to a speech-density heuristic if no key is set), cuts
clips with ffmpeg, reframes to 9:16, and burns in animated
karaoke-style captions (words highlight as they're spoken).

Run it with:
    uvicorn pipeline:app --reload --port 8000

Then POST a video URL to http://localhost:8000/process
See README.md in this folder for full setup instructions.
"""

import asyncio
import base64
import gc
import hashlib
import importlib.util
import os
import json
import random
import re
import shutil
import subprocess
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Self-hosted background remover (see /remove-background below). Imported
# once at module load — not per-request — so the ONNX model session is
# built a single time and reused, instead of paying that cost on every
# call. Guarded the same way Piper's voice files are: if the package or
# model isn't there (e.g. an older deploy that predates this feature),
# the whole service still starts up fine and just reports the tool as
# unavailable instead of crashing.
try:
    from rembg import new_session as _rembg_new_session, remove as rembg_remove
except Exception as e:
    print(f"rembg not available: {e}")
    rembg_remove = None
    _rembg_new_session = None

# The U^2-Net session itself is built lazily on first use, not at import —
# this container has a 1GB memory ceiling (Railway Hobby plan), and holding
# a ~200-300MB ONNX session resident for the life of the process ate into
# the headroom Demucs/DeepFilterNet need for their own (much bigger) spikes,
# even on requests that never touch background removal at all.
_rembg_session_cache = None
def get_rembg_session():
    global _rembg_session_cache
    if _rembg_session_cache is None and _rembg_new_session is not None:
        _rembg_session_cache = _rembg_new_session("u2net")
    return _rembg_session_cache

app = FastAPI(title="Clipsmith cloud pipeline")

# Allow your website to call this service directly. Tighten this to your
# real frontend domain once you're live — "*" is fine while testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).parent / "clips"
OUTPUT_DIR.mkdir(exist_ok=True)

# This container has a hard 1GB memory ceiling (Railway Hobby plan) — that
# ceiling is NOT independently raisable via the API, it's tied to the plan
# tier. Confirmed by an actual OOM kill in production logs (a bare "Killed"
# from the kernel) during a round of manual testing that hit several AI
# Tools back to back. Every tool on the /remove-*, /enhance-*, /generate-*,
# /synthesize-voice, and /download-social-video endpoints now shares this
# one lock, so at most one of them runs at a time per replica — none of
# them individually should approach 1GB, but several running concurrently
# (or overlapping with a real /process clip job) can stack past it. This
# trades some latency under concurrent load for not getting silently
# killed mid-request. If you outgrow this, the real fix is either
# upgrading the Railway plan (raises the per-replica ceiling) or splitting
# these tools into their own service with its own separate 1GB budget.
HEAVY_TASK_LOCK = asyncio.Lock()

# Gemini's free tier gives this whole SITE a shared per-minute request
# budget for image generation — it's not per-user, so two people generating
# within the same few seconds of each other can knock each other into a 429
# even when neither of them is individually over any limit. Rather than let
# that be a race (whoever's request lands first "wins", everyone else gets
# a raw error), every /generate-image call queues behind this lock and
# waits out a minimum spacing since the last call before firing — so
# concurrent users share the quota fairly (FIFO, evenly paced) instead of
# fighting over it. GEMINI_MIN_INTERVAL_SECONDS is conservative (10 req/min)
# since Google doesn't publish an exact number for this preview model.
GEMINI_RATE_LOCK = asyncio.Lock()
GEMINI_MIN_INTERVAL_SECONDS = float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", "6"))
_last_gemini_call_at = 0.0

async def _wait_for_gemini_turn():
    global _last_gemini_call_at
    async with GEMINI_RATE_LOCK:
        now = time.monotonic()
        wait = GEMINI_MIN_INTERVAL_SECONDS - (now - _last_gemini_call_at)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_gemini_call_at = time.monotonic()

# Serves finished clips at http://<host>/clips/<filename>.mp4 — without
# this, the URLs returned by /process below 404.
app.mount("/clips", StaticFiles(directory=str(OUTPUT_DIR)), name="clips")

# Each /process call gets a job directory here (source video + transcript +
# metadata) that OUTLIVES the request, instead of the old tempfile approach
# that deleted everything the instant the response was sent. This is what
# lets /reclip regenerate a different moment from the SAME video without
# re-downloading it — important both for speed and because repeat downloads
# were part of what triggered YouTube's rate limiting in the first place.
JOBS_DIR = Path(__file__).parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
JOB_TTL_SECONDS = 3 * 60 * 60  # old job folders are swept on a delay, not kept forever

# Scratch space for the standalone AI tools (background remover, vocal
# remover, speech enhancer) — each request gets its own subfolder here,
# deleted again once the response is built. Unlike JOBS_DIR these aren't
# meant to outlive the request (no reclip-style follow-up call needs them).
TOOLS_DIR = Path(__file__).parent / "tools_tmp"
TOOLS_DIR.mkdir(exist_ok=True)


def cleanup_old_jobs():
    """Deletes job folders older than JOB_TTL_SECONDS. Called at the start
    of /process so disk usage on Railway's (small, ephemeral) volume stays
    bounded without needing a separate cron job."""
    now = time.time()
    for job_dir in JOBS_DIR.iterdir():
        try:
            if job_dir.is_dir() and (now - job_dir.stat().st_mtime) > JOB_TTL_SECONDS:
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception as e:
            print(f"Job cleanup skipped {job_dir}: {e}")


# Shared cache of already-downloaded SOURCE videos, keyed by URL — separate
# from JOBS_DIR (which is per-request scratch space). If two different
# users clip the same YouTube video, or the same user hits "try a different
# moment" on a fresh job for a video already seen recently, this skips
# yt-dlp entirely instead of re-downloading (and, once a paid proxy tier
# exists — see download_video() below — re-paying for it). Doesn't survive
# a redeploy (this directory isn't on a persistent volume), but does
# survive for the container's whole uptime between deploys, which is where
# the real savings are: any video more than one person clips in that
# window only ever costs one download.
SOURCE_CACHE_DIR = Path(__file__).parent / "source_cache"
SOURCE_CACHE_DIR.mkdir(exist_ok=True)
SOURCE_CACHE_TTL_SECONDS = 24 * 60 * 60  # a day is plenty for "still trending" reuse
SOURCE_CACHE_MAX_ENTRIES = 40  # crude size cap for Railway's small ephemeral disk


def _source_cache_path(url: str) -> Path:
    key = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:20]
    return SOURCE_CACHE_DIR / f"{key}.mp4"


def _source_cache_get(url: str) -> Optional[Path]:
    cached = _source_cache_path(url)
    if not cached.exists():
        return None
    if (time.time() - cached.stat().st_mtime) > SOURCE_CACHE_TTL_SECONDS:
        cached.unlink(missing_ok=True)
        return None
    cached.touch()  # bump mtime — this entry was just reused, keep it around longer (crude LRU)
    return cached


def _source_cache_put(url: str, downloaded_path: Path):
    """Best-effort — caching a source video should never fail the request
    it's attached to, so any error here is swallowed after logging."""
    try:
        shutil.copyfile(downloaded_path, _source_cache_path(url))
        entries = sorted(SOURCE_CACHE_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        while len(entries) > SOURCE_CACHE_MAX_ENTRIES:
            oldest = entries.pop(0)
            oldest.unlink(missing_ok=True)
    except Exception as e:
        print(f"Source cache write skipped: {e}")


# ---------------------------------------------------------------------------
# Config — all via env vars so this runs the same locally and on Railway.
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
# Optional: raw contents of a YouTube cookies.txt (Netscape format), from a
# logged-in browser session. Paste it as a single Railway variable value —
# see download_video() below for why this helps with YouTube's bot checks.
# Note: some of the cookies YouTube issues (SIDCC, PSIDTS) are short-lived
# security tokens Google auto-rotates as normal session hygiene, so an
# exported snapshot can go stale within hours — cookies alone aren't a
# permanent fix, just a temporary boost, hence the POT provider below.
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
# Google Cloud Text-to-Speech — a plain API key (Cloud Console > APIs &
# Services > Credentials > Create API Key, then restrict it to the
# "Cloud Text-to-Speech API"), not a service-account JSON file. Simpler to
# hand to Railway as one env var, same pattern as every other key here.
# Free tier: 4M chars/mo (Standard voices) + 1M chars/mo (Neural2) forever —
# see synthesize_voiceover() below for why this is tried before ElevenLabs.
GOOGLE_TTS_API_KEY = os.environ.get("GOOGLE_TTS_API_KEY")

# Google AI Studio (Gemini) — same Google Cloud API-key pattern as TTS
# above, but a separate key since it's issued from aistudio.google.com,
# not the Cloud Console. Free tier as of this writing: several hundred
# image-generation requests/day, no card on file — see /generate-image.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Groq — OpenAI-compatible chat completions API, free tier with no card
# on file. Used for /brainstorm-ideas instead of Anthropic, same
# zero-Claude-dependency philosophy as the transcript-based voiceover
# script fallback above.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# D-ID — talking-avatar video generation. Free tier: ~5 min of video/month,
# no card on file. Unlike the tools above, there's no free/self-hosted
# substitute for this one (realistic lip-synced avatar video is a hard
# problem), so this genuinely depends on the account's D-ID quota holding
# up — see /generate-avatar-video below. Auth is HTTP Basic: D-ID hands
# you a key already shaped "API_USERNAME:API_PASSWORD"; the whole string
# gets base64-encoded for the Authorization header (standard HTTP Basic,
# not a raw/unencoded value despite how the docs table reads).
DID_API_KEY = os.environ.get("DID_API_KEY")

# This service's own public URL — needed because D-ID's /talks endpoint
# takes a source_url it fetches itself, not a raw file upload. An
# uploaded photo is saved to OUTPUT_DIR (already served at /clips) and
# referenced by its full public URL so D-ID's servers can reach it.
PIPELINE_PUBLIC_URL = os.environ.get("PIPELINE_PUBLIC_URL", "https://clipsmith-ai-pipeline-production.up.railway.app")

# URL of a self-hosted bgutil-ytdlp-pot-provider instance (see
# https://github.com/Brainicism/bgutil-ytdlp-pot-provider) — generates
# proof-of-origin tokens that help yt-dlp's traffic look legitimate to
# YouTube from a datacenter IP. Not a guaranteed bypass (YouTube's own docs
# say so), but a free, real improvement with no account/cookies required.
POT_PROVIDER_URL = os.environ.get("POT_PROVIDER_URL", "").rstrip("/")

# Two OPTIONAL proxy tiers, both off by default and only ever used as a
# fallback — see download_video() below for the full ordering. Nothing
# routes through either of these unless the free tier (PO token + cookies
# + client rotation, above) has already failed for a given video, so
# there's no bandwidth cost paid on requests that would have worked
# anyway. Standard yt-dlp --proxy URL format for both, e.g.
# "http://user:pass@host:port" or "socks5://host:port".
#   YTDLP_OWN_PROXY_URL  — anything you're running yourself (a home
#                          connection, a personal VPS, whatever) — free
#                          to you, so it's tried before paying for anything.
#   YTDLP_PAID_PROXY_URL — a commercial residential proxy (e.g. Decodo) —
#                          the true last resort, only reached if BOTH the
#                          free tier and your own proxy have failed.
YTDLP_OWN_PROXY_URL = os.environ.get("YTDLP_OWN_PROXY_URL", "").strip()
YTDLP_PAID_PROXY_URL = os.environ.get("YTDLP_PAID_PROXY_URL", "").strip()

# A handful of ElevenLabs' stable premade voice IDs, exposed under friendly
# names for the frontend's voice picker. (These IDs are ElevenLabs' own
# public premade voices, not anything tied to this account.)
ELEVENLABS_VOICES = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",   # calm, female, US
    "Adam":   "pNInz6obpgDQGcFmaJgB",   # deep, male, US
    "Bella":  "EXAVITQu4vr4xnSDxMaL",   # soft, female, US
    "Antoni": "ErXwobaYiN019PkySvjV",   # warm, male, US
    "Elli":   "MF3mGyEYCl7XYWbV9V6O",   # young, female, US
    "Josh":   "TxGEqnHWrfWFTfGW9XjX",   # casual, male, US
}
# Same friendly names, mapped onto Google's en-US Neural2 voices instead —
# keeps the /voices list and frontend voice picker identical no matter
# which provider actually ends up synthesizing the audio (see
# synthesize_voiceover() below). Google's Neural2-{A,D,I,J} are male,
# {C,E,F,G,H} are female.
GOOGLE_TTS_VOICES = {
    "Rachel": {"languageCode": "en-US", "name": "en-US-Neural2-F", "ssmlGender": "FEMALE"},
    "Adam":   {"languageCode": "en-US", "name": "en-US-Neural2-D", "ssmlGender": "MALE"},
    "Bella":  {"languageCode": "en-US", "name": "en-US-Neural2-C", "ssmlGender": "FEMALE"},
    "Antoni": {"languageCode": "en-US", "name": "en-US-Neural2-A", "ssmlGender": "MALE"},
    "Elli":   {"languageCode": "en-US", "name": "en-US-Neural2-G", "ssmlGender": "FEMALE"},
    "Josh":   {"languageCode": "en-US", "name": "en-US-Neural2-J", "ssmlGender": "MALE"},
}
# Same friendly names again, mapped onto self-hosted Piper voice models
# (downloaded into /app/piper_voices at build time — see Dockerfile). No
# API key, no external network call, no per-character cost, ever — the
# always-available last resort in synthesize_voiceover()'s fallback chain.
PIPER_VOICES_DIR = Path(__file__).parent / "piper_voices"
PIPER_VOICES = {
    "Rachel": "en_US-amy-medium",       # calm, female
    "Adam":   "en_US-ryan-medium",      # deep, male
    "Bella":  "en_US-kristin-medium",   # soft, female
    "Antoni": "en_US-joe-medium",       # warm, male
    "Elli":   "en_US-ljspeech-medium",  # young, female
    "Josh":   "en_US-john-medium",      # casual, male
}
DEFAULT_VOICE = "Rachel"


# One-click color grading presets, applied at render time via ffmpeg's eq/
# colorbalance/hue filters — no separate render pass, just extra filter
# chain stages before the caption burn-in.
COLOR_PRESETS = [
    "none", "warm", "moody", "vibrant", "bw", "cinematic",
    # Paid-plan-only (see PREMIUM_COLOR_PRESETS below).
    "teal_orange", "vintage", "noir", "pastel", "dreamy",
]

# Caption text-style presets — each controls font, size, color, outline
# weight, per-word chunk size, and whether the pop-bounce/karaoke-color
# animation is used at all. Colours are ASS's &HAABBGGRR order.
CAPTION_STYLES = {
    "bold": dict(font="Liberation Sans Bold", size=80, primary="&H0000FFFF", secondary="&H00FFFFFF",
                 outline=5, shadow=2, bold=1, pop=True, karaoke=True, fade=False, chunk_size=4),
    "karaoke": dict(font="Liberation Sans Bold", size=74, primary="&H00FFFF00", secondary="&H00FFFFFF",
                     outline=4, shadow=1, bold=1, pop=False, karaoke=True, fade=False, chunk_size=5),
    "meme": dict(font="DejaVu Sans", size=88, primary="&H00FFFFFF", secondary="&H00FFFFFF",
                 outline=7, shadow=0, bold=1, pop=False, karaoke=False, fade=False, chunk_size=3),
    "minimal": dict(font="Liberation Sans", size=58, primary="&H00FFFFFF", secondary="&H00DDDDDD",
                     outline=2, shadow=0, bold=0, pop=False, karaoke=False, fade=True, chunk_size=6),
    # ---- Paid-plan-only styles (gated in the frontend, see PREMIUM_
    # markers on index.html's style-cards) — same rendering engine as the
    # free styles above, just different font/color/animation combos, so
    # there's no extra server cost either way. ----
    "neon": dict(font="Liberation Sans Bold", size=78, primary="&H00FF00FF", secondary="&H00FFFF00",
                 outline=6, shadow=2, bold=1, pop=True, karaoke=True, fade=False, chunk_size=4),
    "royal": dict(font="Liberation Sans Bold", size=76, primary="&H0000D7FF", secondary="&H00FFFFFF",
                  outline=5, shadow=2, bold=1, pop=False, karaoke=True, fade=False, chunk_size=5),
    "typewriter": dict(font="Liberation Mono Bold", size=70, primary="&H00FFFFFF", secondary="&H00FFFFFF",
                        outline=4, shadow=1, bold=1, pop=False, karaoke=False, fade=False, chunk_size=1),
    "shadow": dict(font="Liberation Sans Bold", size=82, primary="&H00FFFFFF", secondary="&H00FFFFFF",
                   outline=0, shadow=6, bold=1, pop=False, karaoke=False, fade=True, chunk_size=4),
    "impact": dict(font="Liberation Sans Bold", size=92, primary="&H00FFFFFF", secondary="&H0000FFFF",
                   outline=8, shadow=0, bold=1, pop=True, karaoke=True, fade=False, chunk_size=3),
}
DEFAULT_CAPTION_STYLE = "bold"
# Styles/presets only unlocked for paid plans — enforced in the frontend
# (these buttons are dimmed/locked for free-plan users, see index.html);
# the pipeline itself doesn't do auth/billing, it just renders whatever
# style key it's given, same as every other option here. Exposed via
# /caption-styles and /color-presets so the frontend doesn't need to keep
# its own duplicate list in sync.
PREMIUM_CAPTION_STYLES = {"neon", "royal", "typewriter", "shadow", "impact"}
PREMIUM_COLOR_PRESETS = {"teal_orange", "vintage", "noir", "pastel", "dreamy"}

# Voice-over tone presets — steer generate_voiceover_script()'s prompt
# rather than changing the TTS voice actor (that's the separate `voice`
# param). "ugc_ad" specifically asks Claude to read any on-screen text it
# can see in the sampled frames out loud (it's already looking at the
# frames for context, so this just tells it to use that ability) and close
# with a punchy confirmation line, matching a typical UGC ad format.
VOICEOVER_STYLES = {
    "narration": "a natural, documentary-style narrator describing what's happening on screen",
    "ugc_ad": (
        "a casual, enthusiastic UGC/creator-style advertisement voice — first-person, "
        "like a real person showing off a product to camera. If any on-screen text, "
        "captions, callouts, or prompts are visible in the frames, read them out loud "
        "naturally as part of the pitch instead of ignoring them. Close the script with "
        "a short, punchy confirmation line like \"and yeah, it definitely works\" or "
        "similar in spirit"
    ),
    "hype": "a high-energy hype voice — short punchy sentences, built for a viral moment",
    "calm": "a calm, relaxed, slow-paced voice",
}
DEFAULT_VOICEOVER_STYLE = "narration"

# Real, user-provided sound effect clips committed under sfx/ (see repo) —
# these replaced the earlier synthesized (ffmpeg lavfi) placeholders.
# "none" stays in the list only for the legacy one-shot ProcessRequest/
# ReclipRequest.sfx field below (unused by the new timeline placement UI,
# kept for backward compatibility) so "no effect" is still a valid value there.
SOUND_EFFECTS = ["none", "ting", "pop", "whoosh", "click", "keyboard"]
SFX_DIR = Path(__file__).parent / "sfx"
SFX_DIR.mkdir(exist_ok=True)

# Hard cap on how many sound-effect placements one /apply-sound-effects call
# will accept — generous for any real editing session, just a guard against
# an absurd/malformed request building a giant ffmpeg command.
MAX_SFX_PLACEMENTS = 60


class ProcessRequest(BaseModel):
    url: str
    clip_count: int = 3
    clip_seconds: int = 30
    instruction: str = ""  # optional free-text steer for Claude's highlight picking,
                            # e.g. "focus on the part where they argue about money"
    voiceover: bool = False           # force AI voice-over even if the clip has speech
    voiceover_voice: str = DEFAULT_VOICE
    voiceover_style: str = DEFAULT_VOICEOVER_STYLE  # one of VOICEOVER_STYLES
    zoom_pan: bool = False             # subtle continuous Ken Burns-style zoom-in
    color_preset: str = "none"         # one of COLOR_PRESETS
    caption_style: str = DEFAULT_CAPTION_STYLE  # one of CAPTION_STYLES
    sfx: str = "none"                  # one of SOUND_EFFECTS, one-shot
    sfx_position: str = "end"          # "start" or "end"
    typing_sound: bool = False         # ambient typing-click bed under the whole clip
    flash_intro: bool = False          # white flash-in + shutter click instead of a plain fade
    # Facecam crop — cuts an arbitrary rectangle out of the SOURCE frame
    # (fractions 0..1) and blows it up to fill the whole 1080x1920 output,
    # instead of just cropping a centered 9:16 window. Lets someone cut a
    # small facecam out of a corner and make IT the whole clip. crop_w<=0
    # is the sentinel for "no manual crop" — falls back to the old
    # default behavior (auto-centered 9:16 crop of the full frame). See
    # cut_and_caption() below for the actual filter.
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_w: float = 0.0
    crop_h: float = 1.0
    top_text: str = ""                 # optional pinned title/hook text at the top of the frame
    top_text_colors: List[str] = []    # per-word color for top_text, see TOP_TEXT_COLORS


class ReclipRequest(BaseModel):
    job_id: str
    start: Optional[float] = None     # explicit new start time; omit to auto-pick a fresh moment
    clip_seconds: int = 20
    instruction: str = ""
    exclude_starts: List[float] = []  # start times already used, so auto-pick doesn't repeat them
    voiceover: bool = False
    voiceover_voice: str = DEFAULT_VOICE
    voiceover_style: str = DEFAULT_VOICEOVER_STYLE
    zoom_pan: bool = False
    color_preset: str = "none"
    caption_style: str = DEFAULT_CAPTION_STYLE
    sfx: str = "none"
    sfx_position: str = "end"
    typing_sound: bool = False
    flash_intro: bool = False
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_w: float = 0.0
    crop_h: float = 1.0
    top_text: str = ""
    top_text_colors: List[str] = []


class RegenerateVoiceoverRequest(BaseModel):
    job_id: str
    clip_file: str        # the "file" value from a previous clip response, e.g. "70b799b2_0.mp4"
    script: str            # user-edited narration text
    voice: str = DEFAULT_VOICE


class SfxPlacement(BaseModel):
    effect: str    # one of SOUND_EFFECTS, e.g. "whoosh" — never "none" here
    time: float    # seconds from the clip's start where this effect should start playing


class ApplySoundEffectsRequest(BaseModel):
    clip_file: str                          # the "file" value from a previous clip response
    placements: List[SfxPlacement] = []     # empty list = strip effects back to the clean track


class ApplyGameplayBgRequest(BaseModel):
    clip_file: str
    enabled: bool               # True = composite a gameplay clip in; False = revert to the plain clip
    preset: Optional[str] = None  # a name from GET /gameplay-backgrounds; omit to use the job's custom upload instead


# ---------------------------------------------------------------------------
# Download + audio extraction
# ---------------------------------------------------------------------------
def _ytdlp_attempt(url: str, out_path: Path, dest: Path, proxy_url: str = "") -> str:
    """Runs the full client-rotation loop (see download_video's docstring
    for why each client is tried) ONCE, optionally through a single proxy
    for every client attempt. Returns "" on success, or the last error's
    stderr tail on failure — never raises, so callers can chain multiple
    tiers (no proxy → own proxy → paid proxy) without a try/except per
    tier."""
    base_args = [
        "yt-dlp",
        # Capped at 1080p: the final output is always cropped/scaled to
        # 1080x1920 anyway, so pulling a 4K (or higher) source just wastes
        # bandwidth and — more importantly — makes the later ffmpeg decode
        # step dramatically more memory-hungry. On a resource-limited
        # Railway container, decoding a 4K AV1 source was silently
        # OOM-killing the ffmpeg render step with no error message at all
        # (process just died right after starting, 0 frames encoded).
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--remote-components", "ejs:github",
        "--js-runtimes", "deno",
    ]
    if proxy_url:
        base_args += ["--proxy", proxy_url]
    # Point yt-dlp at our self-hosted PO token provider (see
    # POT_PROVIDER_URL above) so its bgutil plugin can fetch a
    # proof-of-origin token — this is a separate --extractor-args flag
    # from the per-client one below since it targets a different
    # extractor ("youtubepot-bgutilhttp" vs "youtube").
    if POT_PROVIDER_URL:
        base_args += ["--extractor-args", f"youtubepot-bgutilhttp:base_url={POT_PROVIDER_URL}"]

    # Optional: a real YouTube session's cookies, exported by the site owner
    # and set as the YTDLP_COOKIES env var (Netscape cookies.txt format).
    # YouTube increasingly rate-limits / bot-checks requests from datacenter
    # IPs like Railway's — a logged-in session's cookies are the most
    # reliable fix when that happens. Optional: if unset, we just don't
    # pass --cookies and rely on the client-spoofing fallbacks below.
    cookies_path = None
    if YTDLP_COOKIES:
        cookies_path = dest / "cookies.txt"
        cookies_path.write_text(YTDLP_COOKIES, encoding="utf-8")

    # YouTube now requires a "PO token" for some of its player clients
    # (notably the default "web" client), which yt-dlp can't always obtain
    # server-side — this shows up as "Sign in to confirm you're not a bot"
    # or HTTP 429 even for public videos. The embedded/mobile clients use a
    # different auth flow that usually doesn't need a PO token, so we try
    # those first and only fall back to "web" last. mweb and web_creator
    # are included because as of 2026 they're commonly the two that still
    # get through when android/ios/tv have started requiring PO tokens too
    # (YouTube's specific bot-check requirements shift client-by-client
    # over time, which is also why the Dockerfile force-upgrades yt-dlp on
    # every build instead of trusting whatever version got cached). Trying
    # several clients in one process also means a single transient failure
    # doesn't require restarting the whole clip request.
    client_attempts = ["android", "ios", "tv", "mweb", "web_creator", "web"]
    last_error = ""
    for client in client_attempts:
        args = base_args + ["--extractor-args", f"youtube:player_client={client}"]
        if cookies_path:
            args += ["--cookies", str(cookies_path)]
        args += ["-o", str(out_path), url]

        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists():
            return ""
        last_error = result.stderr[-1200:]
        print(f"yt-dlp attempt with player_client={client}"
              f"{' via proxy' if proxy_url else ''} failed, trying next client if any:\n{last_error}")
    return last_error


def download_video(url: str, dest: Path) -> Tuple[Path, str]:
    """Downloads the source video with yt-dlp (installed separately, see
    README), checking the shared source cache first (see SOURCE_CACHE_DIR
    above) so a video already downloaded recently — by anyone, for any
    job — never gets pulled twice.

    On a cache miss, tries three tiers in order, each one only reached if
    the previous genuinely failed:
      1. No proxy at all — just the PO token provider + cookies + the
         client-rotation loop. This is free and, per yt-dlp's own
         guidance, the most effective mitigation there is; it should
         succeed for the large majority of videos on its own.
      2. YTDLP_OWN_PROXY_URL, if set — anything free/self-hosted you're
         already running.
      3. YTDLP_PAID_PROXY_URL, if set — a commercial residential proxy
         (e.g. Decodo). The true last resort, so its bandwidth cost scales
         with how often YouTube actually blocks tier 1, not with total
         traffic.
    Every successful download (any tier) is written into the shared cache
    for next time.

    Returns (path, tier) where tier is one of "cache", "free", "own_proxy",
    "paid_proxy" — the caller (_process_source, below) uses this to decide
    whether the job actually cost anything on the paid proxy, instead of
    assuming every YouTube-link job did. Before this, EVERY youtube job
    reported its full downloaded size as "proxy usage" to the billing
    backend regardless of which tier actually served it — meaning testing
    (or any request the free tier handled fine) still chewed into the
    site-wide Decodo dollar budget and each plan's proxy-MB cap even though
    $0 was actually spent. Only a genuine paid_proxy download should count
    against that budget.
    """
    out_path = dest / "source.mp4"

    cached = _source_cache_get(url)
    if cached:
        shutil.copyfile(cached, out_path)
        return out_path, "cache"

    last_error = _ytdlp_attempt(url, out_path, dest, proxy_url="")

    if last_error and YTDLP_OWN_PROXY_URL:
        print("Free tier failed — retrying via YTDLP_OWN_PROXY_URL...")
        last_error = _ytdlp_attempt(url, out_path, dest, proxy_url=YTDLP_OWN_PROXY_URL) or ""
        if not last_error:
            _source_cache_put(url, out_path)
            return out_path, "own_proxy"
    elif not last_error:
        _source_cache_put(url, out_path)
        return out_path, "free"

    if last_error and YTDLP_PAID_PROXY_URL:
        print("Own-proxy tier failed (or unset) — retrying via YTDLP_PAID_PROXY_URL...")
        last_error = _ytdlp_attempt(url, out_path, dest, proxy_url=YTDLP_PAID_PROXY_URL) or ""
        if not last_error:
            _source_cache_put(url, out_path)
            return out_path, "paid_proxy"

    hint = (
        "\n\nYouTube is blocking downloads from this server (common for cloud-hosted "
        "IPs, even with a PO token provider and cookies configured — YouTube doesn't "
        "guarantee either bypasses its bot checks). Use the \"Upload your own video\" "
        "option instead — it skips YouTube entirely and always works."
    )
    raise RuntimeError(f"yt-dlp failed on all player clients across every available tier: {last_error}{hint}")


def extract_audio(video_path: Path, dest: Path) -> Path:
    """Pulls just the audio track out to a small AAC file — much faster to
    upload to Deepgram than shipping the whole video, especially over a
    server's network connection rather than a home GPU box's local disk."""
    audio_path = dest / "audio.m4a"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn",
         "-acodec", "aac", "-b:a", "128k", str(audio_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return audio_path


def get_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[-500:]}")
    try:
        return float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not read video duration ({e}). ffprobe output: {result.stdout[:300]}")


# ---------------------------------------------------------------------------
# Transcription — Deepgram (hosted, no GPU required)
# ---------------------------------------------------------------------------
def transcribe(audio_path: Path):
    """Sends the audio to Deepgram's pre-recorded API and returns
    word-level timestamps in the same shape the rest of the pipeline
    (highlight picking, caption generation) already expects."""
    if not DEEPGRAM_API_KEY:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is not set. Get a free key at "
            "https://console.deepgram.com/ and set it as an env var."
        )

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    resp = httpx.post(
        "https://api.deepgram.com/v1/listen",
        params={"model": "nova-2", "smart_format": "true", "punctuate": "true"},
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/mp4",
        },
        content=audio_bytes,
        timeout=300,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Deepgram request failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    try:
        raw_words = data["results"]["channels"][0]["alternatives"][0]["words"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Deepgram response shape ({e}): {json.dumps(data)[:500]}")

    words = [
        {
            "start": w["start"],
            "end": w["end"],
            "word": (w.get("punctuated_word") or w["word"]).strip(),
        }
        for w in raw_words
    ]
    return words


# ---------------------------------------------------------------------------
# Highlight picking — Claude first (semantic), heuristic as a fallback
# ---------------------------------------------------------------------------
def build_timestamped_transcript(words, bucket_seconds: int = 10) -> str:
    """Compacts word-level timestamps into readable `[12s] some words...`
    lines so the transcript fits comfortably in an LLM prompt."""
    if not words:
        return ""
    lines = []
    bucket_start = words[0]["start"]
    current = []
    for w in words:
        if w["start"] - bucket_start >= bucket_seconds and current:
            lines.append(f"[{bucket_start:.0f}s] " + " ".join(current))
            current = []
            bucket_start = w["start"]
        current.append(w["word"])
    if current:
        lines.append(f"[{bucket_start:.0f}s] " + " ".join(current))
    return "\n".join(lines)


def pick_highlights_llm(words, clip_count: int, clip_seconds: int, total_duration: float, instruction: str = "", exclude_ranges=None):
    """Asks Claude to pick the best moments in the video for short-form
    clips — hooks, punchlines, surprising or emotional beats — rather than
    just the windows with the most words in them. Returns None (so the
    caller falls back to the heuristic) if no API key is set or anything
    about the call/parsing goes wrong, so a flaky LLM response never takes
    down clip generation entirely.

    `instruction` is an optional free-text steer from the user (e.g. "focus
    on the part where they argue about money", "find the funniest moment")
    — when present it's given priority over the generic hook/punchline
    criteria, letting the user directly control what the AI looks for
    instead of only ever getting one fixed notion of "best".

    `exclude_ranges` is an optional list of (start, end) tuples the caller
    already has clips from — used by /reclip's "pick a different moment"
    so regenerating doesn't just hand back the same window again."""
    if not ANTHROPIC_API_KEY or not words:
        return None

    transcript = build_timestamped_transcript(words)
    instruction = (instruction or "").strip()
    steer = (
        f"The user specifically asked for: \"{instruction}\" — prioritize this over "
        f"generic criteria; only fall back to hooks/punchlines/emotional beats for any "
        f"remaining clips if the instruction doesn't specify enough moments.\n\n"
        if instruction else ""
    )
    exclude_note = (
        "Avoid these time ranges (already used for other clips): "
        + ", ".join(f"{s:.0f}s-{e:.0f}s" for s, e in exclude_ranges) + ".\n\n"
        if exclude_ranges else ""
    )
    prompt = (
        f"You're picking the {clip_count} best {clip_seconds}-second moments from this "
        f"video transcript, to turn into short-form clips for TikTok/Reels/Shorts. Look "
        f"for hooks, punchlines, surprising claims, or emotional beats — not just the "
        f"parts with the most talking.\n\n"
        f"{steer}"
        f"{exclude_note}"
        f"Transcript (format: [seconds] words spoken from that point):\n{transcript}\n\n"
        f"The video is {total_duration:.0f} seconds long. Pick {clip_count} start times, "
        f"each at least {clip_seconds} seconds before the video ends, and at least "
        f"{clip_seconds} seconds apart from each other so the clips don't overlap.\n\n"
        f"Respond with ONLY a JSON array, no other text, like:\n"
        f'[{{"start": 42, "reason": "short reason"}}, ...]'
    )

    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if resp.status_code >= 400:
            # Log the response body — Anthropic's error responses explain
            # exactly what's wrong (bad model name, bad request shape,
            # etc), which raise_for_status()'s exception message alone
            # doesn't include.
            print(f"Anthropic API error {resp.status_code}: {resp.text[:1000]}")
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]

        match = re.search(r"\[.*\]", text, re.DOTALL)
        picks = json.loads(match.group(0) if match else text)

        def overlaps_excluded(start, end):
            return any(not (end <= ex[0] or start >= ex[1]) for ex in (exclude_ranges or []))

        highlights = []
        for p in picks[:clip_count]:
            start = max(0.0, float(p["start"]))
            end = min(total_duration, start + clip_seconds)
            if end - start < clip_seconds * 0.5:
                continue
            if overlaps_excluded(start, end):
                continue
            highlights.append({"start": start, "end": end, "reason": p.get("reason", "")})

        highlights.sort(key=lambda h: h["start"])
        return highlights or None
    except Exception as e:
        print(f"LLM highlight picking failed, falling back to heuristic: {e}")
        return None


def _even_spaced_highlights(clip_count: int, clip_seconds: int, total_duration: float, exclude_ranges=None):
    """Spaces clip_count windows evenly across the video instead of
    scoring by word density — used when there's no transcript to score at
    all (silent or music-only source). This is exactly the case the
    voice-over feature exists for, so highlight picking must not hard-fail
    just because nobody's talking; it should still hand back clip windows
    for apply_voiceover_if_wanted() to narrate afterward."""
    exclude_ranges = exclude_ranges or []
    if total_duration <= 0:
        return []
    if total_duration <= clip_seconds:
        return [{"start": 0.0, "end": total_duration, "score": 0}]

    max_start = total_duration - clip_seconds
    if clip_count <= 1:
        starts = [0.0]
    else:
        step = max_start / (clip_count - 1)
        starts = [min(max_start, i * step) for i in range(clip_count)]

    highlights = []
    for s in starts:
        e = min(total_duration, s + clip_seconds)
        if any(not (e <= ex[0] or s >= ex[1]) for ex in exclude_ranges):
            continue
        highlights.append({"start": s, "end": e, "score": 0})
    return highlights


def pick_highlights_heuristic(words, clip_count: int, clip_seconds: int, total_duration: float, exclude_ranges=None):
    """Speech-density fallback: scores fixed-length windows by how many
    words land in them and returns the top non-overlapping windows. Used
    when no ANTHROPIC_API_KEY is set, or if the LLM call fails for any
    reason — clip generation should never hard-fail just because the
    smarter picker had a bad day.

    If there's no transcript at all (silent/music-only video), there's
    nothing to score by density — fall back to evenly-spaced windows
    instead of returning nothing, so silent videos can still get clips
    (with voice-over auto-applied to them downstream)."""
    if not words:
        return _even_spaced_highlights(clip_count, clip_seconds, total_duration, exclude_ranges)

    exclude_ranges = exclude_ranges or []
    stride = max(5, clip_seconds // 3)
    candidates = []
    t = 0.0
    while t + clip_seconds <= total_duration:
        window_words = [w for w in words if t <= w["start"] < t + clip_seconds]
        candidates.append({"start": t, "end": t + clip_seconds, "score": len(window_words)})
        t += stride

    candidates.sort(key=lambda c: c["score"], reverse=True)

    chosen = []
    for c in candidates:
        if len(chosen) >= clip_count:
            break
        overlaps_chosen = any(not (c["end"] <= x["start"] or c["start"] >= x["end"]) for x in chosen)
        overlaps_excluded = any(not (c["end"] <= ex[0] or c["start"] >= ex[1]) for ex in exclude_ranges)
        if not overlaps_chosen and not overlaps_excluded:
            chosen.append(c)

    # Very short or sparsely-worded videos can still come up empty even
    # with words present (e.g. a 10s clip request against a 12s video with
    # one word in it) — fall back to even spacing rather than failing.
    if not chosen:
        return _even_spaced_highlights(clip_count, clip_seconds, total_duration, exclude_ranges)

    chosen.sort(key=lambda c: c["start"])
    return chosen


def pick_highlights(words, clip_count: int, clip_seconds: int, total_duration: float, instruction: str = "", exclude_ranges=None):
    llm_picks = pick_highlights_llm(words, clip_count, clip_seconds, total_duration, instruction, exclude_ranges)
    if llm_picks:
        return llm_picks
    return pick_highlights_heuristic(words, clip_count, clip_seconds, total_duration, exclude_ranges)


# ---------------------------------------------------------------------------
# Animated captions — karaoke-style word highlighting via ASS + libass
# ---------------------------------------------------------------------------
def _ass_timestamp(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    cs = int(round((s - int(s)) * 100))
    if cs == 100:
        cs = 0
        s += 1
    return f"{h:d}:{m:02d}:{int(s):02d}.{cs:02d}"


# Word colors available for the top-of-frame title/hook text (see
# words_to_ass's top_text params below) — deliberately a small fixed
# palette rather than a free color picker, so it always reads clearly
# against video. ASS inline color override tags use \c&HBBGGRR& (blue-
# green-red hex, NOT rgb order).
TOP_TEXT_COLORS = {
    "white": "FFFFFF",
    "red": "0000FF",
    "green": "00FF00",
    "yellow": "00FFFF",
}
DEFAULT_TOP_TEXT_COLOR = "white"


def words_to_ass(words, clip_start: float, clip_end: float, ass_path: Path,
                  chunk_size: Optional[int] = None, caption_style: str = DEFAULT_CAPTION_STYLE,
                  top_text: str = "", top_text_colors: Optional[List[str]] = None):
    """Writes an .ass subtitle file scoped to one clip's time range. Text
    color, font, outline weight, per-line word count, and which animation
    (if any) is used are all driven by `caption_style` (see CAPTION_STYLES)
    — this is what makes the frontend's Bold Pop / Karaoke / Meme Stack /
    Minimal picker actually change the real burned-in captions, not just
    the demo preview.

    - "karaoke" styles use ASS \\k tags so words highlight from
      SecondaryColour to PrimaryColour exactly as they're spoken.
    - "pop" styles additionally bounce each word's scale via \\t
      transforms the instant it becomes active.
    - Styles with neither just render each line's full text statically —
      still perfectly readable, just no animation (e.g. Meme Stack, which
      traditionally isn't karaoke-highlighted at all).
    - "fade" styles (Minimal) fade each line in/out instead of a hard cut.

    `top_text` is an optional static title/hook line pinned to the top of
    the frame for the clip's whole duration — independent of the spoken
    captions above (different Style, different vertical position, no
    karaoke timing since it isn't tied to speech). `top_text_colors` is a
    list of colors (see TOP_TEXT_COLORS) matched to top_text's words by
    index — words past the end of the list, or an unrecognized color,
    fall back to white. This reuses the SAME libass burn-in pass as the
    spoken captions (one extra Dialogue line in the same .ass file) rather
    than a second ffmpeg filter stage, so it's effectively free.

    PlayResX/Y match the 1080x1920 output frame so font sizes and margins
    line up correctly after the crop+scale filter runs.
    """
    style = CAPTION_STYLES.get(caption_style, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])
    if chunk_size is None:
        chunk_size = style["chunk_size"]

    clip_words = [w for w in words if clip_start <= w["start"] < clip_end]

    # Colours are &HAABBGGRR. Outline/Shadow/Bold/Fontname/Fontsize all
    # come from the selected style so each preset actually looks distinct.
    # The second Style (TopText) is for the optional pinned title line —
    # Alignment 8 = top-center, MarginV 70 keeps it clear of the very top
    # edge/notch area on most phones.
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        # WrapStyle 2 = no automatic line-wrapping. With WrapStyle 0 (the
        # previous setting), libass recalculates line breaks live off the
        # CURRENT rendered width — and since each word's pop-scale bounce
        # briefly makes it ~1/5 wider, a line that fit on one row would
        # momentarily overflow, get auto-wrapped onto two rows, then snap
        # back to one row the instant the pop finished. That's the visible
        # "scales and goes back to normal" glitch. WrapStyle 2 fixes it by
        # never re-wrapping — combined with a small pop and modest
        # word-group sizes, lines stay comfortably within frame.
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Karaoke,{style['font']},{style['size']},{style['primary']},{style['secondary']},"
        f"&H00000000,&H00000000,{style['bold']},0,0,0,100,100,0,0,1,{style['outline']},"
        f"{style['shadow']},2,60,60,190,1\n"
        "Style: TopText,Liberation Sans Bold,66,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,4,1,8,50,50,70,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    top_words = top_text.strip().split() if top_text and top_text.strip() else []
    if top_words:
        colors = top_text_colors or []
        runs = []
        for i, w in enumerate(top_words):
            color = colors[i] if i < len(colors) else DEFAULT_TOP_TEXT_COLOR
            hexcode = TOP_TEXT_COLORS.get(color, TOP_TEXT_COLORS[DEFAULT_TOP_TEXT_COLOR])
            runs.append(f"{{\\c&H{hexcode}&}}{w}")
        top_line = (
            f"Dialogue: 1,{_ass_timestamp(0)},{_ass_timestamp(clip_end - clip_start)},"
            f"TopText,,0,0,0,,{' '.join(runs)}\n"
        )
    else:
        top_line = ""

    # group into on-screen lines of `chunk_size` words each
    chunks = [clip_words[i:i + chunk_size] for i in range(0, len(clip_words), chunk_size)]

    POP_MS = 90      # how long the scale-up half of each word's pop takes
    POP_SCALE = 110  # smaller bump leaves margin before a line could ever
                      # overflow its row, on top of the WrapStyle fix above
    FADE_MS = 150    # in/out fade duration for "fade"-style presets

    lines = [header, top_line]
    for chunk in chunks:
        if not chunk:
            continue
        line_start = chunk[0]["start"] - clip_start
        line_end = chunk[-1]["end"] - clip_start

        if not style["karaoke"]:
            # Static (or fade-only) presets: no per-word \k timing needed,
            # just render the whole line's text as one plain string.
            text = " ".join(w["word"].strip() for w in chunk if w["word"].strip())
            if not text:
                continue
            prefix = f"{{\\fad({FADE_MS},{FADE_MS})}}" if style["fade"] else ""
            lines.append(
                f"Dialogue: 0,{_ass_timestamp(line_start)},{_ass_timestamp(line_end)},"
                f"Karaoke,,0,0,0,,{prefix}{text}\n"
            )
            continue

        karaoke_text = ""
        for w in chunk:
            dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))  # centiseconds
            word = w["word"].strip()
            if not word:
                continue
            # Offset of this word's own start, in ms, relative to the
            # Dialogue line's start — \t's time args are line-relative,
            # which is what lets each word in the group pop at its own
            # moment instead of all together.
            word_offset_ms = max(0, int(round((w["start"] - clip_start - line_start) * 1000)))
            if style["pop"]:
                pop_start = word_offset_ms
                pop_mid = word_offset_ms + POP_MS
                pop_end = word_offset_ms + POP_MS * 2
                karaoke_text += (
                    f"{{\\k{dur_cs}"
                    f"\\t({pop_start},{pop_mid},\\fscx{POP_SCALE}\\fscy{POP_SCALE})"
                    f"\\t({pop_mid},{pop_end},\\fscx100\\fscy100)}}"
                    f"{word} "
                )
            else:
                karaoke_text += f"{{\\k{dur_cs}}}{word} "

        if not karaoke_text.strip():
            continue

        lines.append(
            f"Dialogue: 0,{_ass_timestamp(line_start)},{_ass_timestamp(line_end)},"
            f"Karaoke,,0,0,0,,{karaoke_text.strip()}\n"
        )

    ass_path.write_text("".join(lines), encoding="utf-8")


def _zoompan_filter(fps: int = 30) -> str:
    """A slow, continuous zoom-in (subtle Ken Burns effect) applied
    directly to normal video frames — d=1 keeps one output frame per
    input frame instead of zoompan's default image-slideshow behavior,
    which is what lets this run on a real video clip instead of a still.
    The zoom grows a tiny amount every frame and is capped at 1.15x so
    the crop window never tightens enough to clip into the caption
    safe-margins burned in afterward."""
    return (
        "zoompan=z='min(zoom+0.0008,1.15)':d=1:"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s=1080x1920:fps={fps}"
    )


def _color_grade_filter(preset: str) -> str:
    """Returns an ffmpeg filter fragment for a named look, or "" for
    'none'/unknown presets (caller just skips this stage). These are
    built from eq (brightness/contrast/saturation/gamma) and colorbalance
    (per-channel shadow/midtone push) rather than external LUTs, so no
    extra files need to ship with the service."""
    presets = {
        "warm": "eq=contrast=1.05:brightness=0.02:saturation=1.15,"
                "colorbalance=rs=0.08:gs=0.02:bs=-0.08:rm=0.06:bm=-0.06",
        "moody": "eq=contrast=1.15:brightness=-0.03:saturation=0.85:gamma=0.92,"
                 "colorbalance=bs=0.08:bm=0.05",
        "vibrant": "eq=contrast=1.1:saturation=1.4:brightness=0.01",
        "bw": "hue=s=0,eq=contrast=1.15",
        "cinematic": "eq=contrast=1.12:saturation=0.9:gamma=0.95,"
                     "colorbalance=rs=0.04:bs=0.06:rm=0.02:bm=0.04",
        # ---- Paid-plan-only (see PREMIUM_COLOR_PRESETS) — same eq/
        # colorbalance/hue building blocks as the free presets above, no
        # external LUTs or extra ffmpeg passes, so identical render cost. ----
        "teal_orange": "eq=contrast=1.15:saturation=1.2,"
                        "colorbalance=rs=0.1:gs=-0.05:bs=-0.15:rm=0.08:bm=-0.1",
        "vintage": "eq=contrast=0.95:brightness=0.03:saturation=0.7:gamma=1.05,"
                   "colorbalance=rs=0.1:gs=0.05:bs=-0.1",
        "noir": "hue=s=0,eq=contrast=1.3:brightness=-0.05:gamma=0.85",
        "pastel": "eq=contrast=0.9:saturation=0.75:brightness=0.05:gamma=1.08",
        "dreamy": "eq=contrast=0.92:saturation=0.85:brightness=0.04:gamma=1.1,"
                  "colorbalance=rs=0.05:bs=0.05",
    }
    return presets.get(preset, "")


def cut_and_caption(source: Path, start: float, end: float, ass_path: Path, out_path: Path,
                     zoom_pan: bool = False, color_preset: str = "none", flash_intro: bool = False,
                     crop_x: float = 0.0, crop_y: float = 0.0, crop_w: float = 0.0, crop_h: float = 1.0):
    """Cuts the clip, reframes to 9:16, optionally applies a Ken Burns zoom
    and/or a color grading preset, burns in the animated karaoke captions,
    and adds a short automatic fade-in/fade-out on every clip — all via
    ffmpeg. Captions are always the LAST video filter stage so they stay
    crisp on top of any zoom/color grading rather than getting graded or
    zoomed themselves.

    `flash_intro` swaps the fade-in from black to white, giving a quick
    camera-flash feel instead of a plain fade — paired with a synthesized
    shutter "click" mixed into the audio afterward by apply_audio_extras().

    The ass/subtitles filters' path handling is notoriously fragile on
    Windows: ffmpeg's filtergraph mini-language uses ':' to separate
    filter options and '\\' as an escape character, both of which collide
    with normal Windows paths like C:\\Users\\you\\AppData\\...\\clip_0.ass.
    Instead of fighting that escaping, we run ffmpeg with its working
    directory set to the ASS file's own folder and reference it by BARE
    FILENAME only — no colons, no backslashes, nothing for the filtergraph
    parser to misinterpret.
    """
    duration = end - start
    ass_dir = ass_path.parent.resolve()
    ass_name = ass_path.name  # bare filename — no separators, nothing to escape

    # Fade duration is capped relative to clip length so a very short clip
    # never has its fade-in and fade-out overlap.
    fade_d = max(0.0, min(0.3, duration / 6))
    fade_color = "white" if flash_intro else "black"

    # Facecam crop — crop_w/crop_h/crop_x/crop_y (all fractions 0..1 of the
    # SOURCE frame) let the caller cut an ARBITRARY rectangle out of the
    # frame and blow it up to fill the whole 1080x1920 output, e.g. cutting
    # just a facecam out of a corner and making it the entire clip. This
    # replaced the old single-point "reposition" pan control, which could
    # only slide a fixed 9:16 window around and couldn't target something
    # as small as a corner facecam.
    #
    # crop_w<=0 is the sentinel for "no manual crop" (the default) — falls
    # back to the previous behavior, an auto-centered 9:16 crop of the full
    # frame, using ffmpeg's own iw/ih expressions so it adapts to whatever
    # the source resolution actually is without Python needing to know it
    # up front.
    if crop_w and crop_w > 0:
        # Manual box — clamped defensively since this ultimately comes from
        # user input via the API. scale+crop (rather than a plain stretch)
        # so a facecam box that isn't already 9:16 fills the frame cleanly
        # without warping faces.
        crop_x = max(0.0, min(1.0, crop_x))
        crop_y = max(0.0, min(1.0, crop_y))
        crop_w = max(0.05, min(1.0 - crop_x, crop_w))
        crop_h = max(0.05, min(1.0 - crop_y, crop_h))
        vf_stages = [
            f"crop=w='iw*{crop_w:.4f}':h='ih*{crop_h:.4f}':"
            f"x='iw*{crop_x:.4f}':y='ih*{crop_y:.4f}'",
            "scale=1080:1920:force_original_aspect_ratio=increase",
            "crop=1080:1920",
        ]
    else:
        vf_stages = [
            "crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)':x='(in_w-out_w)/2':y='(in_h-out_h)/2'",
            "scale=1080:1920",
        ]
    if zoom_pan:
        vf_stages.append(_zoompan_filter())
    if color_preset and color_preset != "none":
        cg = _color_grade_filter(color_preset)
        if cg:
            vf_stages.append(cg)
    vf_stages.append(f"ass='{ass_name}'")
    if fade_d > 0:
        vf_stages.append(f"fade=t=in:st=0:d={fade_d}:color={fade_color}")
        vf_stages.append(f"fade=t=out:st={max(0.0, duration - fade_d)}:d={fade_d}")
    vf = ",".join(vf_stages)

    af_stages = []
    if fade_d > 0:
        af_stages.append(f"afade=t=in:st=0:d={fade_d}")
        af_stages.append(f"afade=t=out:st={max(0.0, duration - fade_d)}:d={fade_d}")
    af = ",".join(af_stages)

    # -threads 2: without this, libx264 auto-detects the host's full core
    # count (60+ on some Railway hosts) and allocates per-thread buffers
    # accordingly, which was spiking memory usage enough to get the process
    # OOM-killed silently (no ffmpeg error text at all, just a nonzero exit
    # with 0 frames encoded). Capping threads keeps memory use predictable
    # on a resource-limited container.
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", str(source.resolve()), "-t", str(duration),
           "-vf", vf]
    if af:
        cmd += ["-af", af]
    # -pix_fmt yuv420p: without this, ffmpeg inherits whatever chroma
    # subsampling the SOURCE video uses. Some uploaded files (certain screen
    # recordings/exports) are natively 4:4:4 — libx264 then encodes in "High
    # 4:4:4 Predictive" profile, which takes roughly double the memory of
    # standard 4:2:0 and was very likely what pushed a single render over
    # this container's 1GB ceiling (confirmed via a real "Killed" kernel
    # log line right after a 4:4:4 encode). 4:2:0 is also what almost every
    # platform (TikTok, Instagram, etc) and player actually expects — 4:4:4
    # output isn't reliably compatible anyway, so this is a strict
    # improvement, not just a memory workaround.
    # Quality bump: previously no -crf was set at all, so libx264 fell back
    # to its own default (23) at "veryfast", which is tuned for encode
    # speed over compression efficiency — soft/blocky on higher-motion
    # footage (gameplay, fast cuts). -crf 20 is a real, visible sharpness
    # improvement (lower = higher quality/bigger file); "fast" trades a bit
    # more encode time for noticeably better compression than "veryfast" at
    # the same quality. This does NOT touch the levers that actually caused
    # the OOM history above (thread count, pixel format, source resolution)
    # — those stay exactly as fixed — so this should be a safe quality win,
    # not a reintroduction of that risk. Bumped -b:a too since audio was
    # left at ffmpeg's low-ish AAC default.
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "2", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", str(out_path.resolve())]
    result = subprocess.run(
        cmd,
        capture_output=True, text=True,
        cwd=str(ass_dir),  # <-- this is what makes the bare filename resolve correctly
    )
    if result.returncode != 0:
        # Print the FULL stderr to the server's own logs (Railway captures
        # stdout/stderr) since the exception message shown to the client is
        # truncated — the real cause is often earlier in ffmpeg's output
        # than what fits in that truncated tail.
        print(f"ffmpeg command: {' '.join(cmd)}")
        print(f"ffmpeg exit code: {result.returncode}")
        print(f"ffmpeg full stderr:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr[-2500:]}")


def clip_word_count(words, start: float, end: float) -> int:
    return len([w for w in words if start <= w["start"] < end])


# ---------------------------------------------------------------------------
# Voice-over — the clip's own transcript is the script (free, no LLM call,
# used whenever there's real dialogue in range), Claude+vision is a paid
# fallback for genuinely silent/music-only clips, and Google/ElevenLabs/
# Piper speak whatever script comes out of either path (see
# synthesize_voiceover further down).
# ---------------------------------------------------------------------------
def extract_sample_frames(source: Path, start: float, end: float, dest: Path, n: int = 3) -> List[Path]:
    """Grabs n evenly-spaced still frames from [start, end] so Claude can
    "see" what's happening in a clip that has no dialogue to transcribe."""
    frames = []
    span = max(end - start, 1.0)
    for i in range(1, n + 1):
        t = start + span * (i / (n + 1))
        out = dest / f"vo_frame_{uuid.uuid4().hex[:6]}_{i}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(source.resolve()), "-frames:v", "1", "-q:v", "3", str(out)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and out.exists():
            frames.append(out)
        else:
            print(f"Frame grab at {t:.1f}s failed: {result.stderr[-300:]}")
    return frames


def transcript_script(words, start: float, end: float) -> str:
    """Builds a narration script straight from the clip's own real
    transcript for [start, end] — no LLM call, so it costs nothing and
    never depends on ANTHROPIC_API_KEY having credits. This is now the
    primary script source for a forced voice-over (see
    apply_voiceover_if_wanted below): re-voicing the clip's existing
    dialogue in a different voice, rather than Claude writing new
    descriptive narration. Returns "" if nothing was transcribed in that
    range (a genuinely silent/music-only clip has no words to read back),
    in which case Claude+vision scene description is the only thing that
    could fill it in — see the fallback in apply_voiceover_if_wanted."""
    segment_words = [w["word"] for w in words if start <= w["start"] < end]
    return " ".join(segment_words).strip()


def generate_voiceover_script(frame_paths: List[Path], clip_seconds: int, instruction: str = "",
                               style: str = DEFAULT_VOICEOVER_STYLE) -> str:
    """Asks Claude to look at sampled frames and write a short narration
    script sized to fit the clip's duration, in the tone described by
    `style` (see VOICEOVER_STYLES) — this is what lets the frontend's
    Narration / UGC Ad / Hype / Calm picker actually change how the script
    is written, including UGC Ad mode reading any on-screen text visible
    in the frames out loud and closing with a tagline. Returns "" (never
    raises) on any failure so a bad script never blocks the rest of clip
    generation — the caller just skips voice-over for that clip."""
    if not ANTHROPIC_API_KEY or not frame_paths:
        return ""

    words_budget = max(8, int(clip_seconds * 2.5))  # ~2.5 spoken words/sec is a natural narration pace
    steer = f' The creator specifically asked for: "{instruction}".' if instruction else ""
    style_desc = VOICEOVER_STYLES.get(style, VOICEOVER_STYLES[DEFAULT_VOICEOVER_STYLE])

    content = []
    for fp in frame_paths:
        try:
            b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        except Exception:
            continue
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
    if not content:
        return ""
    content.append({
        "type": "text",
        "text": (
            f"These are frames sampled evenly across a {clip_seconds}-second video clip that has "
            f"no spoken dialogue (music-only or silent). Write a short voice-over narration script "
            f"for it, written in the voice of {style_desc}. Aim for about {words_budget} words so it "
            f"fits {clip_seconds} seconds read at a natural pace."
            f"{steer} Respond with ONLY the narration text — no quotes, no stage directions, no "
            f"timestamps."
        ),
    })

    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=60,
        )
        if resp.status_code >= 400:
            print(f"Anthropic API error (voiceover script) {resp.status_code}: {resp.text[:1000]}")
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception as e:
        print(f"Voiceover script generation failed: {e}")
        return ""


def _synthesize_google_tts(script: str, voice_name: str, dest: Path) -> Optional[Path]:
    """Calls Google Cloud Text-to-Speech (REST, plain API key — no service
    account JSON needed). Returns None on any failure, same "never raise"
    contract as _synthesize_elevenlabs below."""
    if not GOOGLE_TTS_API_KEY:
        return None

    voice = GOOGLE_TTS_VOICES.get(voice_name, GOOGLE_TTS_VOICES[DEFAULT_VOICE])
    try:
        resp = httpx.post(
            f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}",
            json={
                "input": {"text": script},
                "voice": voice,
                "audioConfig": {"audioEncoding": "MP3"},
            },
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"Google TTS error {resp.status_code}: {resp.text[:500]}")
            return None
        audio_b64 = resp.json().get("audioContent")
        if not audio_b64:
            print("Google TTS response had no audioContent")
            return None
        audio_path = dest / f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
        audio_path.write_bytes(base64.b64decode(audio_b64))
        return audio_path
    except Exception as e:
        print(f"Google TTS request failed: {e}")
        return None


def _synthesize_elevenlabs(script: str, voice_name: str, dest: Path) -> Optional[Path]:
    """Calls ElevenLabs to turn a script into speech. Returns None (never
    raises) on any failure — same "degrade gracefully" pattern as the rest
    of this pipeline, since voice-over is always an enhancement, never a
    requirement for a clip to come back successfully."""
    if not ELEVENLABS_API_KEY:
        return None

    voice_id = ELEVENLABS_VOICES.get(voice_name, ELEVENLABS_VOICES[DEFAULT_VOICE])
    try:
        resp = httpx.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": script,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"ElevenLabs error {resp.status_code}: {resp.text[:500]}")
            return None
        audio_path = dest / f"voiceover_{uuid.uuid4().hex[:8]}.mp3"
        audio_path.write_bytes(resp.content)
        return audio_path
    except Exception as e:
        print(f"ElevenLabs request failed: {e}")
        return None


def _synthesize_piper(script: str, voice_name: str, dest: Path) -> Optional[Path]:
    """Synthesizes speech locally with Piper — no API key, no network
    call, no per-character cost. Voice models are baked into the image at
    /app/piper_voices (see Dockerfile). This is the last resort in
    synthesize_voiceover()'s fallback chain, so it should essentially
    never fail unless the model files themselves are missing."""
    model_name = PIPER_VOICES.get(voice_name, PIPER_VOICES[DEFAULT_VOICE])
    model_path = PIPER_VOICES_DIR / f"{model_name}.onnx"
    if not model_path.exists():
        print(f"Piper voice model missing: {model_path}")
        return None

    try:
        audio_path = dest / f"voiceover_{uuid.uuid4().hex[:8]}.wav"
        # Current piper-tts (OHF-Voice/piper1-gpl) CLI takes the text as a
        # positional arg after `--`, not via stdin — see
        # https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md.
        # Invoked as `python3 -m piper` rather than a bare `piper` binary
        # since the module entry point is guaranteed by the pip package;
        # a console-script shim isn't.
        result = subprocess.run(
            ["python3", "-m", "piper", "-m", str(model_path), "-f", str(audio_path), "--", script],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0 or not audio_path.exists():
            print(f"Piper synthesis failed: {result.stderr[-1000:]}")
            return None
        return audio_path
    except Exception as e:
        print(f"Piper request failed: {e}")
        return None


def synthesize_voiceover(script: str, voice_name: str, dest: Path) -> Optional[Path]:
    """Turns a narration script into speech, trying Google Cloud TTS first
    (generous permanent free tier — see GOOGLE_TTS_API_KEY above), then
    ElevenLabs if Google isn't configured or fails, then Piper (self-
    hosted, no key needed — see _synthesize_piper above) as the final,
    always-available fallback. Returns None (never raises) only if every
    provider fails outright — voice-over is always an enhancement, never a
    requirement for a clip to come back successfully."""
    if not script.strip():
        return None
    return (
        _synthesize_google_tts(script, voice_name, dest)
        or _synthesize_elevenlabs(script, voice_name, dest)
        or _synthesize_piper(script, voice_name, dest)
    )


def mux_voiceover(clip_path: Path, voiceover_path: Path, out_path: Path) -> bool:
    """Replaces a clip's audio track entirely with the synthesized
    narration (these are silent/music-only clips by the time we get here,
    so there's no dialogue to preserve or mix under). -shortest keeps the
    output in sync with whichever of the two streams is shorter."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path.resolve()),
        "-i", str(voiceover_path.resolve()),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Voiceover mux failed: {result.stderr[-1500:]}")
        return False
    return True


def apply_voiceover_if_wanted(source: Path, job_dir: Path, out_path: Path, start: float, end: float,
                               clip_seconds: int, words, force: bool, voice_name: str, instruction: str = "",
                               voiceover_style: str = DEFAULT_VOICEOVER_STYLE) -> Optional[str]:
    """Runs the full voice-over pipeline for one clip IF it's wanted — either
    because the caller explicitly asked for it (force=True) or because the
    clip has almost no spoken words in it (auto-detected as silent/music).
    Mutates out_path in place on success. Returns the narration script text
    on success, or None if voice-over wasn't attempted/wanted or failed —
    callers should treat None as "no voice-over on this clip", never as an
    error to surface to the user."""
    if not (force or clip_word_count(words, start, end) < 3):
        return None
    # No provider-configured check here anymore — Piper (self-hosted, no
    # API key) is always available as the last resort in
    # synthesize_voiceover()'s fallback chain.

    try:
        # Real transcript first — free, no LLM call, always available.
        # Claude+vision is only reached for as a fallback when there's
        # nothing transcribed to read back (a genuinely silent/music-only
        # clip), and even then only does something if ANTHROPIC_API_KEY
        # has credits — otherwise generate_voiceover_script() returns ""
        # same as before and voice-over is skipped for that clip.
        script = transcript_script(words, start, end)
        if not script:
            frames = extract_sample_frames(source, start, end, job_dir)
            script = generate_voiceover_script(frames, clip_seconds, instruction, voiceover_style)
        if not script:
            return None
        vo_audio = synthesize_voiceover(script, voice_name, job_dir)
        if not vo_audio:
            return None
        muxed = out_path.with_suffix(".vo.mp4")
        if mux_voiceover(out_path, vo_audio, muxed):
            muxed.replace(out_path)
            return script
    except Exception as e:
        print(f"Voiceover pipeline failed for {out_path.name}: {e}")
    return None


# ---------------------------------------------------------------------------
# Sound effects, typing-click ambience, and YouTube-link background music —
# all layered onto the clip's existing audio (original speech or AI
# narration, whichever came out of the steps above) as a final mix pass.
# ---------------------------------------------------------------------------
SFX_FILE_EXTS = (".mp3", ".wav", ".m4a", ".ogg")


def _ensure_sfx(name: str) -> Optional[Path]:
    """Resolves a sound effect name to its audio file. These are real
    clips committed under sfx/ in the repo (whoosh/click/pop/ting/
    keyboard) rather than synthesized. Matches case-insensitively (e.g.
    "Whoosh.mp3" or "WHOOSH.MP3" both resolve to "whoosh") since these
    were hand-uploaded via GitHub's web UI and Railway's Linux containers
    are case-sensitive on disk — without this, a capitalized filename
    would silently fail to match and the effect would just no-op. Returns
    None for "none"/unknown names, or if no matching asset exists in this
    deploy (caller treats that as "skip the effect", never a hard error)."""
    if not name or name == "none" or name not in SOUND_EFFECTS:
        return None
    if not SFX_DIR.exists():
        return None
    name_lower = name.lower()
    for f in SFX_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in SFX_FILE_EXTS and f.stem.lower() == name_lower:
            return f
    print(f"SFX asset missing for '{name}' — expected a file like {SFX_DIR}/{name}.mp3 (any case)")
    return None


def _ensure_typing_loop() -> Optional[Path]:
    """A single click padded out to ~180ms of silence, meant to be played
    back-to-back via -stream_loop to build a steady typing-keyboard
    texture without needing per-word timing precision."""
    path = SFX_DIR / "typing_loop.wav"
    if path.exists():
        return path
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "sine=frequency=2800:duration=0.035,afade=t=out:st=0:d=0.03,volume=0.5,apad=pad_dur=0.15",
             "-ar", "44100", str(path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not path.exists():
            print(f"Typing-loop synth failed: {result.stderr[-400:]}")
            return None
        return path
    except Exception as e:
        print(f"Typing-loop synth error: {e}")
        return None


BG_MUSIC_EXTS = (".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac")


def find_bg_music(job_dir: Path) -> Optional[Path]:
    """Background music is now a direct file upload (see /upload-music)
    saved once per job as job_dir/bgmusic.<ext>, instead of a YouTube link
    that had to be downloaded via yt-dlp on every render — which used to
    be a real source of bot-detection failures on top of the video
    download itself. Every clip in a job shares the same track, so this
    is checked fresh on each render instead of being threaded through
    request bodies."""
    for ext in BG_MUSIC_EXTS:
        candidate = job_dir / f"bgmusic{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Split-screen gameplay background — the "Subway Surfers on the bottom
# half" retention/anti-shadowban trick. Two sources, either can be picked
# per clip in /apply-gameplay-bg:
#   1) Built-in presets — real gameplay clips committed under
#      gameplay_bg/ in the repo (see GAMEPLAY_BG_DIR below), the same
#      pattern as the sfx/ sound effects: whatever files are dropped in
#      there just show up as options, no code change needed. This app
#      doesn't record or license this footage itself — the founder
#      sources and commits it, same as the sound effects were.
#   2) A per-job custom upload (/upload-gameplay-bg) — for anyone who
#      wants to use their own footage instead of a built-in preset.
# Either way this is an explicit opt-in per clip, never automatic.
# ---------------------------------------------------------------------------
GAMEPLAY_BG_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".m4v")
GAMEPLAY_BG_DIR = Path(__file__).parent / "gameplay_bg"
GAMEPLAY_BG_DIR.mkdir(exist_ok=True)


def list_gameplay_bg_presets() -> List[str]:
    """Scans gameplay_bg/ for committed preset clips — returns names with
    no extension (e.g. "minecraft_parkour"), sorted. Dynamic on purpose:
    dropping a new file into that folder and committing it is the whole
    process for adding a new preset, no other code changes needed."""
    if not GAMEPLAY_BG_DIR.exists():
        return []
    return sorted({p.stem for p in GAMEPLAY_BG_DIR.iterdir() if p.suffix.lower() in GAMEPLAY_BG_EXTS})


def resolve_gameplay_bg_preset(name: str) -> Optional[Path]:
    """Case-insensitive lookup of a preset name to its file — presets get
    hand-typed into a request body (well, picked from a frontend button,
    but same idea as sfx names), so this is forgiving about case the same
    way _ensure_sfx is."""
    if not name:
        return None
    name_lower = name.lower()
    for f in GAMEPLAY_BG_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in GAMEPLAY_BG_EXTS and f.stem.lower() == name_lower:
            return f
    return None


def find_gameplay_bg(job_dir: Path) -> Optional[Path]:
    for ext in GAMEPLAY_BG_EXTS:
        candidate = job_dir / f"gameplaybg{ext}"
        if candidate.exists():
            return candidate
    return None


def apply_gameplay_split(clip_path: Path, out_path: Path, bg_path: Path) -> bool:
    """Composites the clip into the TOP half of the frame and a random
    segment of the gameplay footage into the BOTTOM half — the finished
    clip's own audio (speech/voiceover/sfx/music, whatever's already
    mixed in) is kept as-is; the gameplay footage is always muted so it
    never competes with it. A random start offset into the background
    footage (looped if it's shorter than the clip) means the same
    uploaded gameplay file doesn't look identical on every clip that uses
    it. Re-encodes video (this is a real composite, not a stream copy),
    but it's still just one ffmpeg pass over an already-finished clip —
    no re-transcription, re-captioning, or re-cutting involved."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path.resolve())],
            capture_output=True, text=True,
        )
        clip_duration = float(probe.stdout.strip() or 0) or 1.0
    except Exception:
        clip_duration = 1.0

    try:
        probe_bg = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(bg_path.resolve())],
            capture_output=True, text=True,
        )
        bg_duration = float(probe_bg.stdout.strip() or 0) or clip_duration
    except Exception:
        bg_duration = clip_duration

    max_start = max(0.0, bg_duration - clip_duration)
    bg_start = random.uniform(0, max_start) if max_start > 0 else 0.0

    # Fade duration capped the same way cut_and_caption does, so a very
    # short clip never has its fade-in and fade-out overlap.
    fade_d = max(0.0, min(0.3, clip_duration / 6))

    # Same 1GB-container memory ceiling as the main render path (see notes
    # near HEAVY_TASK_LOCK above) — this composite decodes TWO video
    # streams at once (the clip + the gameplay footage) instead of one,
    # so it's more memory-hungry than a normal render for the same
    # output size. Confirmed OOM-killed in production at 1080x1920 with
    # -preset veryfast (silent nonzero exit, no ffmpeg error text, dies
    # right as encoding starts) — the exact same signature as the
    # OOM documented at the top of this file. Dropping to 720x1280 and
    # -preset ultrafast, plus capping BOTH inputs' decode threads (not
    # just the encoder's), keeps peak memory well under the ceiling.
    # 720x1280 is still plenty sharp for a phone screen.
    #
    # The TOP half (the real clip) uses force_original_aspect_ratio=
    # DECREASE + pad, not crop — it used to crop-to-fill like the bottom
    # half does, but that silently cropped away the burned-in captions AND
    # the top-text overlay every time, since both sit right at the very
    # top/bottom edges of the 1080x1920 frame (by design — that's where
    # subtitles and pinned titles go) and a center-crop down to a squarish
    # 720x640 box discards exactly those edge bands first. Fit+pad (small
    # black bars on the sides instead) guarantees nothing burned into the
    # clip ever gets cut off. The BOTTOM half (gameplay b-roll) still
    # crops-to-fill since there's no fixed UI element there that matters if
    # it gets cropped.
    vf_out = (
        f"fade=t=in:st=0:d={fade_d}:color=black,"
        f"fade=t=out:st={max(0.0, clip_duration - fade_d)}:d={fade_d}"
    ) if fade_d > 0 else "null"
    cmd = [
        "ffmpeg", "-y",
        "-threads", "1", "-i", str(clip_path.resolve()),
        "-threads", "1", "-ss", f"{bg_start:.2f}", "-stream_loop", "-1", "-i", str(bg_path.resolve()),
        "-filter_complex",
        "[0:v]scale=720:640:force_original_aspect_ratio=decrease,"
        "pad=720:640:(ow-iw)/2:(oh-ih)/2:color=black[top];"
        "[1:v]scale=720:640:force_original_aspect_ratio=increase,crop=720:640[bot];"
        f"[top][bot]vstack=inputs=2[stacked];[stacked]{vf_out}[v]",
        "-map", "[v]", "-map", "0:a?",
        "-t", f"{clip_duration:.2f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(out_path.resolve()),
    ]
    if fade_d > 0:
        cmd += ["-af", f"afade=t=in:st=0:d={fade_d},afade=t=out:st={max(0.0, clip_duration - fade_d)}:d={fade_d}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Gameplay split-screen composite failed (exit {result.returncode}): {result.stderr[-1500:]}")
        return False
    return True


def apply_audio_extras(clip_path: Path, out_path: Path, duration: float, sfx: str, sfx_position: str,
                        typing_sound: bool, bg_music_path: Optional[Path]) -> bool:
    """Layers an optional one-shot sound effect, an ambient typing-click
    bed, and/or background music UNDER the clip's current audio track
    (whatever cut_and_caption/apply_voiceover_if_wanted left it with).
    Returns True and writes out_path if anything was actually mixed in;
    False (no-op, caller keeps the original file) if nothing was
    requested or the mix failed — same degrade-gracefully pattern as the
    rest of this pipeline."""
    input_args = ["-i", str(clip_path.resolve())]
    filter_parts = ["[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0]"]
    mix_labels = ["[a0]"]
    idx = 1

    if sfx and sfx != "none":
        sfx_path = _ensure_sfx(sfx)
        if sfx_path:
            input_args += ["-i", str(sfx_path.resolve())]
            delay_ms = 0 if sfx_position == "start" else max(0, int((duration - 0.4) * 1000))
            filter_parts.append(
                f"[{idx}:a]adelay=delays={delay_ms}:all=1,"
                f"aformat=sample_rates=44100:channel_layouts=stereo[a{idx}]"
            )
            mix_labels.append(f"[a{idx}]")
            idx += 1

    if typing_sound:
        loop_path = _ensure_typing_loop()
        if loop_path:
            input_args += ["-stream_loop", "-1", "-i", str(loop_path.resolve())]
            filter_parts.append(
                f"[{idx}:a]atrim=0:{duration},aformat=sample_rates=44100:channel_layouts=stereo,volume=0.35[a{idx}]"
            )
            mix_labels.append(f"[a{idx}]")
            idx += 1

    if bg_music_path and bg_music_path.exists():
        input_args += ["-stream_loop", "-1", "-i", str(bg_music_path.resolve())]
        filter_parts.append(
            f"[{idx}:a]atrim=0:{duration},aformat=sample_rates=44100:channel_layouts=stereo,volume=0.16[a{idx}]"
        )
        mix_labels.append(f"[a{idx}]")
        idx += 1

    if idx == 1:
        return False  # nothing requested (or everything failed to prepare)

    filter_parts.append(
        "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0[aout]"
    )
    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + input_args + [
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Audio extras mix failed: {result.stderr[-1500:]}")
        return False
    return True


def _mix_sfx_placements(clip_path: Path, out_path: Path, placements: List[tuple]) -> bool:
    """Mixes N sound-effect files into clip_path's existing audio track,
    each one starting at its own timestamp (via adelay) — the fast path
    behind the timeline editor's click-to-place sound effects. The video
    stream is never touched (`-c:v copy`), so this runs in roughly the
    time it takes to read+remux the audio, regardless of clip length or
    how many effects are placed — no re-cut, re-caption, or re-encode.
    `placements` is a list of (sfx_file_path, start_seconds) tuples,
    already resolved/validated by the caller. A placement past the
    clip's own duration just gets silently cut off by `duration=first`
    below (amix follows the original track's length), not an error."""
    input_args = ["-i", str(clip_path.resolve())]
    filter_parts = ["[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0]"]
    mix_labels = ["[a0]"]
    for idx, (sfx_path, when) in enumerate(placements, start=1):
        input_args += ["-i", str(sfx_path.resolve())]
        delay_ms = max(0, int(when * 1000))
        filter_parts.append(
            f"[{idx}:a]adelay=delays={delay_ms}:all=1,"
            f"aformat=sample_rates=44100:channel_layouts=stereo[a{idx}]"
        )
        mix_labels.append(f"[a{idx}]")

    filter_parts.append(
        "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0[aout]"
    )
    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y"] + input_args + [
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"SFX placement mix failed: {result.stderr[-1500:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def _default_opts() -> dict:
    """Default clip-rendering options — used as a base that callers
    override, so any new option added later only needs a default here
    instead of touching every call site."""
    return dict(
        voiceover=False, voiceover_voice=DEFAULT_VOICE, voiceover_style=DEFAULT_VOICEOVER_STYLE,
        zoom_pan=False, color_preset="none", caption_style=DEFAULT_CAPTION_STYLE,
        sfx="none", sfx_position="end", typing_sound=False, flash_intro=False,
        crop_x=0.0, crop_y=0.0, crop_w=0.0, crop_h=1.0, top_text="", top_text_colors=[],
    )


def _process_source(source: Path, job_dir: Path, job_id: str, clip_count: int, clip_seconds: int,
                     instruction: str, opts: dict, source_url: str = "",
                     download_tier: Optional[str] = None) -> dict:
    """Shared pipeline body for both /process (download-from-URL) and
    /process-upload (user's own file) — everything after "we have a
    source.mp4 on disk" is identical between the two entry points.
    `opts` holds every clip-rendering option (voiceover, zoom/color,
    caption style, sfx, background music, etc) so this signature doesn't
    grow a new positional parameter every time a feature is added.

    download_tier is whatever download_video() returned for this source
    ("cache" | "free" | "own_proxy" | "paid_proxy"), or None for an
    uploaded file that never went through download_video() at all — see
    billable_proxy_bytes below for why this matters."""
    try:
        duration = get_duration(source)
    except Exception as e:
        raise HTTPException(500, f"Video seems invalid: {e}")

    try:
        audio = extract_audio(source, job_dir)
        words = transcribe(audio)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")

    # Persist the transcript + duration so /reclip can regenerate a
    # different moment from this same video later without re-downloading
    # or re-transcribing (both slow, and repeat downloads are part of what
    # triggers YouTube's rate limiting).
    (job_dir / "words.json").write_text(json.dumps(words))
    (job_dir / "meta.json").write_text(json.dumps({"duration": duration, "url": source_url}))

    highlights = pick_highlights(words, clip_count, clip_seconds, duration, instruction)

    if not highlights:
        raise HTTPException(422, "Couldn't find enough speech to build a clip from this video.")

    # Background music is a file uploaded via /upload-music (see
    # find_bg_music) — looked up once per /process call, not once per
    # clip, since every clip in the batch shares the same music track.
    # A fresh job has none yet on its very first render; it's picked up
    # automatically once uploaded, on the next reclip/restyle.
    bg_music_path = find_bg_music(job_dir)

    clips = []
    for i, h in enumerate(highlights):
        ass_path = job_dir / f"clip_{i}.ass"
        words_to_ass(words, h["start"], h["end"], ass_path, caption_style=opts["caption_style"],
                     top_text=opts.get("top_text", ""), top_text_colors=opts.get("top_text_colors", []))

        out_name = f"{job_id}_{i}.mp4"
        out_path = OUTPUT_DIR / out_name

        try:
            cut_and_caption(source, h["start"], h["end"], ass_path, out_path,
                             zoom_pan=opts["zoom_pan"], color_preset=opts["color_preset"],
                             flash_intro=opts["flash_intro"],
                             crop_x=opts.get("crop_x", 0.0), crop_y=opts.get("crop_y", 0.0),
                             crop_w=opts.get("crop_w", 0.0), crop_h=opts.get("crop_h", 1.0))
        except Exception as e:
            raise HTTPException(500, f"Failed to render clip {i}: {e}")

        voiceover_script = apply_voiceover_if_wanted(
            source, job_dir, out_path, h["start"], h["end"], clip_seconds,
            words, opts["voiceover"], opts["voiceover_voice"], instruction, opts["voiceover_style"],
        )

        # Sound effects / typing / background music layer onto whatever
        # audio track is currently on out_path (original speech or the
        # narration voiceover just muxed in above).
        clip_duration = h["end"] - h["start"]
        mixed_path = out_path.with_suffix(".mix.mp4")
        if apply_audio_extras(out_path, mixed_path, clip_duration, opts["sfx"], opts["sfx_position"],
                               opts["typing_sound"], bg_music_path):
            mixed_path.replace(out_path)

        clips.append({
            "file": out_name,
            "start": h["start"],
            "end": h["end"],
            "reason": h.get("reason", ""),
            "url": f"/clips/{out_name}",
            "voiceover_script": voiceover_script,
            "has_voiceover": voiceover_script is not None,
            # Lets the frontend meter free-tier MB usage against the real
            # rendered file instead of guessing from clip duration.
            "size_bytes": out_path.stat().st_size if out_path.exists() else 0,
        })

    # source_bytes/is_youtube are informational (e.g. showing file size in
    # the UI) — billable_proxy_bytes below is what the frontend should
    # actually report to the billing backend as proxy usage.
    #
    # Only a download that genuinely went through YTDLP_PAID_PROXY_URL
    # ("paid_proxy") costs real Decodo money. A cache hit, the free tier
    # (PO token + cookies + client rotation), or your own free proxy all
    # cost $0 — reporting those as proxy usage would eat into the site-wide
    # Decodo dollar budget and each plan's proxy-MB cap for jobs that never
    # touched the paid proxy at all, which is exactly what was happening
    # before this field existed (every YouTube job reported its full
    # download size as "proxy usage" regardless of which tier served it).
    billable_proxy_bytes = (
        source.stat().st_size
        if (download_tier == "paid_proxy" and source.exists())
        else 0
    )
    return {
        "job_id": job_id,
        "clip_count": len(clips),
        "clips": clips,
        "source_bytes": source.stat().st_size if source.exists() else 0,
        "is_youtube": source_url not in ("", "uploaded file"),
        "download_tier": download_tier,
        "billable_proxy_bytes": billable_proxy_bytes,
        # Lets the backend enforce each plan's monthly "minutes of source
        # video" cap (see PLAN_MONTHLY_MINUTES in server.js) — this is the
        # length of the source video that was downloaded/uploaded and
        # transcribed, not the length of the rendered clips.
        "source_minutes": round(duration / 60, 2),
    }


@app.post("/process")
async def process(req: ProcessRequest):
    cleanup_old_jobs()

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # This is the core clip-rendering pipeline (transcription + ffmpeg
    # re-encode + voiceover + audio mix) — by far the heaviest, longest-
    # running work in the whole app, and previously the one major path NOT
    # covered by HEAVY_TASK_LOCK (only the smaller AI Tools endpoints were).
    # That gap is what let a rendering job get OOM-killed (exit code -9)
    # when it landed alongside other heavy work — see /reclip and
    # /process-upload below for the same fix.
    async with HEAVY_TASK_LOCK:
        try:
            source, download_tier = download_video(req.url, job_dir)
        except Exception as e:
            raise HTTPException(400, f"Could not download video: {e}")

        opts = dict(
            voiceover=req.voiceover, voiceover_voice=req.voiceover_voice, voiceover_style=req.voiceover_style,
            zoom_pan=req.zoom_pan, color_preset=req.color_preset, caption_style=req.caption_style,
            sfx=req.sfx, sfx_position=req.sfx_position, typing_sound=req.typing_sound,
            flash_intro=req.flash_intro,
            crop_x=req.crop_x, crop_y=req.crop_y, crop_w=req.crop_w, crop_h=req.crop_h,
            top_text=req.top_text, top_text_colors=req.top_text_colors,
        )
        return _process_source(source, job_dir, job_id, req.clip_count, req.clip_seconds, req.instruction,
                                opts, req.url, download_tier)


# Upload size cap: keeps a single request from blowing out Railway's small
# ephemeral volume / container memory. 500MB comfortably covers a phone
# video several minutes long at 1080p.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    clip_count: int = Form(3),
    clip_seconds: int = Form(30),
    instruction: str = Form(""),
    voiceover: bool = Form(False),
    voiceover_voice: str = Form(DEFAULT_VOICE),
    voiceover_style: str = Form(DEFAULT_VOICEOVER_STYLE),
    zoom_pan: bool = Form(False),
    color_preset: str = Form("none"),
    caption_style: str = Form(DEFAULT_CAPTION_STYLE),
    sfx: str = Form("none"),
    sfx_position: str = Form("end"),
    typing_sound: bool = Form(False),
    flash_intro: bool = Form(False),
    crop_x: float = Form(0.0),
    crop_y: float = Form(0.0),
    crop_w: float = Form(0.0),
    crop_h: float = Form(1.0),
    top_text: str = Form(""),
    top_text_colors: str = Form(""),  # comma-separated, e.g. "white,red,white" — multipart
                                       # forms don't carry real arrays, unlike the JSON endpoints
):
    """Same pipeline as /process, but for a video the user uploads directly
    instead of a YouTube/Twitch/etc URL — skips yt-dlp entirely, so this
    also sidesteps YouTube's bot-detection/rate-limiting altogether."""
    cleanup_old_jobs()

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    source = job_dir / "source.mp4"
    size = 0
    try:
        with open(source, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video is too large (500MB max). Try a shorter or lower-resolution file.")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not read uploaded file: {e}")

    if size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "Uploaded file is empty.")

    opts = dict(
        voiceover=voiceover, voiceover_voice=voiceover_voice, voiceover_style=voiceover_style,
        zoom_pan=zoom_pan, color_preset=color_preset, caption_style=caption_style,
        sfx=sfx, sfx_position=sfx_position, typing_sound=typing_sound,
        flash_intro=flash_intro, crop_x=crop_x, crop_y=crop_y, crop_w=crop_w, crop_h=crop_h,
        top_text=top_text,
        top_text_colors=[c.strip() for c in top_text_colors.split(",") if c.strip()],
    )
    # See /process above — same memory-safety lock around the actual
    # transcription/render/voiceover/mix work, not the upload itself.
    async with HEAVY_TASK_LOCK:
        return _process_source(source, job_dir, job_id, clip_count, clip_seconds, instruction,
                                opts, "uploaded file")


@app.post("/reclip")
async def reclip(req: ReclipRequest):
    """Regenerates one clip from an already-processed video — either at an
    explicit start time, or auto-picked to avoid the time ranges already
    used (exclude_starts). Reuses the source video + transcript saved by
    /process instead of re-downloading, both for speed and to avoid
    hammering YouTube again for a video we already have locally."""
    job_dir = JOBS_DIR / req.job_id
    source = job_dir / "source.mp4"
    words_path = job_dir / "words.json"
    meta_path = job_dir / "meta.json"
    if not (job_dir.exists() and source.exists() and words_path.exists() and meta_path.exists()):
        raise HTTPException(404, "This session has expired — paste the link again to start a new one.")

    words = json.loads(words_path.read_text())
    meta = json.loads(meta_path.read_text())
    duration = meta["duration"]

    if req.start is not None:
        start = max(0.0, min(req.start, max(0.0, duration - req.clip_seconds)))
        end = min(duration, start + req.clip_seconds)
    else:
        exclude_ranges = [(s, s + req.clip_seconds) for s in req.exclude_starts]
        picks = pick_highlights(words, 1, req.clip_seconds, duration, req.instruction, exclude_ranges)
        if not picks:
            raise HTTPException(422, "Couldn't find another distinct moment in this video.")
        start, end = picks[0]["start"], picks[0]["end"]

    ass_path = job_dir / f"reclip_{uuid.uuid4().hex[:6]}.ass"
    words_to_ass(words, start, end, ass_path, caption_style=req.caption_style,
                 top_text=req.top_text, top_text_colors=req.top_text_colors)

    out_name = f"{req.job_id}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = OUTPUT_DIR / out_name

    # Same memory-safety lock as /process — this single "restyle/regenerate"
    # button in the editor is a full re-render (cut + caption + voiceover +
    # audio mix), not a cheap operation, and was previously unprotected.
    async with HEAVY_TASK_LOCK:
        try:
            cut_and_caption(source, start, end, ass_path, out_path,
                             zoom_pan=req.zoom_pan, color_preset=req.color_preset, flash_intro=req.flash_intro,
                             crop_x=req.crop_x, crop_y=req.crop_y, crop_w=req.crop_w, crop_h=req.crop_h)
        except Exception as e:
            raise HTTPException(500, f"Failed to render clip: {e}")

        voiceover_script = apply_voiceover_if_wanted(
            source, job_dir, out_path, start, end, req.clip_seconds,
            words, req.voiceover, req.voiceover_voice, req.instruction, req.voiceover_style,
        )

        bg_music_path = find_bg_music(job_dir)
        mixed_path = out_path.with_suffix(".mix.mp4")
        if apply_audio_extras(out_path, mixed_path, end - start, req.sfx, req.sfx_position,
                               req.typing_sound, bg_music_path):
            mixed_path.replace(out_path)

    return {
        "file": out_name,
        "start": start,
        "end": end,
        "url": f"/clips/{out_name}",
        "voiceover_script": voiceover_script,
        "has_voiceover": voiceover_script is not None,
        "size_bytes": out_path.stat().st_size if out_path.exists() else 0,
    }


@app.post("/apply-sound-effects")
async def apply_sound_effects(req: ApplySoundEffectsRequest):
    """Stamps one or more sound effects onto an already-rendered clip's
    audio track at specific timestamps — the fast path behind the
    editor's timeline: click a sound effect, click anywhere on the
    timeline to drop it (as many times, with as many different effects,
    as wanted). This does NOT re-cut, re-caption, or re-run voiceover —
    only the audio track is touched (`-c:v copy` in _mix_sfx_placements),
    so it's quick regardless of clip length, unlike /reclip which is a
    full re-render.

    The first call for a given clip stashes a copy of its then-current
    audio as clip.orig.mp4 — every call after that (including ones that
    change or remove placements) re-mixes from that clean baseline
    instead of layering onto whatever the previous mix left behind, so
    the timeline's placements always fully REPLACE the clip's effects,
    never stack on top of earlier ones. Passing an empty placements list
    restores that clean baseline (i.e. "remove all sound effects")."""
    clip_path = OUTPUT_DIR / req.clip_file
    if not clip_path.exists():
        raise HTTPException(404, "That clip no longer exists on the server.")
    if len(req.placements) > MAX_SFX_PLACEMENTS:
        raise HTTPException(400, f"Too many sound effects at once (max {MAX_SFX_PLACEMENTS}).")

    orig_path = clip_path.with_suffix(".orig.mp4")
    if not orig_path.exists():
        shutil.copyfile(clip_path, orig_path)

    if not req.placements:
        async with HEAVY_TASK_LOCK:
            shutil.copyfile(orig_path, clip_path)
        # This endpoint just overwrote clip_path — /apply-gameplay-bg's own
        # "clean" baseline (clip.novideobg.mp4) would now be stale if it
        # predates this change, so drop it and let it re-snapshot fresh
        # next time that feature is touched. See the matching comment in
        # /apply-gameplay-bg for why both directions do this.
        clip_path.with_suffix(".novideobg.mp4").unlink(missing_ok=True)
        return {"file": req.clip_file, "url": f"/clips/{req.clip_file}"}

    resolved = []
    for p in req.placements:
        sfx_path = _ensure_sfx(p.effect)
        if not sfx_path:
            raise HTTPException(400, f"Unknown or unavailable sound effect: {p.effect}")
        resolved.append((sfx_path, max(0.0, p.time)))

    out_tmp = clip_path.with_suffix(".sfxmix.mp4")
    async with HEAVY_TASK_LOCK:
        ok = _mix_sfx_placements(orig_path, out_tmp, resolved)
        if not ok:
            raise HTTPException(500, "Failed to mix sound effects into the clip.")
        out_tmp.replace(clip_path)

    clip_path.with_suffix(".novideobg.mp4").unlink(missing_ok=True)
    return {"file": req.clip_file, "url": f"/clips/{req.clip_file}"}


@app.post("/apply-gameplay-bg")
async def apply_gameplay_bg(req: ApplyGameplayBgRequest):
    """Toggles the split-screen gameplay background on or off for an
    already-rendered clip — the fast path behind the editor's "Background
    gameplay" checkbox. Like /apply-sound-effects, this works on the
    finished clip directly instead of going through a full /reclip
    render: no re-cut, re-caption, or re-voiceover, just one ffmpeg
    composite pass (see apply_gameplay_split above).

    The first call for a given clip stashes a copy of its then-current
    state as clip.novideobg.mp4 ("no video background") — every later
    call, on or off, works from that same clean baseline, so toggling
    back and forth never stacks split-screens on split-screens or loses
    the original framing. enabled=false just restores that baseline.

    Requires a gameplay background already uploaded for this job via
    /upload-gameplay-bg — this is deliberately never automatic or
    on-by-default; it's an explicit per-clip opt-in."""
    clip_path = OUTPUT_DIR / req.clip_file
    if not clip_path.exists():
        raise HTTPException(404, "That clip no longer exists on the server.")

    base_path = clip_path.with_suffix(".novideobg.mp4")
    if not base_path.exists():
        shutil.copyfile(clip_path, base_path)

    if not req.enabled:
        async with HEAVY_TASK_LOCK:
            shutil.copyfile(base_path, clip_path)
        # Mirror of the invalidation in /apply-sound-effects — this
        # endpoint just overwrote clip_path, so the sfx feature's own
        # baseline needs to re-snapshot fresh next time it's used too.
        clip_path.with_suffix(".orig.mp4").unlink(missing_ok=True)
        return {"file": req.clip_file, "url": f"/clips/{req.clip_file}", "gameplay_bg_enabled": False}

    # Two sources for the background footage: a named preset (committed
    # to gameplay_bg/, picked from the frontend's button row — no upload
    # needed), or the job's own custom upload if no preset was given.
    # Gameplay backgrounds are stored per JOB, not per clip, but this
    # endpoint only receives a clip_file — job_id isn't threaded through
    # OUTPUT_DIR filenames the way it is for job_dir, so it's parsed off
    # the front of the clip filename (same "{job_id}_{suffix}.mp4" naming
    # every clip is already given in _process_source/reclip).
    bg_path = None
    if req.preset:
        bg_path = resolve_gameplay_bg_preset(req.preset)
        if not bg_path:
            raise HTTPException(400, f"Unknown gameplay background preset: {req.preset}")
    else:
        job_id = req.clip_file.split("_")[0]
        job_dir = JOBS_DIR / job_id
        bg_path = find_gameplay_bg(job_dir) if job_dir.exists() else None
        if not bg_path:
            raise HTTPException(400, "No gameplay background uploaded for this project yet — pick a preset, or upload your own first.")

    out_tmp = clip_path.with_suffix(".gpbg.mp4")
    async with HEAVY_TASK_LOCK:
        ok = apply_gameplay_split(base_path, out_tmp, bg_path)
        if not ok:
            raise HTTPException(500, "Failed to composite the gameplay background onto this clip.")
        out_tmp.replace(clip_path)

    clip_path.with_suffix(".orig.mp4").unlink(missing_ok=True)
    return {"file": req.clip_file, "url": f"/clips/{req.clip_file}", "gameplay_bg_enabled": True}


# Upload size cap for background music — generous for a compressed audio
# track (a 20MB mp3 is well over 20 minutes), tiny next to MAX_UPLOAD_BYTES
# for video.
BG_MUSIC_MAX_BYTES = 20 * 1024 * 1024


@app.post("/upload-music")
async def upload_music(job_id: str = Form(...), file: UploadFile = File(...)):
    """Saves a user-uploaded background-music file for an existing job.
    Replaces the old paste-a-YouTube-link approach, which needed a
    yt-dlp download subject to the same bot-detection as video downloads.
    Stored once per job as job_dir/bgmusic.<ext> (see find_bg_music) and
    picked up automatically by every /reclip call afterward, since every
    clip in a job shares the same track — call /reclip (or let the
    frontend's debounced restyle do it) right after this to actually
    re-render the current clip with the new track mixed in."""
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "This session has expired — generate a clip again first.")

    for old in job_dir.glob("bgmusic.*"):
        old.unlink(missing_ok=True)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in BG_MUSIC_EXTS:
        ext = ".mp3"
    dest = job_dir / f"bgmusic{ext}"
    size = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > BG_MUSIC_MAX_BYTES:
                    raise HTTPException(413, "Music file is too large (20MB max).")
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read uploaded file: {e}")

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty.")

    return {"ok": True}


@app.post("/remove-music")
def remove_music(job_id: str = Form(...)):
    """Clears a job's background-music track — the next /reclip call will
    render with no music mixed in again."""
    job_dir = JOBS_DIR / job_id
    for old in job_dir.glob("bgmusic.*"):
        old.unlink(missing_ok=True)
    return {"ok": True}


# Upload size cap for a gameplay background clip — video, so a much
# bigger allowance than background music. Generous enough for a couple
# minutes of 1080p footage; the split-screen composite only ever uses as
# much of it as the current clip's own length anyway (see
# apply_gameplay_split's random-start-offset + loop logic).
GAMEPLAY_BG_MAX_BYTES = 300 * 1024 * 1024


@app.post("/upload-gameplay-bg")
async def upload_gameplay_bg(job_id: str = Form(...), file: UploadFile = File(...)):
    """Saves a user-uploaded gameplay clip (Subway Surfers, an obby, etc.)
    for the split-screen background feature — this app doesn't ship or
    fetch any gameplay footage itself, it's entirely whatever the user
    provides. Stored once per job as job_dir/gameplaybg.<ext> (see
    find_gameplay_bg), reused by /apply-gameplay-bg for every clip in the
    job. Uploading a new file here just replaces the old one; it doesn't
    enable the split-screen effect on anything by itself — that's a
    separate, explicit opt-in per clip."""
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "This session has expired — generate a clip again first.")

    for old in job_dir.glob("gameplaybg.*"):
        old.unlink(missing_ok=True)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in GAMEPLAY_BG_EXTS:
        ext = ".mp4"
    dest = job_dir / f"gameplaybg{ext}"
    size = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > GAMEPLAY_BG_MAX_BYTES:
                    raise HTTPException(413, "Gameplay clip is too large (300MB max).")
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read uploaded file: {e}")

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded file is empty.")

    return {"ok": True}


@app.post("/remove-gameplay-bg")
def remove_gameplay_bg(job_id: str = Form(...)):
    """Deletes a job's uploaded gameplay clip. Doesn't touch any clip
    that already has the split-screen effect applied — remove it from an
    individual clip first via /apply-gameplay-bg (enabled=false) if
    needed."""
    job_dir = JOBS_DIR / job_id
    for old in job_dir.glob("gameplaybg.*"):
        old.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/download-social-video")
async def download_social_video_endpoint(url: str = Form(...)):
    """Downloads a video straight from a YouTube/TikTok/etc link — no
    clipping, transcription, or captioning, just the raw source file, for
    the standalone Download Social Videos tool. Reuses download_video()'s
    existing YouTube bot-detection handling (multi-client fallback, PO
    token, optional cookies) — those extractor-args are simply ignored by
    yt-dlp for non-YouTube sites, so this works unmodified for TikTok and
    anything else yt-dlp supports."""
    url = url.strip()
    if not url:
        raise HTTPException(400, "Paste a video link first.")

    work_dir = TOOLS_DIR / uuid.uuid4().hex[:8]
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with HEAVY_TASK_LOCK:
            try:
                source, _download_tier = download_video(url, work_dir)
            except Exception as e:
                raise HTTPException(400, f"Could not download that video: {e}")

            out_name = f"download_{uuid.uuid4().hex[:8]}.mp4"
            shutil.copy(source, OUTPUT_DIR / out_name)
            return {"url": f"/clips/{out_name}", "size_bytes": (OUTPUT_DIR / out_name).stat().st_size}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


@app.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Removes the background from an uploaded image using rembg — fully
    self-hosted (U^2-Net model baked into the Docker image, see Dockerfile),
    so this has no API key, no per-call cost, and no rate limit. Returns a
    transparent PNG.

    Video background removal isn't wired up yet: doing it at real quality
    means running this per-frame and re-encoding, which is too slow for a
    single request/response on a CPU-only container — that's a follow-up,
    not something silently half-done here."""
    if rembg_remove is None:
        raise HTTPException(501, "Background removal isn't available on this server (rembg failed to load).")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image is too large (15MB max).")

    try:
        async with HEAVY_TASK_LOCK:
            result = rembg_remove(raw, session=get_rembg_session())
    except Exception as e:
        raise HTTPException(500, f"Background removal failed: {e}")

    out_name = f"bgremoved_{uuid.uuid4().hex[:8]}.png"
    (OUTPUT_DIR / out_name).write_bytes(result)
    return {"url": f"/clips/{out_name}"}


@app.post("/remove-vocals")
async def remove_vocals(file: UploadFile = File(...)):
    """Splits an uploaded audio or video file into vocals + instrumental
    tracks using Demucs (self-hosted, htdemucs model baked into the
    Docker image — no API key, no cost). CPU-only inference, so a full
    song can take a minute or two; that's expected, not a hang."""
    work_dir = TOOLS_DIR / uuid.uuid4().hex[:8]
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_name = file.filename or "input.audio"
        src_ext = Path(raw_name).suffix or ".mp3"
        src_path = work_dir / f"src{src_ext}"
        size = 0
        try:
            with open(src_path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 100 * 1024 * 1024:
                        raise HTTPException(413, "File is too large (100MB max).")
                    f.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Could not read uploaded file: {e}")
        if size == 0:
            raise HTTPException(400, "Uploaded file is empty.")

        # Demucs wants an audio file — pull the audio track out first with
        # ffmpeg (already in this container for clip rendering) if a video
        # was uploaded instead.
        audio_path = src_path
        if src_ext.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
            audio_path = work_dir / "audio.wav"
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_path), "-vn", "-acodec", "pcm_s16le", str(audio_path)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                raise HTTPException(500, f"Could not extract audio from that video: {result.stderr[-800:]}")

        out_dir = work_dir / "out"
        # --segment caps how much audio Demucs holds in memory at once —
        # without it, peak memory scales with the whole track's length,
        # which is what was pushing this container past its 1GB ceiling.
        # -j 1 keeps it to a single worker instead of spinning up parallel
        # copies of the model. Both trade a bit of speed for a lot less RAM.
        async with HEAVY_TASK_LOCK:
            result = subprocess.run(
                ["python3", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
                 "--segment", "8", "-j", "1", "-o", str(out_dir), str(audio_path)],
                capture_output=True, text=True, timeout=600,
            )
        if result.returncode != 0:
            raise HTTPException(500, f"Vocal separation failed: {result.stderr[-1000:]}")

        stem_dir = out_dir / "htdemucs" / audio_path.stem
        vocals_src = stem_dir / "vocals.wav"
        instrumental_src = stem_dir / "no_vocals.wav"
        if not (vocals_src.exists() and instrumental_src.exists()):
            raise HTTPException(500, "Vocal separation finished but the output files are missing.")

        vocals_name = f"vocals_{uuid.uuid4().hex[:8]}.wav"
        instrumental_name = f"instrumental_{uuid.uuid4().hex[:8]}.wav"
        shutil.copy(vocals_src, OUTPUT_DIR / vocals_name)
        shutil.copy(instrumental_src, OUTPUT_DIR / instrumental_name)
        return {"vocals_url": f"/clips/{vocals_name}", "instrumental_url": f"/clips/{instrumental_name}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/enhance-speech")
async def enhance_speech(file: UploadFile = File(...)):
    """Denoises/cleans up an uploaded voice recording using DeepFilterNet
    (self-hosted — model ships inside the pip package, no download at
    request time, no API key, no cost). DeepFilterNet only accepts 48kHz
    mono wav, so the input is resampled with ffmpeg first regardless of
    what format it was uploaded in."""
    work_dir = TOOLS_DIR / uuid.uuid4().hex[:8]
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "Uploaded file is empty.")
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(413, "Audio file is too large (50MB max).")

        src_path = work_dir / "src.audio"
        src_path.write_bytes(raw)

        wav_path = work_dir / "input.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path), "-ar", "48000", "-ac", "1", str(wav_path)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Could not read that audio file: {result.stderr[-800:]}")

        out_dir = work_dir / "out"
        out_dir.mkdir(exist_ok=True)
        async with HEAVY_TASK_LOCK:
            result = subprocess.run(
                ["deepFilter", str(wav_path), "--output-dir", str(out_dir)],
                capture_output=True, text=True, timeout=180,
            )
        if result.returncode != 0:
            raise HTTPException(500, f"Speech enhancement failed: {result.stderr[-1000:]}")

        enhanced_src = out_dir / wav_path.name
        if not enhanced_src.exists():
            raise HTTPException(500, "Enhancement finished but the output file is missing.")

        out_name = f"enhanced_{uuid.uuid4().hex[:8]}.wav"
        shutil.copy(enhanced_src, OUTPUT_DIR / out_name)
        return {"url": f"/clips/{out_name}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/synthesize-voice")
async def synthesize_voice(script: str = Form(...), voice: str = Form(DEFAULT_VOICE)):
    """Standalone text-to-speech — the same provider chain used inside the
    clip editor (Google TTS / ElevenLabs / self-hosted Piper), but without
    needing a job or a clip first. Powers the standalone AI Voiceovers
    tool on the tools page."""
    if not script.strip():
        raise HTTPException(400, "Script can't be empty.")
    if len(script) > 5000:
        raise HTTPException(400, "Script is too long (5000 characters max).")

    work_dir = TOOLS_DIR / uuid.uuid4().hex[:8]
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with HEAVY_TASK_LOCK:
            audio_path = synthesize_voiceover(script, voice, work_dir)
            if not audio_path:
                raise HTTPException(502, "Voice synthesis failed on every configured provider (Google/ElevenLabs/Piper) — check the pipeline service logs.")

            out_name = f"voice_{uuid.uuid4().hex[:8]}{audio_path.suffix}"
            shutil.copy(audio_path, OUTPUT_DIR / out_name)
            return {"url": f"/clips/{out_name}"}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


async def _generate_image_pollinations(prompt: str) -> Optional[bytes]:
    """Fallback image generator. Pollinations.ai is a free, keyless,
    community-run image API (Flux-based) with no per-minute quota like
    Gemini's — used when Gemini's shared free tier is exhausted so AI Images
    doesn't just go down for everyone until the quota resets. Best-effort:
    returns None on any failure so the caller falls back to the original
    Gemini error instead of masking it with a worse, less specific one."""
    try:
        encoded = urllib.parse.quote(prompt, safe="")
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.get(
                f"https://image.pollinations.ai/prompt/{encoded}",
                params={"width": 1024, "height": 1024, "nologo": "true", "model": "flux"},
            )
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return resp.content
    except Exception:
        pass
    return None


@app.post("/generate-image")
async def generate_image(prompt: str = Form(...)):
    """Generates an image from a text prompt via Google's Gemini API
    (gemini-2.5-flash-image, aka "Nano Banana") — free tier, no card on
    file. Not self-hosted like the tools above (there's no realistic
    free/CPU-only substitute for text-to-image), so this depends on
    GEMINI_API_KEY being set and on Google's free-tier quota holding up."""
    if not GEMINI_API_KEY:
        raise HTTPException(501, "Image generation isn't configured on this server (no GEMINI_API_KEY).")
    if not prompt.strip():
        raise HTTPException(400, "Prompt can't be empty.")

    try:
        async with HEAVY_TASK_LOCK:
            # Waiting inside the lock means two concurrent requests queue up
            # in strict arrival order and each waits its fair turn, instead
            # of both waiting the same interval and racing each other for
            # who actually gets to fire first.
            await _wait_for_gemini_turn()
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent",
                    headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
    except Exception as e:
        raise HTTPException(502, f"Could not reach the image-generation service: {e}")
    finally:
        gc.collect()

    if resp.status_code != 200:
        # Gemini's free tier is a small SHARED quota across every user on
        # this site, both per-minute and per-day — a failure here often just
        # means someone (possibly several people) recently used it up, not
        # that anything is broken. Try Pollinations.ai (free, keyless, no
        # per-minute cap) before giving up, so AI Images stays usable instead
        # of going down site-wide until Gemini's quota resets.
        fallback_bytes = await _generate_image_pollinations(prompt)
        if fallback_bytes:
            out_name = f"image_{uuid.uuid4().hex[:8]}.png"
            (OUTPUT_DIR / out_name).write_bytes(fallback_bytes)
            del fallback_bytes
            return {"url": f"/clips/{out_name}"}

        if resp.status_code == 429:
            # Extract Google's suggested retry delay if it sent one.
            retry_after = None
            try:
                detail = resp.json()
                for err in detail.get("error", {}).get("details", []):
                    if err.get("@type", "").endswith("RetryInfo"):
                        retry_after = err.get("retryDelay")
            except Exception:
                pass
            wait_msg = f" Try again in about {retry_after}." if retry_after else " Try again in a minute."
            raise HTTPException(429, f"AI Images is rate-limited right now (this runs on a shared free quota).{wait_msg}")
        raise HTTPException(502, f"Image generation failed ({resp.status_code}): {resp.text[-800:]}")

    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        image_b64 = next(p["inlineData"]["data"] for p in parts if "inlineData" in p)
    except (KeyError, IndexError, StopIteration):
        raise HTTPException(502, "Image generation returned no image — the prompt may have been blocked by safety filters.")

    out_name = f"image_{uuid.uuid4().hex[:8]}.png"
    (OUTPUT_DIR / out_name).write_bytes(base64.b64decode(image_b64))
    del data, image_b64
    return {"url": f"/clips/{out_name}"}


@app.post("/brainstorm-ideas")
async def brainstorm_ideas(topic: str = Form(...), idea_type: str = Form("Story Video")):
    """Generates a short list of content ideas via Groq's free-tier chat
    API (Llama 3.3 70B) instead of Anthropic — same zero-Claude-spend
    philosophy as the transcript-based voiceover fallback."""
    if not GROQ_API_KEY:
        raise HTTPException(501, "Content idea brainstorming isn't configured on this server (no GROQ_API_KEY).")
    if not topic.strip():
        raise HTTPException(400, "Topic can't be empty.")

    system_prompt = (
        "You generate short-form video content ideas for a creator, in the style of a "
        f'"{idea_type}". Reply with ONLY a JSON array of 6 short idea strings — no other text, '
        "no markdown, no numbering. Each idea should be a single punchy sentence a creator "
        "could film today."
    )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Topic: {topic}"},
                    ],
                    "temperature": 0.9,
                },
            )
    except Exception as e:
        raise HTTPException(502, f"Could not reach the idea-generation service: {e}")

    if resp.status_code != 200:
        raise HTTPException(502, f"Idea generation failed ({resp.status_code}): {resp.text[-800:]}")

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # Models occasionally wrap JSON in a markdown code fence despite being
    # told not to — strip that before parsing rather than failing outright.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        ideas = json.loads(raw)
        if not isinstance(ideas, list):
            raise ValueError("not a list")
    except Exception:
        raise HTTPException(502, "Idea generation returned an unexpected format — try again.")

    return {"ideas": [str(i) for i in ideas][:10]}


def _did_auth_header() -> str:
    return "Basic " + base64.b64encode(DID_API_KEY.encode()).decode()


@app.post("/generate-avatar-video")
async def generate_avatar_video(script: str = Form(...), image: UploadFile = File(...)):
    """Generates a talking-avatar video via D-ID: an uploaded photo is
    animated to lip-sync the given script. D-ID needs a public URL for
    the source photo (not a raw upload), so the photo is saved to this
    service's own /clips mount first and referenced by its public URL.
    Generation is async on D-ID's side (POST returns immediately with a
    "created" status), so this polls until it's done or times out.
    D-ID's free tier is ~5 min of video/month, tracked on D-ID's own
    account dashboard — not something this endpoint meters itself."""
    if not DID_API_KEY:
        raise HTTPException(501, "AI Videos isn't configured on this server (no DID_API_KEY).")
    if not script.strip():
        raise HTTPException(400, "Script can't be empty.")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Uploaded photo is empty.")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "Photo is too large (10MB max).")

    ext = Path(image.filename or "").suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    photo_name = f"avatar_src_{uuid.uuid4().hex[:8]}{ext}"
    (OUTPUT_DIR / photo_name).write_bytes(raw)
    del raw
    photo_url = f"{PIPELINE_PUBLIC_URL}/clips/{photo_name}"

    try:
        async with HEAVY_TASK_LOCK:
            async with httpx.AsyncClient(timeout=30) as client:
                create_resp = await client.post(
                    "https://api.d-id.com/talks",
                    headers={"Authorization": _did_auth_header(), "Content-Type": "application/json"},
                    json={"source_url": photo_url, "script": {"type": "text", "input": script}},
                )
    except Exception as e:
        raise HTTPException(502, f"Could not reach D-ID: {e}")
    finally:
        gc.collect()

    if create_resp.status_code == 451 or "moderation" in create_resp.text.lower():
        # D-ID's automated moderation flags a photo before generation even
        # starts — this is a real per-photo rejection (often a false
        # positive on ordinary photos), not something retrying fixes.
        raise HTTPException(451, "D-ID's automatic content moderation rejected this photo — try a different, clearer photo of a face. This isn't a quota or server issue, just this specific image.")
    if create_resp.status_code not in (200, 201):
        raise HTTPException(502, f"D-ID rejected the request ({create_resp.status_code}): {create_resp.text[-500:]}")

    talk_id = create_resp.json().get("id")
    if not talk_id:
        raise HTTPException(502, "D-ID didn't return a talk ID.")

    # Poll every 3s for up to ~4.5 minutes — comfortably under the 5-minute
    # client-side fetch timeout tools.html uses for every tool on this page.
    result_url = None
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(90):
            await asyncio.sleep(3)
            poll = await client.get(f"https://api.d-id.com/talks/{talk_id}", headers={"Authorization": _did_auth_header()})
            if poll.status_code != 200:
                continue
            data = poll.json()
            status = data.get("status")
            if status == "done":
                result_url = data.get("result_url")
                break
            if status == "error":
                raise HTTPException(502, f"D-ID video generation failed: {data.get('error', {})}")

    if not result_url:
        raise HTTPException(504, "D-ID is taking longer than expected to render this video — try again in a bit.")

    return {"video_url": result_url}


@app.post("/regenerate-voiceover")
def regenerate_voiceover(req: RegenerateVoiceoverRequest):
    """Re-synthesizes a clip's narration from user-edited script text,
    without re-cutting or re-captioning the underlying clip — this is what
    makes the voice-over feature "editable" rather than one-shot."""
    job_dir = JOBS_DIR / req.job_id
    clip_path = OUTPUT_DIR / req.clip_file
    if not job_dir.exists():
        raise HTTPException(404, "This session has expired — paste the link again to start a new one.")
    if not clip_path.exists():
        raise HTTPException(404, "That clip no longer exists on the server.")
    if not req.script.strip():
        raise HTTPException(400, "Script can't be empty.")

    vo_audio = synthesize_voiceover(req.script, req.voice, job_dir)
    if not vo_audio:
        raise HTTPException(502, "Voice synthesis failed on every configured provider (Google/ElevenLabs/Piper) — check the pipeline service logs.")
    muxed = clip_path.with_suffix(".vo.mp4")
    if not mux_voiceover(clip_path, vo_audio, muxed):
        raise HTTPException(500, "Failed to combine the new narration with the clip.")
    muxed.replace(clip_path)

    # /apply-sound-effects and /apply-gameplay-bg each cache a "clean"
    # baseline the first time they're used on a clip, and work from that
    # baseline on every later toggle. Both are now stale (they predate
    # this new narration) — drop them so whichever is touched next
    # re-snapshots fresh audio (with the new voice-over included) instead
    # of silently reverting to the old one.
    clip_path.with_suffix(".orig.mp4").unlink(missing_ok=True)
    clip_path.with_suffix(".novideobg.mp4").unlink(missing_ok=True)

    return {"file": req.clip_file, "url": f"/clips/{req.clip_file}", "voiceover_script": req.script, "has_voiceover": True}


@app.get("/voices")
def list_voices():
    return {"voices": list(ELEVENLABS_VOICES.keys()), "default": DEFAULT_VOICE}


@app.get("/color-presets")
def list_color_presets():
    return {"presets": COLOR_PRESETS, "default": "none", "premium": sorted(PREMIUM_COLOR_PRESETS)}


@app.get("/caption-styles")
def list_caption_styles():
    return {
        "styles": list(CAPTION_STYLES.keys()),
        "default": DEFAULT_CAPTION_STYLE,
        "premium": sorted(PREMIUM_CAPTION_STYLES),
    }


@app.get("/voiceover-styles")
def list_voiceover_styles():
    return {"styles": list(VOICEOVER_STYLES.keys()), "default": DEFAULT_VOICEOVER_STYLE}


@app.get("/sound-effects")
def list_sound_effects():
    return {"effects": SOUND_EFFECTS, "default": "none"}


@app.get("/gameplay-backgrounds")
def list_gameplay_backgrounds():
    """Built-in split-screen gameplay presets — whatever's currently
    committed under gameplay_bg/ in the repo (see list_gameplay_bg_presets).
    Purely additive to fetch: dropping a new clip in that folder and
    committing it makes it show up here automatically, no other endpoint
    or frontend change required."""
    return {"presets": list_gameplay_bg_presets()}


@app.get("/health")
def health():
    voiceover_providers = []
    if GOOGLE_TTS_API_KEY:
        voiceover_providers.append("google")
    if ELEVENLABS_API_KEY:
        voiceover_providers.append("elevenlabs")
    # Piper is baked into the image (see Dockerfile) and needs no API key,
    # so it's always available as the last-resort fallback — but if the
    # model files somehow didn't get downloaded during the build, report
    # that honestly instead of claiming a provider that isn't really there.
    if PIPER_VOICES_DIR.exists() and any(PIPER_VOICES_DIR.glob("*.onnx")):
        voiceover_providers.append("piper (self-hosted)")
    return {
        "status": "ok",
        "transcription": "deepgram" if DEEPGRAM_API_KEY else "not configured",
        "highlight_picking": "claude" if ANTHROPIC_API_KEY else "heuristic fallback",
        "youtube_cookies": "configured" if YTDLP_COOKIES else "not set (may hit YouTube bot checks)",
        "pot_provider": "configured" if POT_PROVIDER_URL else "not set",
        "own_proxy_fallback": "configured" if YTDLP_OWN_PROXY_URL else "not set",
        "paid_proxy_fallback": "configured" if YTDLP_PAID_PROXY_URL else "not set",
        "voiceover": "+".join(voiceover_providers) if voiceover_providers else "not configured",
        "direct_upload": "enabled",
        "zoom_pan": "enabled",
        "color_grading": "enabled",
        "caption_styles": "enabled",
        "sound_effects": "enabled",
        "background_music": "enabled",
        "fades": "enabled",
        # Self-hosted tools — checked directly instead of just assumed, so
        # a deploy that's missing a dependency reports honestly instead of
        # claiming a tool that isn't really there.
        "background_remover": "rembg (self-hosted)" if rembg_remove is not None else "unavailable",
        "vocal_remover": "demucs (self-hosted)" if importlib.util.find_spec("demucs") else "unavailable",
        "speech_enhancer": "deepfilternet (self-hosted)" if shutil.which("deepFilter") else "unavailable",
        "ai_images": "gemini" if GEMINI_API_KEY else "not configured",
        "content_ideas": "groq" if GROQ_API_KEY else "not configured",
        "social_video_download": "enabled",
        "ai_videos": "d-id" if DID_API_KEY else "not configured",
    }
