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

import base64
import os
import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

# URL of a self-hosted bgutil-ytdlp-pot-provider instance (see
# https://github.com/Brainicism/bgutil-ytdlp-pot-provider) — generates
# proof-of-origin tokens that help yt-dlp's traffic look legitimate to
# YouTube from a datacenter IP. Not a guaranteed bypass (YouTube's own docs
# say so), but a free, real improvement with no account/cookies required.
POT_PROVIDER_URL = os.environ.get("POT_PROVIDER_URL", "").rstrip("/")

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
DEFAULT_VOICE = "Rachel"


# One-click color grading presets, applied at render time via ffmpeg's eq/
# colorbalance/hue filters — no separate render pass, just extra filter
# chain stages before the caption burn-in.
COLOR_PRESETS = ["none", "warm", "moody", "vibrant", "bw", "cinematic"]

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
}
DEFAULT_CAPTION_STYLE = "bold"

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

# Synthesized (not licensed/downloaded) sound effects — generated once via
# ffmpeg's lavfi audio sources and cached to disk, so there's no external
# provider, API key, or copyright question involved.
SOUND_EFFECTS = ["none", "ting", "pop", "whoosh", "click"]
SFX_DIR = Path(__file__).parent / "sfx"
SFX_DIR.mkdir(exist_ok=True)


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
    bg_music_url: str = ""             # YouTube/YouTube Music link, mixed in at low volume
    flash_intro: bool = False          # white flash-in + shutter click instead of a plain fade


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
    bg_music_url: str = ""
    flash_intro: bool = False


class RegenerateVoiceoverRequest(BaseModel):
    job_id: str
    clip_file: str        # the "file" value from a previous clip response, e.g. "70b799b2_0.mp4"
    script: str            # user-edited narration text
    voice: str = DEFAULT_VOICE


# ---------------------------------------------------------------------------
# Download + audio extraction
# ---------------------------------------------------------------------------
def download_video(url: str, dest: Path) -> Path:
    """Downloads the source video with yt-dlp (installed separately, see README).

    Format selector: modern YouTube usually serves video and audio as
    separate streams rather than one combined file, so a bare '-f mp4'
    often fails or silently grabs a low-quality legacy stream. This
    selector asks for the best available mp4 video + m4a audio and merges
    them, falling back to the best combined stream if that's unavailable
    for a given video. --merge-output-format mp4 forces the merged result
    into an actual .mp4 container regardless of the source formats.

    --remote-components ejs:github --js-runtimes deno: YouTube's player JS
    is obfuscated, and yt-dlp needs to execute it to derive the signature
    used in download URLs. Requires Deno installed and on PATH (see
    README) — without it, downloads fail with a JS-runtime error.
    """
    out_path = dest / "source.mp4"

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
            return out_path
        last_error = result.stderr[-1200:]
        print(f"yt-dlp attempt with player_client={client} failed, trying next client if any:\n{last_error}")

    hint = (
        "\n\nYouTube is blocking downloads from this server (common for cloud-hosted "
        "IPs, even with a PO token provider and cookies configured — YouTube doesn't "
        "guarantee either bypasses its bot checks). Use the \"Upload your own video\" "
        "option instead — it skips YouTube entirely and always works."
    )
    raise RuntimeError(f"yt-dlp failed on all player clients: {last_error}{hint}")


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


def words_to_ass(words, clip_start: float, clip_end: float, ass_path: Path,
                  chunk_size: Optional[int] = None, caption_style: str = DEFAULT_CAPTION_STYLE):
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

    PlayResX/Y match the 1080x1920 output frame so font sizes and margins
    line up correctly after the crop+scale filter runs.
    """
    style = CAPTION_STYLES.get(caption_style, CAPTION_STYLES[DEFAULT_CAPTION_STYLE])
    if chunk_size is None:
        chunk_size = style["chunk_size"]

    clip_words = [w for w in words if clip_start <= w["start"] < clip_end]

    # Colours are &HAABBGGRR. Outline/Shadow/Bold/Fontname/Fontsize all
    # come from the selected style so each preset actually looks distinct.
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
        f"{style['shadow']},2,60,60,190,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # group into on-screen lines of `chunk_size` words each
    chunks = [clip_words[i:i + chunk_size] for i in range(0, len(clip_words), chunk_size)]

    POP_MS = 90      # how long the scale-up half of each word's pop takes
    POP_SCALE = 110  # smaller bump leaves margin before a line could ever
                      # overflow its row, on top of the WrapStyle fix above
    FADE_MS = 150    # in/out fade duration for "fade"-style presets

    lines = [header]
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
    }
    return presets.get(preset, "")


def cut_and_caption(source: Path, start: float, end: float, ass_path: Path, out_path: Path,
                     zoom_pan: bool = False, color_preset: str = "none", flash_intro: bool = False):
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

    vf_stages = [
        "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'",
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
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-threads", "2",
            "-c:a", "aac", str(out_path.resolve())]
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
# Voice-over — Claude (vision) writes the script, ElevenLabs speaks it
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


def synthesize_voiceover(script: str, voice_name: str, dest: Path) -> Optional[Path]:
    """Calls ElevenLabs to turn a script into speech. Returns None (never
    raises) on any failure — same "degrade gracefully" pattern as the rest
    of this pipeline, since voice-over is always an enhancement, never a
    requirement for a clip to come back successfully."""
    if not ELEVENLABS_API_KEY or not script.strip():
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
    if not ELEVENLABS_API_KEY:
        if force:
            print("Voiceover requested but ELEVENLABS_API_KEY is not set — skipping.")
        return None

    try:
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
def _ensure_sfx(name: str) -> Optional[Path]:
    """Synthesizes (once, then caches to disk) a short sound effect purely
    via ffmpeg's lavfi audio sources — no external provider, download, or
    licensing question involved, since nothing is downloaded or sampled
    from anywhere. Returns None for "none"/unknown names or on synth
    failure (caller treats that as "skip the effect", never an error)."""
    if not name or name == "none" or name not in SOUND_EFFECTS:
        return None
    path = SFX_DIR / f"{name}.wav"
    if path.exists():
        return path
    filters = {
        "ting": "sine=frequency=1400:duration=0.35,afade=t=out:st=0.05:d=0.3,volume=0.9",
        "pop": "anoisesrc=d=0.12:c=pink:a=0.9,bandpass=f=300:width_type=h:w=200,afade=t=out:st=0:d=0.12,volume=1.4",
        "whoosh": "anoisesrc=d=0.4:c=white:a=0.6,bandpass=f=1500:width_type=h:w=1800,"
                  "afade=t=in:st=0:d=0.08,afade=t=out:st=0.2:d=0.2,volume=0.8",
        "click": "sine=frequency=2800:duration=0.035,afade=t=out:st=0:d=0.03,volume=0.5",
    }
    filt = filters.get(name)
    if not filt:
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", filt, "-ar", "44100", str(path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not path.exists():
            print(f"SFX synth failed for {name}: {result.stderr[-400:]}")
            return None
        return path
    except Exception as e:
        print(f"SFX synth error for {name}: {e}")
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


def download_bg_music(url: str, dest: Path) -> Optional[Path]:
    """Downloads just the audio from a YouTube/YouTube Music link to use as
    background music. Reuses the same player-client fallback + optional
    cookies approach as download_video() since these are still youtube.com
    requests subject to the same bot-detection. Returns None (never
    raises) on failure — background music is always an enhancement, never
    a requirement for a clip to render."""
    out_template = str(dest / "bgmusic.%(ext)s")
    cookies_path = None
    if YTDLP_COOKIES:
        cookies_path = dest / "cookies_music.txt"
        cookies_path.write_text(YTDLP_COOKIES, encoding="utf-8")

    for client in ["android", "ios", "tv", "mweb", "web_creator", "web"]:
        args = [
            "yt-dlp", "-f", "bestaudio/best", "--extract-audio", "--audio-format", "m4a",
            "--no-playlist", "--extractor-args", f"youtube:player_client={client}",
        ]
        if POT_PROVIDER_URL:
            args += ["--extractor-args", f"youtubepot-bgutilhttp:base_url={POT_PROVIDER_URL}"]
        if cookies_path:
            args += ["--cookies", str(cookies_path)]
        args += ["-o", out_template, url]
        result = subprocess.run(args, capture_output=True, text=True)
        candidate = dest / "bgmusic.m4a"
        if result.returncode == 0 and candidate.exists():
            return candidate
        print(f"bg-music download attempt with player_client={client} failed:\n{result.stderr[-600:]}")
    return None


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
        sfx="none", sfx_position="end", typing_sound=False, bg_music_url="", flash_intro=False,
    )


def _process_source(source: Path, job_dir: Path, job_id: str, clip_count: int, clip_seconds: int,
                     instruction: str, opts: dict, source_url: str = "") -> dict:
    """Shared pipeline body for both /process (download-from-URL) and
    /process-upload (user's own file) — everything after "we have a
    source.mp4 on disk" is identical between the two entry points.
    `opts` holds every clip-rendering option (voiceover, zoom/color,
    caption style, sfx, background music, etc) so this signature doesn't
    grow a new positional parameter every time a feature is added."""
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

    # Background music is downloaded once per /process call (not once per
    # clip) since every clip in the batch shares the same music track.
    bg_music_path = None
    if opts.get("bg_music_url"):
        bg_music_path = download_bg_music(opts["bg_music_url"], job_dir)
        if not bg_music_path:
            print(f"Background music download failed for {opts['bg_music_url']!r} — continuing without it.")

    clips = []
    for i, h in enumerate(highlights):
        ass_path = job_dir / f"clip_{i}.ass"
        words_to_ass(words, h["start"], h["end"], ass_path, caption_style=opts["caption_style"])

        out_name = f"{job_id}_{i}.mp4"
        out_path = OUTPUT_DIR / out_name

        try:
            cut_and_caption(source, h["start"], h["end"], ass_path, out_path,
                             zoom_pan=opts["zoom_pan"], color_preset=opts["color_preset"],
                             flash_intro=opts["flash_intro"])
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

    return {"job_id": job_id, "clip_count": len(clips), "clips": clips}


@app.post("/process")
def process(req: ProcessRequest):
    cleanup_old_jobs()

    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        source = download_video(req.url, job_dir)
    except Exception as e:
        raise HTTPException(400, f"Could not download video: {e}")

    opts = dict(
        voiceover=req.voiceover, voiceover_voice=req.voiceover_voice, voiceover_style=req.voiceover_style,
        zoom_pan=req.zoom_pan, color_preset=req.color_preset, caption_style=req.caption_style,
        sfx=req.sfx, sfx_position=req.sfx_position, typing_sound=req.typing_sound,
        bg_music_url=req.bg_music_url, flash_intro=req.flash_intro,
    )
    return _process_source(source, job_dir, job_id, req.clip_count, req.clip_seconds, req.instruction,
                            opts, req.url)


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
    bg_music_url: str = Form(""),
    flash_intro: bool = Form(False),
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
        bg_music_url=bg_music_url, flash_intro=flash_intro,
    )
    return _process_source(source, job_dir, job_id, clip_count, clip_seconds, instruction,
                            opts, "uploaded file")


@app.post("/reclip")
def reclip(req: ReclipRequest):
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
    words_to_ass(words, start, end, ass_path, caption_style=req.caption_style)

    out_name = f"{req.job_id}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = OUTPUT_DIR / out_name
    try:
        cut_and_caption(source, start, end, ass_path, out_path,
                         zoom_pan=req.zoom_pan, color_preset=req.color_preset, flash_intro=req.flash_intro)
    except Exception as e:
        raise HTTPException(500, f"Failed to render clip: {e}")

    voiceover_script = apply_voiceover_if_wanted(
        source, job_dir, out_path, start, end, req.clip_seconds,
        words, req.voiceover, req.voiceover_voice, req.instruction, req.voiceover_style,
    )

    bg_music_path = download_bg_music(req.bg_music_url, job_dir) if req.bg_music_url else None
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
        raise HTTPException(502, "Voice synthesis failed — check that ELEVENLABS_API_KEY is set correctly.")

    muxed = clip_path.with_suffix(".vo.mp4")
    if not mux_voiceover(clip_path, vo_audio, muxed):
        raise HTTPException(500, "Failed to combine the new narration with the clip.")
    muxed.replace(clip_path)

    return {"file": req.clip_file, "url": f"/clips/{req.clip_file}", "voiceover_script": req.script, "has_voiceover": True}


@app.get("/voices")
def list_voices():
    return {"voices": list(ELEVENLABS_VOICES.keys()), "default": DEFAULT_VOICE}


@app.get("/color-presets")
def list_color_presets():
    return {"presets": COLOR_PRESETS, "default": "none"}


@app.get("/caption-styles")
def list_caption_styles():
    return {"styles": list(CAPTION_STYLES.keys()), "default": DEFAULT_CAPTION_STYLE}


@app.get("/voiceover-styles")
def list_voiceover_styles():
    return {"styles": list(VOICEOVER_STYLES.keys()), "default": DEFAULT_VOICEOVER_STYLE}


@app.get("/sound-effects")
def list_sound_effects():
    return {"effects": SOUND_EFFECTS, "default": "none"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "transcription": "deepgram" if DEEPGRAM_API_KEY else "not configured",
        "highlight_picking": "claude" if ANTHROPIC_API_KEY else "heuristic fallback",
        "youtube_cookies": "configured" if YTDLP_COOKIES else "not set (may hit YouTube bot checks)",
        "pot_provider": "configured" if POT_PROVIDER_URL else "not set",
        "voiceover": "elevenlabs" if ELEVENLABS_API_KEY else "not configured",
        "direct_upload": "enabled",
        "zoom_pan": "enabled",
        "color_grading": "enabled",
        "caption_styles": "enabled",
        "sound_effects": "enabled",
        "background_music": "enabled",
        "fades": "enabled",
    }
