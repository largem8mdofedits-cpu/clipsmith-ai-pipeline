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

import os
import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
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

# ---------------------------------------------------------------------------
# Config — all via env vars so this runs the same locally and on Railway.
# ---------------------------------------------------------------------------
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")


class ProcessRequest(BaseModel):
    url: str
    clip_count: int = 3
    clip_seconds: int = 30


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
    result = subprocess.run(
        ["yt-dlp",
         "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
         "--merge-output-format", "mp4",
         "--no-playlist",
         "--remote-components", "ejs:github",
         "--js-runtimes", "deno",
         "-o", str(out_path), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[-800:]}")
    if not out_path.exists():
        raise RuntimeError(
            f"yt-dlp reported success but {out_path} wasn't created — "
            f"check that ffmpeg is on PATH (needed for merging).\n{result.stdout[-500:]}"
        )
    return out_path


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


def pick_highlights_llm(words, clip_count: int, clip_seconds: int, total_duration: float):
    """Asks Claude to pick the best moments in the video for short-form
    clips — hooks, punchlines, surprising or emotional beats — rather than
    just the windows with the most words in them. Returns None (so the
    caller falls back to the heuristic) if no API key is set or anything
    about the call/parsing goes wrong, so a flaky LLM response never takes
    down clip generation entirely."""
    if not ANTHROPIC_API_KEY or not words:
        return None

    transcript = build_timestamped_transcript(words)
    prompt = (
        f"You're picking the {clip_count} best {clip_seconds}-second moments from this "
        f"video transcript, to turn into short-form clips for TikTok/Reels/Shorts. Look "
        f"for hooks, punchlines, surprising claims, or emotional beats — not just the "
        f"parts with the most talking.\n\n"
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
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]

        match = re.search(r"\[.*\]", text, re.DOTALL)
        picks = json.loads(match.group(0) if match else text)

        highlights = []
        for p in picks[:clip_count]:
            start = max(0.0, float(p["start"]))
            end = min(total_duration, start + clip_seconds)
            if end - start < clip_seconds * 0.5:
                continue
            highlights.append({"start": start, "end": end, "reason": p.get("reason", "")})

        highlights.sort(key=lambda h: h["start"])
        return highlights or None
    except Exception as e:
        print(f"LLM highlight picking failed, falling back to heuristic: {e}")
        return None


def pick_highlights_heuristic(words, clip_count: int, clip_seconds: int, total_duration: float):
    """Speech-density fallback: scores fixed-length windows by how many
    words land in them and returns the top non-overlapping windows. Used
    when no ANTHROPIC_API_KEY is set, or if the LLM call fails for any
    reason — clip generation should never hard-fail just because the
    smarter picker had a bad day."""
    if not words:
        return []

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
        overlaps = any(not (c["end"] <= x["start"] or c["start"] >= x["end"]) for x in chosen)
        if not overlaps:
            chosen.append(c)

    chosen.sort(key=lambda c: c["start"])
    return chosen


def pick_highlights(words, clip_count: int, clip_seconds: int, total_duration: float):
    llm_picks = pick_highlights_llm(words, clip_count, clip_seconds, total_duration)
    if llm_picks:
        return llm_picks
    return pick_highlights_heuristic(words, clip_count, clip_seconds, total_duration)


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


def words_to_ass(words, clip_start: float, clip_end: float, ass_path: Path, chunk_size: int = 5):
    """Writes an .ass subtitle file scoped to one clip's time range: words
    highlight from white to yellow exactly as they're spoken (ASS karaoke
    \\k tags), AND each word pops with a quick scale bounce the instant
    it becomes active (ASS \\t transforms) — the combination is what gives
    auto-captions from tools like Submagic/Opus their punchy feel, rather
    than a flat color change. libass (built into ffmpeg) renders all of
    this directly from the tags below — no frame-by-frame image generation.

    PlayResX/Y match the 1080x1920 output frame so font sizes and margins
    line up correctly after the crop+scale filter runs.
    """
    clip_words = [w for w in words if clip_start <= w["start"] < clip_end]

    # PrimaryColour = the "already spoken" / highlighted colour (bright
    # yellow). SecondaryColour = the "not yet spoken" colour (white) that
    # \k tags start in before their timer elapses. Colours are &HAABBGGRR.
    # Outline/Shadow are pushed up from the original pass for a bolder,
    # more legible look at small preview sizes (matches what most
    # short-form caption tools default to).
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Karaoke,Liberation Sans Bold,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "1,0,0,0,100,100,0,0,1,5,2,2,60,60,190,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # group into on-screen lines of `chunk_size` words each
    chunks = [clip_words[i:i + chunk_size] for i in range(0, len(clip_words), chunk_size)]

    POP_MS = 90  # how long the scale-up half of each word's pop takes

    lines = [header]
    for chunk in chunks:
        if not chunk:
            continue
        line_start = chunk[0]["start"] - clip_start
        line_end = chunk[-1]["end"] - clip_start

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
            pop_start = word_offset_ms
            pop_mid = word_offset_ms + POP_MS
            pop_end = word_offset_ms + POP_MS * 2
            karaoke_text += (
                f"{{\\k{dur_cs}"
                f"\\t({pop_start},{pop_mid},\\fscx122\\fscy122)"
                f"\\t({pop_mid},{pop_end},\\fscx100\\fscy100)}}"
                f"{word} "
            )

        if not karaoke_text.strip():
            continue

        lines.append(
            f"Dialogue: 0,{_ass_timestamp(line_start)},{_ass_timestamp(line_end)},"
            f"Karaoke,,0,0,0,,{karaoke_text.strip()}\n"
        )

    ass_path.write_text("".join(lines), encoding="utf-8")


def cut_and_caption(source: Path, start: float, end: float, ass_path: Path, out_path: Path):
    """Cuts the clip, reframes to 9:16, and burns in the animated
    karaoke captions — all via ffmpeg.

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

    vf = (
        "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',"
        "scale=1080:1920,"
        f"ass='{ass_name}'"
    )
    result = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(source.resolve()), "-t", str(duration),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path.resolve())],
        capture_output=True, text=True,
        cwd=str(ass_dir),  # <-- this is what makes the bare filename resolve correctly
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.post("/process")
def process(req: ProcessRequest):
    job_id = str(uuid.uuid4())[:8]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        try:
            source = download_video(req.url, tmp)
        except Exception as e:
            raise HTTPException(400, f"Could not download video: {e}")

        try:
            duration = get_duration(source)
        except Exception as e:
            raise HTTPException(500, f"Downloaded video seems invalid: {e}")

        try:
            audio = extract_audio(source, tmp)
            words = transcribe(audio)
        except Exception as e:
            raise HTTPException(500, f"Transcription failed: {e}")

        highlights = pick_highlights(words, req.clip_count, req.clip_seconds, duration)

        if not highlights:
            raise HTTPException(422, "Couldn't find enough speech to build a clip from this video.")

        clips = []
        for i, h in enumerate(highlights):
            ass_path = tmp / f"clip_{i}.ass"
            words_to_ass(words, h["start"], h["end"], ass_path)

            out_name = f"{job_id}_{i}.mp4"
            out_path = OUTPUT_DIR / out_name

            try:
                cut_and_caption(source, h["start"], h["end"], ass_path, out_path)
            except Exception as e:
                raise HTTPException(500, f"Failed to render clip {i}: {e}")

            clips.append({
                "file": out_name,
                "start": h["start"],
                "end": h["end"],
                "reason": h.get("reason", ""),
                "url": f"/clips/{out_name}",
            })

        return {"job_id": job_id, "clip_count": len(clips), "clips": clips}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "transcription": "deepgram" if DEEPGRAM_API_KEY else "not configured",
        "highlight_picking": "claude" if ANTHROPIC_API_KEY else "heuristic fallback",
    }
