"""
Skating Music Editor — local Flask app for cutting and crossfading audio
for figure skating programs, with AI-powered cut suggestions.

Pipeline:
    upload  ->  waveform UI  ->  [optional AI analysis]  ->  manual refinement
            ->  cut + crossfade + export

AI analysis:
    librosa extracts tempo, beat grid, section boundaries, and an energy
    curve from the audio. That compact summary is sent to the Claude API,
    which returns suggested cut regions snapped to musical phrase boundaries.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
import statistics
import threading
import time
import uuid
from pathlib import Path

from typing import Iterable

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from pydub import AudioSegment
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "workspace"
WORK_DIR.mkdir(exist_ok=True)

# Short build identifier — combines the mtime of app.py with the server start time.
# The app.py component changes when the code changes; the start-time component
# changes on every restart. If the value shown in the UI doesn't match what the
# server is currently returning, the browser is serving a cached page.
import time as _time
try:
    _app_mtime = int(Path(__file__).stat().st_mtime)
except OSError:
    _app_mtime = 0
BUILD_ID = f"{_app_mtime:x}.{int(_time.time()):x}"[-10:]

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "aac", "flac", "ogg"}
MAX_UPLOAD_MB = 50
MAX_URL_DURATION_SEC = 20 * 60

# User-facing and AI-facing descriptions of each program_style choice. Keep in
# sync with AI_STYLE_HINTS in templates/index.html — same keys, same wording.
STYLE_DESCRIPTIONS = {
    "balanced": "Default — even mix of rhythmic coherence, vocal phrasing, and emotional arc. No single priority dominates.",
    "dramatic": "Prioritizes big moments — climaxes, drops, key changes. Willing to accept slightly rougher joins if they land on a peak.",
    "lyrical": "Preserves vocal phrases and melodic lines. Gentler transitions, fewer cuts mid-phrase, more on-breath edits.",
    "technical": "Tight on-beat/downbeat joins and rhythmic coherence. Skating-grid friendly; prioritizes predictable pulse over emotion.",
    "aggressive": "Bigger removals, willing to skip whole sections for pace. Favors high-energy material; drops softer bridges and intros.",
}


def style_description(name: str) -> str:
    return STYLE_DESCRIPTIONS.get((name or "").strip().lower(), STYLE_DESCRIPTIONS["balanced"])


def analysis_sidecar_path(file_id: str) -> Path:
    return WORK_DIR / f"{file_id}.analysis.json"


def save_analysis_sidecar(file_id: str, analysis: dict) -> None:
    """Persist the analysis dict so follow-up endpoints (e.g. /optimize_audition)
    don't have to re-run librosa — that costs 20–90s on a typical track."""
    try:
        analysis_sidecar_path(file_id).write_text(json.dumps(analysis), encoding="utf-8")
    except OSError:
        # Best-effort: the feature degrades to "run analyze first" without this, not a hard failure.
        pass


def load_analysis_sidecar(file_id: str) -> dict | None:
    p = analysis_sidecar_path(file_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

# Default multimodal provider. Gemini is the primary full-audio path.
AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# Auto-reload Jinja templates even without debug mode. Without this, template
# edits are silently ignored until the server restarts, which is a very easy
# way to waste 20 minutes chasing a "my change didn't take effect" bug.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_cuts(cuts: list[dict], duration_sec: float) -> list[dict]:
    normalized = []
    for c in sorted(cuts or [], key=lambda c: float(c.get("start", 0.0))):
        try:
            start = max(0.0, float(c["start"]))
            end = min(float(duration_sec), float(c["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        if normalized and start < normalized[-1]["end"]:
            start = normalized[-1]["end"]
        if end <= start:
            continue
        normalized.append({
            "start": round(start, 2),
            "end": round(end, 2),
        })
    return normalized


def _audiosegment_to_float_array(audio: AudioSegment) -> np.ndarray:
    """Return a (frames, channels) float32 array normalized to ~[-1, 1]. pydub
    stores samples as interleaved ints; this reshapes and scales them once so
    the render loop can splice on sample boundaries."""
    raw = np.array(audio.get_array_of_samples())
    if audio.channels > 1:
        raw = raw.reshape(-1, audio.channels)
    else:
        raw = raw.reshape(-1, 1)
    max_val = float(1 << (8 * audio.sample_width - 1))  # e.g. 32768 for 16-bit
    return raw.astype(np.float32) / max_val


def _float_array_to_audiosegment(buf: np.ndarray, frame_rate: int, sample_width: int, channels: int) -> AudioSegment:
    """Inverse of _audiosegment_to_float_array. Clips to the int range before
    casting so a crossfade sum doesn't wrap around into the opposite polarity."""
    max_val = float(1 << (8 * sample_width - 1))
    int_dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    clipped = np.clip(buf, -1.0, 1.0 - 1.0 / max_val)
    ints = (clipped * max_val).astype(int_dtype)
    return AudioSegment(
        ints.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels,
    )


def nearest_zero_crossing(mono: np.ndarray, frame: int, radius: int) -> int:
    """Find a frame near `frame` (within ±radius samples) where the mono
    downmix crosses zero, falling back to the local minimum-amplitude frame if
    no sign change is found. Splicing here turns the splice-point discontinuity
    from a step to (near) zero, which is what eliminates audible clicks."""
    n = mono.shape[0]
    if n == 0:
        return max(0, min(n, frame))
    lo = max(1, frame - radius)
    hi = min(n - 1, frame + radius)
    if hi <= lo:
        return max(0, min(n, frame))
    window = mono[lo - 1:hi + 1]
    # A sign change between samples i-1 and i means a zero crossing at index i (in window coords).
    prev = window[:-1]
    curr = window[1:]
    crossings = np.where((prev * curr) <= 0)[0]
    if crossings.size > 0:
        # Pick the crossing whose sample is closest to zero — avoids landing on
        # a zero-touching peak where the neighborhood slope is still huge.
        candidates = crossings + 1  # shift from diff index to window index
        abs_at = np.abs(window[candidates])
        best_local = candidates[int(np.argmin(abs_at))]
        return lo - 1 + int(best_local)
    # No sign change in window — fall back to the absolute-minimum amplitude.
    local_idx = int(np.argmin(np.abs(window)))
    return lo - 1 + local_idx


def equal_power_crossfade(left_tail: np.ndarray, right_head: np.ndarray) -> np.ndarray:
    """Constant-power (sqrt) crossfade across the overlap region. pydub's
    built-in crossfade uses linear gain ramps, which produce a ~3 dB midpoint
    dip for uncorrelated musical content or a loudness bump for correlated
    content. sin/cos ramps keep perceived loudness flat across the seam."""
    n = min(left_tail.shape[0], right_head.shape[0])
    if n <= 0:
        return np.zeros((0, left_tail.shape[1] if left_tail.ndim > 1 else 1), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    gain_left = np.cos(t * (np.pi / 2.0))  # sqrt(1-t) family
    gain_right = np.sin(t * (np.pi / 2.0))  # sqrt(t) family
    if left_tail.ndim > 1:
        gain_left = gain_left[:, None]
        gain_right = gain_right[:, None]
    return left_tail[:n] * gain_left + right_head[:n] * gain_right


def render_audio_from_cuts(
    src: Path,
    cuts: list[dict],
    crossfade_ms: int = 400,
    fade_in_ms: int = 100,
    fade_out_ms: int = 2000,
    target_format: str = "mp3",
    bitrate: str = "192k",
    output_id: str | None = None,
    zero_crossing_radius_ms: float = 10.0,
) -> dict:
    audio = AudioSegment.from_file(src)
    frame_rate = audio.frame_rate
    sample_width = audio.sample_width
    channels = audio.channels
    total_ms = len(audio)
    total_frames = int(audio.frame_count())
    norm_cuts = normalize_cuts(cuts, total_ms / 1000.0)

    samples = _audiosegment_to_float_array(audio)
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    zc_radius = max(1, int(zero_crossing_radius_ms * 0.001 * frame_rate))

    def ms_to_frame(ms: float) -> int:
        return max(0, min(total_frames, int(round(ms * 0.001 * frame_rate))))

    # Cuts are REMOVE regions — invert into KEEP regions in frame units.
    keep_regions: list[tuple[int, int]] = []
    cursor = 0
    for c in norm_cuts:
        s = ms_to_frame(c["start"] * 1000.0)
        e = ms_to_frame(c["end"] * 1000.0)
        if s > cursor:
            keep_regions.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_frames:
        keep_regions.append((cursor, total_frames))

    if not keep_regions:
        raise ValueError("Nothing left to keep — your cuts cover the entire song")

    # Zero-crossing snap both sides of every splice. The very start and very
    # end of the program are left alone (fade_in/fade_out will handle them).
    refined: list[tuple[int, int]] = []
    for idx, (s, e) in enumerate(keep_regions):
        rs = s if idx == 0 else nearest_zero_crossing(mono, s, zc_radius)
        re_ = e if idx == len(keep_regions) - 1 else nearest_zero_crossing(mono, e, zc_radius)
        # Keep regions ordered and non-empty even after the snap.
        if refined and rs < refined[-1][1]:
            rs = refined[-1][1]
        if re_ <= rs:
            continue
        refined.append((rs, re_))

    if not refined:
        raise ValueError("Cuts left no renderable audio after alignment")

    cf_frames_requested = int(max(0, crossfade_ms) * 0.001 * frame_rate)

    pieces: list[np.ndarray] = []
    for idx, (s, e) in enumerate(refined):
        clip = samples[s:e]
        if idx == 0:
            pieces.append(clip)
            continue
        prev = pieces[-1]
        # Clamp crossfade to half of the shorter clip so a splice never eats
        # the entire previous keep region or bleeds into the next one.
        cf = min(cf_frames_requested, prev.shape[0] // 2, clip.shape[0] // 2)
        if cf <= 0:
            pieces.append(clip)
            continue
        try:
            left_tail = prev[-cf:]
            right_head = clip[:cf]
            blended = equal_power_crossfade(left_tail, right_head)
            pieces[-1] = prev[:-cf]
            pieces.append(blended)
            pieces.append(clip[cf:])
        except (ValueError, IndexError) as err:
            # Shape mismatch or similar — surface it rather than silently
            # hard-concatenating; a click in this case is a fixable bug.
            app.logger.warning("crossfade failed, falling back to hard concat: %s", err)
            pieces.append(clip)

    out_buf = np.concatenate(pieces, axis=0) if pieces else np.zeros((0, channels), dtype=np.float32)
    out = _float_array_to_audiosegment(out_buf, frame_rate, sample_width, channels)

    if fade_in_ms > 0 and len(out) > fade_in_ms:
        out = out.fade_in(fade_in_ms)
    if fade_out_ms > 0 and len(out) > fade_out_ms:
        out = out.fade_out(fade_out_ms)

    out_id = output_id or uuid.uuid4().hex
    out_path = WORK_DIR / f"{out_id}.{target_format}"
    export_kwargs: dict = {"format": target_format}
    if target_format == "mp3":
        export_kwargs["bitrate"] = bitrate
    out.export(out_path, **export_kwargs)

    return {
        "output_id": out_id,
        "format": target_format,
        "duration_sec": len(out) / 1000.0,
        "size_bytes": out_path.stat().st_size,
        "cuts": norm_cuts,
    }


# ────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    # Disable caching for the HTML shell so a reload always pulls the latest
    # template; the inline JS is fingerprinted with BUILD_ID for a visible signal.
    resp = app.make_response(render_template("index.html", build_id=BUILD_ID))
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/build_id")
def build_id():
    return jsonify({"build_id": BUILD_ID})


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed(f.filename):
        return jsonify({"error": f"Unsupported type. Use: {sorted(ALLOWED_EXTENSIONS)}"}), 400

    ext = f.filename.rsplit(".", 1)[1].lower()
    file_id = uuid.uuid4().hex
    path = WORK_DIR / f"{file_id}.{ext}"
    f.save(path)

    try:
        audio = AudioSegment.from_file(path)
    except Exception as e:
        path.unlink(missing_ok=True)
        return jsonify({"error": f"Could not decode audio: {e}"}), 400

    return jsonify({
        "file_id": file_id,
        "ext": ext,
        "duration_sec": len(audio) / 1000.0,
        "original_name": f.filename,
    })


@app.route("/download_url", methods=["POST"])
def download_url():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    if not re.match(r"^https?://", url, re.I):
        return jsonify({"error": "only http(s) URLs are supported"}), 400

    file_id = uuid.uuid4().hex
    out_path = WORK_DIR / f"{file_id}.mp3"

    # yt-dlp needs a JavaScript runtime to negotiate YouTube's signed media URLs;
    # without it, downloads fall back to formats that often 403. node is widely
    # available and works for the YouTube extractor.
    yt_js_runtimes = {"node": {}}

    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "js_runtimes": yt_js_runtimes,
    }
    try:
        with YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        return jsonify({"error": f"Could not read URL: {e}"}), 400

    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            return jsonify({"error": "Playlist has no entries"}), 400
        info = entries[0]

    duration = float(info.get("duration") or 0)
    if duration <= 0:
        return jsonify({"error": "Could not determine duration for this URL"}), 400
    if duration > MAX_URL_DURATION_SEC:
        return jsonify({
            "error": f"Video is {duration/60:.1f} min; max {MAX_URL_DURATION_SEC//60} min"
        }), 400

    title = (info.get("title") or "download").strip()

    download_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio/best",
        "outtmpl": str(WORK_DIR / f"{file_id}.%(ext)s"),
        "js_runtimes": yt_js_runtimes,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    try:
        with YoutubeDL(download_opts) as ydl:
            ydl.download([url])
    except DownloadError as e:
        for p in WORK_DIR.glob(f"{file_id}.*"):
            p.unlink(missing_ok=True)
        return jsonify({"error": f"Download failed: {e}"}), 502

    if not out_path.exists():
        for p in WORK_DIR.glob(f"{file_id}.*"):
            p.unlink(missing_ok=True)
        return jsonify({"error": "Download completed but mp3 was not produced"}), 500

    if out_path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        out_path.unlink(missing_ok=True)
        return jsonify({"error": f"Downloaded file exceeds {MAX_UPLOAD_MB} MB"}), 400

    try:
        audio = AudioSegment.from_file(out_path)
    except Exception as e:
        out_path.unlink(missing_ok=True)
        return jsonify({"error": f"Could not decode downloaded audio: {e}"}), 500

    return jsonify({
        "file_id": file_id,
        "ext": "mp3",
        "duration_sec": len(audio) / 1000.0,
        "original_name": f"{title}.mp3",
    })


@app.route("/audio/<file_id>.<ext>")
def serve_audio(file_id: str, ext: str):
    path = WORK_DIR / f"{file_id}.{ext}"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype=f"audio/{ext}")


@app.route("/waveform_image/<file_id>.<ext>")
def serve_waveform_image(file_id: str, ext: str):
    src = WORK_DIR / f"{file_id}.{ext}"
    out = WORK_DIR / f"{file_id}_waveform.png"
    if not src.exists():
        return jsonify({'error': 'source file not found'}), 404
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        try:
            analysis = analyze_audio(src)
            generate_waveform_image(src, file_id=file_id, analysis=analysis)
        except Exception as e:
            return jsonify({'error': f'waveform image generation failed: {e}'}), 500
    return send_file(out, mimetype='image/png')


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Run local music-structure analysis and optionally ask the LLM for
    stronger cut suggestions built on top of those heuristics.

    Response is NDJSON (one JSON object per line) so the connection stays warm
    during the slow multimodal AI step — which takes 60-120s and otherwise gets
    killed by browser/proxy idle timeouts. Lines:
      {"type":"status", "stage":"...", "elapsed":float, ...}
      {"type":"heartbeat", "elapsed":float}    # emitted every few seconds
      {"type":"result", "analysis":..., "suggestion":..., "ai_error":...}
      {"type":"error", "error":"..."}
    """
    data = request.get_json(force=True)
    file_id = data["file_id"]
    ext = data["ext"]
    target_sec = data.get("target_sec")
    discipline = data.get("discipline", "")
    program_style = data.get("program_style", "balanced")
    aggressiveness = int(data.get("aggressiveness", 50))
    num_options = max(1, min(3, int(data.get("num_options", 3))))
    use_ai = data.get("use_ai", True)
    ai_provider = str(data.get("ai_provider", AI_PROVIDER)).strip().lower()

    src = WORK_DIR / f"{file_id}.{ext}"
    if not src.exists():
        return jsonify({"error": "source file not found"}), 404

    t0 = time.monotonic()

    def _emit(obj):
        return json.dumps(obj, separators=(",", ":")) + "\n"

    def generate():
        try:
            yield _emit({"type": "status", "stage": "analyzing_audio", "elapsed": round(time.monotonic() - t0, 2)})
            try:
                analysis = analyze_audio(src)
            except ImportError as e:
                yield _emit({"type": "error", "error": f"librosa not installed: {e}"})
                return
            except Exception as e:
                yield _emit({"type": "error", "error": f"audio analysis failed: {e}"})
                return

            if target_sec:
                analysis["local_candidates"] = build_cut_candidates(analysis, float(target_sec), aggressiveness)
                analysis["recommended_opening_guard_sec"] = min(20.0, max(10.0, analysis["duration_sec"] * 0.08))
                analysis["recommended_ending_guard_sec"] = min(25.0, max(12.0, analysis["duration_sec"] * 0.1))
            try:
                waveform_path = generate_waveform_image(src, file_id=file_id, analysis=analysis)
                analysis["waveform_image_url"] = f"/waveform_image/{file_id}.{ext}?v={int(waveform_path.stat().st_mtime)}"
            except Exception as e:
                analysis["waveform_image_error"] = str(e)

            # Cache the full analysis to disk so /optimize_audition can reuse it
            # without paying the librosa cost again.
            save_analysis_sidecar(file_id, analysis)

            suggestion = None
            ai_error = None

            if use_ai and target_sec:
                yield _emit({
                    "type": "status",
                    "stage": "asking_ai",
                    "provider": ai_provider,
                    "elapsed": round(time.monotonic() - t0, 2),
                })
                # Run the AI call in a thread so we can keep the HTTP connection
                # warm with heartbeats; multimodal Gemini can take 1–2 minutes.
                result_ref = {"suggestion": None, "ai_error": None}
                done = threading.Event()

                def _run_ai():
                    try:
                        result_ref["suggestion"] = ai_suggest_cuts(
                            src_path=src,
                            analysis=analysis,
                            target_sec=float(target_sec),
                            discipline=discipline,
                            program_style=program_style,
                            aggressiveness=aggressiveness,
                            num_options=num_options,
                            provider=ai_provider,
                        )
                    except Exception as e:
                        result_ref["ai_error"] = str(e)
                    finally:
                        done.set()

                worker = threading.Thread(target=_run_ai, daemon=True)
                worker.start()
                while not done.wait(timeout=4.0):
                    yield _emit({"type": "heartbeat", "elapsed": round(time.monotonic() - t0, 2)})
                suggestion = result_ref["suggestion"]
                ai_error = result_ref["ai_error"]

            yield _emit({
                "type": "result",
                "elapsed": round(time.monotonic() - t0, 2),
                "analysis": analysis,
                "suggestion": suggestion,
                "ai_error": ai_error,
            })
        except GeneratorExit:
            # Client disconnected — nothing to do; the daemon thread will finish on its own.
            return

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
    )


@app.route("/process", methods=["POST"])
def process():
    """
    Body: {
        "file_id", "ext",
        "cuts": [{"start": float, "end": float}, ...],  # seconds, REMOVE
        "crossfade_ms", "fade_in_ms", "fade_out_ms",
        "target_format": "mp3" | "wav",
        "bitrate": "192k"
    }
    """
    data = request.get_json(force=True)
    file_id = data["file_id"]
    ext = data["ext"]
    cuts = sorted(data.get("cuts", []), key=lambda c: c["start"])
    crossfade_ms = int(data.get("crossfade_ms", 400))
    fade_out_ms = int(data.get("fade_out_ms", 2000))
    fade_in_ms = int(data.get("fade_in_ms", 100))
    target_format = data.get("target_format", "mp3")
    bitrate = data.get("bitrate", "192k")
    align = bool(data.get("align", True))

    src = WORK_DIR / f"{file_id}.{ext}"
    if not src.exists():
        return jsonify({"error": "source file not found"}), 404

    # Run user-submitted cuts through the same beat/vocal alignment the AI
    # cuts get — users dragging regions on the waveform would otherwise splice
    # at arbitrary sub-beat positions. Falls through to the raw cuts only when
    # the analysis sidecar is missing (pre-analyze state).
    aligned_cuts = cuts
    if align:
        analysis = load_analysis_sidecar(file_id)
        if analysis:
            aligned_cuts = sanitize_cut_list(cuts, analysis)

    try:
        result = render_audio_from_cuts(
            src=src,
            cuts=aligned_cuts,
            crossfade_ms=crossfade_ms,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            target_format=target_format,
            bitrate=bitrate,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Echo the aligned cuts back so the frontend can redraw regions on the
    # snapped positions — the user sees where the splice actually landed.
    result["aligned_cuts"] = aligned_cuts
    return jsonify(result)


@app.route("/render_auditions", methods=["POST"])
def render_auditions():
    """Render audio previews for one or more AI edit plans."""
    data = request.get_json(force=True)
    file_id = data["file_id"]
    ext = data["ext"]
    plans = data.get("plans", [])
    target_format = data.get("target_format", "mp3")
    bitrate = data.get("bitrate", "192k")
    crossfade_ms = int(data.get("crossfade_ms", 120))
    fade_in_ms = int(data.get("fade_in_ms", 40))
    fade_out_ms = int(data.get("fade_out_ms", 600))

    src = WORK_DIR / f"{file_id}.{ext}"
    if not src.exists():
        return jsonify({"error": "source file not found"}), 404
    if not plans:
        return jsonify({"error": "No plans supplied"}), 400

    rendered = []
    for idx, plan in enumerate(plans[:5]):
        plan_id = str(plan.get("plan_id", idx))
        title = str(plan.get("title") or f"Plan {idx + 1}")[:80]
        cuts = plan.get("cuts", [])
        output_id = f"aud_{file_id[:8]}_{uuid.uuid4().hex[:12]}"
        try:
            result = render_audio_from_cuts(
                src=src,
                cuts=cuts,
                crossfade_ms=crossfade_ms,
                fade_in_ms=fade_in_ms,
                fade_out_ms=fade_out_ms,
                target_format=target_format,
                bitrate=bitrate,
                output_id=output_id,
            )
        except Exception as e:
            rendered.append({
                "plan_id": plan_id,
                "title": title,
                "error": str(e),
            })
            continue

        rendered.append({
            "plan_id": plan_id,
            "title": title,
            "summary": str(plan.get("summary") or "")[:300],
            "duration_sec": result["duration_sec"],
            "format": result["format"],
            "size_bytes": result["size_bytes"],
            "output_id": result["output_id"],
            "download_url": f"/download/{result['output_id']}.{result['format']}",
            "preview_url": f"/audio/{result['output_id']}.{result['format']}",
            "cuts": result["cuts"],
        })

    return jsonify({"auditions": rendered})


@app.route("/optimize_audition", methods=["POST"])
def optimize_audition():
    """Send a rendered audition preview to Gemini, get a refined cut list,
    re-render, return the new audition entry. NDJSON stream so the slow Gemini
    call (60–120s) doesn't die on browser/proxy idle timeouts."""
    data = request.get_json(force=True)
    file_id = data["file_id"]
    ext = data["ext"]
    audition_output_id = data["audition_output_id"]
    audition_format = data.get("audition_format", "mp3")
    original_plan = data.get("original_plan") or {}
    target_sec = float(data.get("target_sec") or 0)
    discipline = data.get("discipline", "")
    program_style = data.get("program_style", "balanced")
    aggressiveness = int(data.get("aggressiveness", 50))
    render_opts = data.get("render_opts") or {}

    src = WORK_DIR / f"{file_id}.{ext}"
    audition_path = WORK_DIR / f"{audition_output_id}.{audition_format}"
    if not src.exists():
        return jsonify({"error": "source file not found"}), 404
    if not audition_path.exists():
        return jsonify({"error": "audition preview not found"}), 404

    analysis = load_analysis_sidecar(file_id)
    if not analysis:
        return jsonify({"error": "no cached analysis for this source — run Analyze first"}), 400

    t0 = time.monotonic()

    def _emit(obj):
        return json.dumps(obj, separators=(",", ":")) + "\n"

    def generate():
        try:
            yield _emit({
                "type": "status",
                "stage": "asking_ai",
                "provider": "gemini",
                "elapsed": round(time.monotonic() - t0, 2),
            })
            result_ref = {"refined": None, "ai_error": None}
            done = threading.Event()

            def _run_ai():
                try:
                    result_ref["refined"] = ai_optimize_audition_with_gemini(
                        audition_path=audition_path,
                        analysis=analysis,
                        original_plan=original_plan,
                        target_sec=target_sec,
                        discipline=discipline,
                        program_style=program_style,
                        aggressiveness=aggressiveness,
                    )
                except Exception as e:
                    result_ref["ai_error"] = str(e)
                finally:
                    done.set()

            worker = threading.Thread(target=_run_ai, daemon=True)
            worker.start()
            while not done.wait(timeout=4.0):
                yield _emit({"type": "heartbeat", "elapsed": round(time.monotonic() - t0, 2)})

            refined = result_ref["refined"]
            ai_error = result_ref["ai_error"]
            if ai_error or not refined:
                yield _emit({
                    "type": "result",
                    "elapsed": round(time.monotonic() - t0, 2),
                    "ai_error": ai_error or "AI returned no plan",
                    "audition": None,
                })
                return

            # Same sanitization pipeline as AI-suggested cuts: snap to section
            # + beat grid, escape vocal intervals.
            refined_cuts = sanitize_cut_list(refined.get("cuts") or [], analysis)
            if not refined_cuts:
                yield _emit({
                    "type": "result",
                    "elapsed": round(time.monotonic() - t0, 2),
                    "ai_error": "refined plan had no valid cuts after sanitization",
                    "audition": None,
                })
                return

            yield _emit({
                "type": "status",
                "stage": "rendering",
                "elapsed": round(time.monotonic() - t0, 2),
            })
            try:
                rendered = render_audio_from_cuts(
                    src=src,
                    cuts=refined_cuts,
                    crossfade_ms=int(render_opts.get("crossfade_ms", 250)),
                    fade_in_ms=int(render_opts.get("fade_in_ms", 100)),
                    fade_out_ms=int(render_opts.get("fade_out_ms", 2000)),
                    target_format=str(render_opts.get("target_format", "mp3")),
                    bitrate=str(render_opts.get("bitrate", "192k")),
                    output_id=f"opt_{file_id[:8]}_{uuid.uuid4().hex[:12]}",
                )
            except Exception as e:
                yield _emit({
                    "type": "result",
                    "elapsed": round(time.monotonic() - t0, 2),
                    "ai_error": f"render failed: {e}",
                    "audition": None,
                })
                return

            audition_entry = {
                "plan_id": f"optimized-of-{original_plan.get('title') or 'plan'}",
                "title": str(refined.get("title") or "Optimized")[:80],
                "summary": str(refined.get("summary") or "")[:300],
                "confidence": refined.get("confidence"),
                "transition_risk": (str(refined.get("transition_risk") or "")[:20] or None),
                "changes": [str(c)[:240] for c in (refined.get("changes") or [])][:10],
                "removed_sec": round(sum(c["end"] - c["start"] for c in refined_cuts), 2),
                "duration_sec": rendered["duration_sec"],
                "format": rendered["format"],
                "size_bytes": rendered["size_bytes"],
                "output_id": rendered["output_id"],
                "download_url": f"/download/{rendered['output_id']}.{rendered['format']}",
                "preview_url": f"/audio/{rendered['output_id']}.{rendered['format']}",
                "cuts": refined_cuts,
                "origin": {
                    "optimized_from_output_id": audition_output_id,
                    "optimized_from_title": str(original_plan.get("title") or "")[:80],
                },
            }
            yield _emit({
                "type": "result",
                "elapsed": round(time.monotonic() - t0, 2),
                "audition": audition_entry,
                "ai_error": None,
            })
        except GeneratorExit:
            return

    return Response(
        stream_with_context(generate()),
        mimetype="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
    )


@app.route("/download/<output_id>.<fmt>")
def download(output_id: str, fmt: str):
    path = WORK_DIR / f"{output_id}.{fmt}"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(
        path,
        as_attachment=True,
        download_name=f"skate_program_{output_id[:8]}.{fmt}",
        mimetype=f"audio/{fmt}",
    )


@app.route("/download_source/<file_id>.<ext>")
def download_source(file_id: str, ext: str):
    path = WORK_DIR / f"{file_id}.{ext}"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    # Respect a client-supplied name if provided; strip path separators so we
    # don't let a malicious query string escape the download filename.
    raw_name = (request.args.get("name") or "").strip()
    safe_name = re.sub(r"[\\/:*?\"<>|\r\n]+", "_", raw_name) if raw_name else ""
    if not safe_name:
        safe_name = f"source_{file_id[:8]}.{ext}"
    elif not safe_name.lower().endswith(f".{ext.lower()}"):
        safe_name = f"{safe_name}.{ext}"
    return send_file(
        path,
        as_attachment=True,
        download_name=safe_name,
        mimetype=f"audio/{ext}",
    )


# ────────────────────────────────────────────────────────────────────────────
# Audio analysis
# ────────────────────────────────────────────────────────────────────────────

def _downsample_curve(times, values, n=40):
    import numpy as np
    if len(values) == 0:
        return []
    n = min(n, len(values))
    idx = np.linspace(0, len(values) - 1, n).astype(int)
    return [[round(float(times[i]), 2), round(float(values[i]), 3)] for i in idx]


def _estimate_key_mode(chroma_mean):
    import numpy as np
    # Krumhansl-Schmuckler key profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
    note_names = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    chroma_mean = np.asarray(chroma_mean, dtype=float)
    chroma_norm = chroma_mean / max(1e-9, chroma_mean.sum())
    scores = []
    for root in range(12):
        maj = np.corrcoef(chroma_norm, np.roll(major_profile, root))[0, 1]
        mino = np.corrcoef(chroma_norm, np.roll(minor_profile, root))[0, 1]
        scores.append((note_names[root], 'major', float(maj)))
        scores.append((note_names[root], 'minor', float(mino)))
    scores.sort(key=lambda x: x[2], reverse=True)
    best = scores[0]
    second = scores[1]
    confidence = max(0.0, min(1.0, 0.5 + (best[2] - second[2]) * 0.8))
    return {
        'key': best[0],
        'mode': best[1],
        'label': f"{best[0]} {best[1]}",
        'confidence': round(confidence, 3),
        'top_matches': [
            {'key': k, 'mode': m, 'score': round(s, 3)}
            for k, m, s in scores[:4]
        ],
    }


def _intervals_from_mask(times, mask, min_duration=1.5, pad=0.0):
    intervals = []
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            s = max(0.0, float(times[start]) - pad)
            e = float(times[min(i, len(times) - 1)]) + pad
            if e - s >= min_duration:
                intervals.append([round(s, 2), round(e, 2)])
            start = None
    if start is not None:
        s = max(0.0, float(times[start]) - pad)
        e = float(times[-1]) + pad
        if e - s >= min_duration:
            intervals.append([round(s, 2), round(e, 2)])
    # merge close intervals
    merged = []
    for s, e in intervals:
        if not merged or s > merged[-1][1] + 0.75:
            merged.append([s, e])
        else:
            merged[-1][1] = round(max(merged[-1][1], e), 2)
    return [{'start': s, 'end': e, 'duration_sec': round(e-s, 2)} for s, e in merged]


def generate_waveform_image(path: Path, file_id: str | None = None, analysis: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa

    out_stem = file_id or path.stem
    out_path = WORK_DIR / f"{out_stem}_waveform.png"
    y, sr = librosa.load(str(path), sr=11025, mono=True)
    t = np.linspace(0, len(y) / sr, num=len(y))

    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=150)
    ax.plot(t, y, linewidth=0.6)
    ax.fill_between(t, y, 0, alpha=0.18)

    if analysis:
        for b in analysis.get('section_boundaries', []):
            ax.axvline(float(b), linestyle='--', linewidth=0.9, alpha=0.8)
        climax = analysis.get('estimated_climax_time')
        if climax is not None:
            ax.axvline(float(climax), linestyle=':', linewidth=1.4, alpha=0.95)
        for interval in analysis.get('likely_vocal_intervals', [])[:20]:
            ax.axvspan(interval['start'], interval['end'], alpha=0.08)

    ax.set_xlim(0, len(y) / sr if sr else 1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform preview')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path


def analyze_audio(path: Path) -> dict:
    """Extract a richer musical-structure summary using librosa."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = float(len(y) / sr)

    # Tempo + beat grid
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # Harmonic/percussive split helps with pitch-centric features
    y_harm, y_perc = librosa.effects.hpss(y)

    # Structural segmentation via chroma-based agglomerative clustering
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    k = max(4, min(10, int(duration / 25)))
    bound_frames = librosa.segment.agglomerative(chroma, k=k)
    section_boundaries = sorted(set(
        round(float(t), 2)
        for t in librosa.frames_to_time(bound_frames, sr=sr).tolist()
        if 0 < t < duration
    ))

    # Core curves
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    rms_max = float(rms.max()) if float(rms.max()) > 0 else 1.0
    energy_curve = _downsample_curve(rms_times, rms / rms_max)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
    onset_max = float(onset_env.max()) if float(onset_env.max()) > 0 else 1.0
    onset_curve = _downsample_curve(onset_times, onset_env / onset_max)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_times = librosa.frames_to_time(np.arange(len(spectral_centroid)), sr=sr)
    centroid_max = float(spectral_centroid.max()) if float(spectral_centroid.max()) > 0 else 1.0
    brightness_curve = _downsample_curve(centroid_times, spectral_centroid / centroid_max)

    # Key / mode estimate
    chroma_mean = chroma.mean(axis=1)
    key_info = _estimate_key_mode(chroma_mean)

    # Chord-change density via chroma delta between adjacent frames
    chroma_t = chroma.T
    if len(chroma_t) > 1:
        chroma_delta = np.linalg.norm(np.diff(chroma_t, axis=0), axis=1)
        delta_times = librosa.frames_to_time(np.arange(1, len(chroma_t)), sr=sr)
        cd_max = float(chroma_delta.max()) if float(chroma_delta.max()) > 0 else 1.0
        chord_change_curve = _downsample_curve(delta_times, chroma_delta / cd_max)
        thresh = float(np.quantile(chroma_delta, 0.7))
        change_events = int(np.sum(chroma_delta >= thresh))
        chord_change_density_per_min = round(change_events / max(duration / 60.0, 1e-6), 2)
    else:
        chord_change_curve = []
        chord_change_density_per_min = 0.0

    # Heuristic likely-vocal detection using pYIN + harmonic dominance + flatness
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_harm,
            sr=sr,
            frame_length=2048,
            hop_length=512,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
        )
        pyin_times = librosa.times_like(f0, sr=sr, hop_length=512)
        voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
        f0 = np.nan_to_num(f0, nan=0.0)
        harmonic_rms = librosa.feature.rms(y=y_harm, hop_length=512)[0]
        total_rms = librosa.feature.rms(y=y, hop_length=512)[0]
        flatness = librosa.feature.spectral_flatness(y=y_harm, hop_length=512)[0]
        n = min(len(pyin_times), len(voiced_prob), len(harmonic_rms), len(total_rms), len(flatness), len(f0))
        pyin_times = pyin_times[:n]
        voiced_prob = voiced_prob[:n]
        harmonic_rms = harmonic_rms[:n]
        total_rms = total_rms[:n]
        flatness = flatness[:n]
        f0 = f0[:n]
        harmonic_ratio = harmonic_rms / np.maximum(total_rms, 1e-6)
        in_vocal_range = (f0 >= 80.0) & (f0 <= 1100.0)
        vocal_mask = (voiced_prob >= 0.6) & (harmonic_ratio >= 0.45) & (flatness <= 0.35) & in_vocal_range
        vocal_presence_curve = _downsample_curve(pyin_times, vocal_mask.astype(float))
        likely_vocal_intervals = _intervals_from_mask(pyin_times, vocal_mask, min_duration=1.75, pad=0.08)
        vocal_presence_ratio = round(float(np.mean(vocal_mask.astype(float))) if len(vocal_mask) else 0.0, 3)
    except Exception:
        vocal_presence_curve = []
        likely_vocal_intervals = []
        vocal_presence_ratio = 0.0

    climax_time = energy_curve[max(range(len(energy_curve)), key=lambda i: energy_curve[i][1])][0] if energy_curve else duration
    phrase_grid_sec = round(max(2.0, min(16.0, (60.0 / max(60.0, tempo_val)) * 8.0)), 2)

    # Heuristic downbeats: assume 4/4 meter, every 4th beat is a downbeat. This is
    # not perfect — waltzes, 6/8, and syncopated pop will be wrong — but for the
    # vast majority of skating music (and as guidance to the LLM) it's useful,
    # and the LLM still ultimately listens to the audio.
    downbeat_times = [round(t, 3) for t in beat_times[::4]] if beat_times else []

    return {
        'duration_sec': duration,
        'tempo_bpm': round(tempo_val, 1),
        'num_beats': len(beat_times),
        'beat_times': [round(t, 3) for t in beat_times],
        'downbeat_times': downbeat_times,
        'section_boundaries': section_boundaries,
        'energy_curve': energy_curve,
        'onset_curve': onset_curve,
        'brightness_curve': brightness_curve,
        'estimated_climax_time': round(float(climax_time), 2),
        'phrase_grid_sec': phrase_grid_sec,
        'estimated_key': key_info,
        'chord_change_density_per_min': chord_change_density_per_min,
        'chord_change_curve': chord_change_curve,
        'vocal_presence_ratio': vocal_presence_ratio,
        'vocal_presence_curve': vocal_presence_curve,
        'likely_vocal_intervals': likely_vocal_intervals,
    }

def snap_to_boundaries(value: float, boundaries: list[float], duration: float, tolerance: float = 0.9) -> float:
    anchors = [0.0, *boundaries, duration]
    nearest = min(anchors, key=lambda x: abs(x - value))
    if abs(nearest - value) <= tolerance:
        return round(float(nearest), 2)
    return round(float(value), 2)


def _nearest(value: float, candidates: list[float]) -> tuple[float, float]:
    """Return (nearest_candidate, absolute_distance). If candidates is empty, returns (value, inf)."""
    if not candidates:
        return value, float("inf")
    best = min(candidates, key=lambda x: abs(x - value))
    return float(best), abs(best - value)


def snap_to_grid(
    value: float,
    boundaries: list[float],
    beats: list[float],
    duration: float,
    section_tolerance: float = 0.75,
    beat_tolerance: float = 0.25,
) -> float:
    """Coarse-then-fine snap: first to a section boundary (big semantic jump),
    then to the nearest beat (fine perceptual alignment). A pure beat snap still
    happens even when no section boundary is in range — this is what stops cuts
    from landing mid-beat, which is the loudest audible tell of an edit."""
    anchors = [0.0, *(boundaries or []), duration]
    snapped, dist = _nearest(value, anchors)
    if dist > section_tolerance:
        snapped = value
    if beats:
        beat, bdist = _nearest(snapped, beats)
        if bdist <= beat_tolerance:
            snapped = beat
    return round(float(max(0.0, min(duration, snapped))), 3)


def _interval_contains(intervals: list[dict], t: float) -> dict | None:
    for iv in intervals or []:
        try:
            s = float(iv.get("start", 0.0))
            e = float(iv.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if s <= t <= e:
            return {"start": s, "end": e}
    return None


def avoid_vocal(
    value: float,
    vocal_intervals: list[dict],
    beats: list[float],
    boundaries: list[float],
    duration: float,
    search_radius: float = 1.5,
) -> float:
    """If `value` lands inside a vocal interval, move it to the closest non-vocal
    moment within ±search_radius, preferring the outer edge of that interval
    (silence just before or just after the sung phrase). Then re-snap to the
    beat grid so the moved point still lands on a strong rhythmic position."""
    iv = _interval_contains(vocal_intervals, value)
    if iv is None:
        return value
    # Candidate "escape" points: just before the interval starts, or just after it ends.
    candidates = []
    if value - iv["start"] <= search_radius:
        candidates.append(max(0.0, iv["start"] - 0.05))
    if iv["end"] - value <= search_radius:
        candidates.append(min(duration, iv["end"] + 0.05))
    if not candidates:
        # Too deep inside a long vocal phrase — leave the caller to penalize rather than teleport.
        return value
    target = min(candidates, key=lambda x: abs(x - value))
    return snap_to_grid(target, boundaries, beats, duration)


def interpolate_curve(curve: list[list[float]], t: float) -> float:
    if not curve:
        return 0.0
    if t <= curve[0][0]:
        return float(curve[0][1])
    if t >= curve[-1][0]:
        return float(curve[-1][1])
    for i in range(1, len(curve)):
        t0, v0 = curve[i - 1]
        t1, v1 = curve[i]
        if t <= t1:
            span = max(0.001, t1 - t0)
            ratio = (t - t0) / span
            return float(v0 + (v1 - v0) * ratio)
    return float(curve[-1][1])


def average_curve(curve: list[list[float]], start: float, end: float) -> float:
    if end <= start:
        return 0.0
    samples = max(3, int((end - start) / 2))
    pts = [start + (end - start) * i / (samples - 1) for i in range(samples)]
    vals = [interpolate_curve(curve, p) for p in pts]
    return round(float(sum(vals) / len(vals)), 3)


def _beat_phase_penalty(t: float, beats: list[float], beat_period: float) -> float:
    """0.0 = exactly on a beat, 1.0 = exactly between two beats."""
    if not beats or beat_period <= 0:
        return 0.0
    _, dist = _nearest(t, beats)
    # normalize by half a beat — anything beyond that is "maximally off-beat"
    return min(1.0, dist / max(0.001, beat_period * 0.5))


def _vocal_cut_penalty(t: float, vocal_intervals: list[dict]) -> float:
    """1.0 if the cut lands inside a sung phrase, 0.0 otherwise."""
    return 1.0 if _interval_contains(vocal_intervals, t) is not None else 0.0


def transition_penalty(analysis: dict, start: float, end: float) -> float:
    """Higher = more audible cut. Combines envelope discontinuity (existing signals),
    beat-phase alignment (cuts off the beat are very audible), and vocal intrusion
    (cuts through a sung phrase are maximally audible)."""
    energy_curve = analysis.get("energy_curve", [])
    onset_curve = analysis.get("onset_curve", [])
    brightness_curve = analysis.get("brightness_curve", [])
    duration = float(analysis.get("duration_sec") or 0.0) or (end + 4.0)
    left_energy = average_curve(energy_curve, max(0.0, start - 4.0), start)
    right_energy = average_curve(energy_curve, end, min(duration, end + 4.0))
    left_onset = average_curve(onset_curve, max(0.0, start - 4.0), start)
    right_onset = average_curve(onset_curve, end, min(duration, end + 4.0))
    left_bright = average_curve(brightness_curve, max(0.0, start - 4.0), start)
    right_bright = average_curve(brightness_curve, end, min(duration, end + 4.0))
    envelope = (
        abs(left_energy - right_energy) * 0.40
        + abs(left_onset - right_onset) * 0.20
        + abs(left_bright - right_bright) * 0.15
    )

    beats = analysis.get("beat_times") or []
    tempo = float(analysis.get("tempo_bpm") or 0.0)
    beat_period = 60.0 / tempo if tempo > 0 else 0.5
    phase = (_beat_phase_penalty(start, beats, beat_period) + _beat_phase_penalty(end, beats, beat_period)) * 0.5

    vocals = analysis.get("likely_vocal_intervals") or []
    voc = (_vocal_cut_penalty(start, vocals) + _vocal_cut_penalty(end, vocals)) * 0.5

    return round(envelope + phase * 0.30 + voc * 0.40, 3)


def build_cut_candidates(analysis: dict, target_sec: float, aggressiveness: int = 50) -> list[dict]:
    duration = float(analysis["duration_sec"])
    needed_cut = max(0.0, duration - target_sec)
    if needed_cut <= 1.5:
        return []

    boundaries = list(analysis.get("section_boundaries", []))
    beats = list(analysis.get("beat_times", []))
    vocals = list(analysis.get("likely_vocal_intervals", []))
    anchors = [0.0, *boundaries, duration]
    opening_guard = min(20.0, max(10.0, duration * 0.08))
    ending_guard = min(25.0, max(12.0, duration * 0.10))
    climax_time = float(analysis.get("estimated_climax_time", duration))
    phrase_grid = float(analysis.get("phrase_grid_sec", 8.0))
    candidates: list[dict] = []

    for i in range(len(anchors) - 1):
        raw_start = float(anchors[i])
        raw_end = float(anchors[i + 1])
        start = _align_boundary(raw_start, boundaries, beats, vocals, duration)
        end = _align_boundary(raw_end, boundaries, beats, vocals, duration)
        seg_len = end - start
        if seg_len < max(6.0, phrase_grid * 0.75):
            continue
        if start < opening_guard or end > duration - ending_guard:
            continue
        if abs((start + end) / 2 - climax_time) < max(12.0, phrase_grid * 1.5):
            continue
        energy = average_curve(analysis.get("energy_curve", []), start, end)
        onset = average_curve(analysis.get("onset_curve", []), start, end)
        brightness = average_curve(analysis.get("brightness_curve", []), start, end)
        penalty = transition_penalty(analysis, start, end)
        repetitiveness = round(max(0.0, 1.0 - abs(energy - statistics.median([p[1] for p in analysis.get("energy_curve", [[0,0.5]])]))), 3)
        score = round((1.0 - penalty) * 0.45 + repetitiveness * 0.25 + (1.0 - onset) * 0.15 + (1.0 - energy) * 0.15, 3)
        candidates.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration_sec": round(seg_len, 2),
            "avg_energy": energy,
            "avg_onset": onset,
            "avg_brightness": brightness,
            "transition_penalty": penalty,
            "score": score,
            "label": "whole_section",
            "reason_hint": "Lower-risk full-section removal candidate aligned to detected structure.",
        })

    # Add mid-section phrase-sized removals for harder cuts
    window = max(phrase_grid, min(needed_cut * (0.65 + aggressiveness / 200.0), 24.0))
    for b0, b1 in zip(anchors[:-1], anchors[1:]):
        if b1 - b0 < window + phrase_grid:
            continue
        start = _align_boundary(b0 + phrase_grid, boundaries, beats, vocals, duration)
        end = _align_boundary(min(b1 - phrase_grid, start + window), boundaries, beats, vocals, duration)
        if end - start < max(6.0, phrase_grid * 0.75):
            continue
        if start < opening_guard or end > duration - ending_guard:
            continue
        if abs((start + end) / 2 - climax_time) < max(12.0, phrase_grid * 1.5):
            continue
        penalty = transition_penalty(analysis, start, end)
        score = round((1.0 - penalty) * 0.60 + (1.0 - average_curve(analysis.get("energy_curve", []), start, end)) * 0.25 + (1.0 - average_curve(analysis.get("onset_curve", []), start, end)) * 0.15, 3)
        candidates.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration_sec": round(end - start, 2),
            "avg_energy": average_curve(analysis.get("energy_curve", []), start, end),
            "avg_onset": average_curve(analysis.get("onset_curve", []), start, end),
            "avg_brightness": average_curve(analysis.get("brightness_curve", []), start, end),
            "transition_penalty": penalty,
            "score": score,
            "label": "phrase_window",
            "reason_hint": "Phrase-sized internal cut candidate for tighter timing control.",
        })

    dedup = {}
    for c in candidates:
        dedup[(c["start"], c["end"])] = c
    ranked = sorted(dedup.values(), key=lambda c: (-c["score"], c["transition_penalty"], abs(c["duration_sec"] - needed_cut)))
    return ranked[:12]


# ────────────────────────────────────────────────────────────────────────────
# AI cut suggestions
# ────────────────────────────────────────────────────────────────────────────

def guess_audio_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime and mime.startswith("audio/"):
        return mime
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(path.suffix.lower(), "audio/mpeg")


def build_music_edit_prompt(analysis: dict, target_sec: float, discipline: str, program_style: str, aggressiveness: int, num_options: int) -> str:
    duration = float(analysis["duration_sec"])
    needed_cut = max(0.0, duration - target_sec)
    discipline_line = f"Skating discipline / context: {discipline}\n" if discipline else ""
    candidate_preview = [
        {
            "start": c["start"],
            "end": c["end"],
            "duration_sec": c["duration_sec"],
            "transition_penalty": c["transition_penalty"],
            "score": c["score"],
            "label": c["label"],
        }
        for c in analysis.get("local_candidates", [])[:8]
    ]
    beat_period = round(60.0 / max(60.0, float(analysis.get('tempo_bpm') or 120.0)), 3)
    # Keep the prompt compact: 40 beats and 24 downbeats is enough for the model
    # to understand the grid; tempo lets it extrapolate the rest, and beyond this
    # the extra tokens slow inference without improving cut quality.
    beats_preview = (analysis.get('beat_times') or [])[:40]
    downbeats_preview = (analysis.get('downbeat_times') or [])[:24]
    return f"""You are an elite figure-skating music editor. The top priority is that edits are INAUDIBLE — a listener who has never heard the original track should not be able to tell where the cuts are. You receive the actual audio plus local signal-processing analysis. Build cut plans that preserve phrasing, momentum, and climax, and place every cut boundary on a musically safe join.

TASK:
- Current duration: {duration:.1f} seconds
- Target duration: {target_sec:.1f} seconds
- Need to remove about: {needed_cut:.1f} seconds
- Requested style: {program_style} — {style_description(program_style)}
- Aggressiveness: {aggressiveness}/100
- Return exactly {num_options} ranked edit plans
{discipline_line}
LOCAL ANALYSIS SUMMARY:
- Tempo BPM: {analysis['tempo_bpm']} (one beat ≈ {beat_period}s)
- Phrase grid seconds: {analysis.get('phrase_grid_sec')}
- Section boundaries: {analysis.get('section_boundaries', [])}
- Estimated climax time: {analysis.get('estimated_climax_time')}
- Estimated key/mode: {analysis.get('estimated_key', {}).get('label')} (confidence {analysis.get('estimated_key', {}).get('confidence')})
- Chord-change density per minute: {analysis.get('chord_change_density_per_min')}
- Likely vocal intervals (DO NOT cut inside these): {analysis.get('likely_vocal_intervals', [])[:20]}
- Beat grid (first 180 beats): {beats_preview}
- Downbeat grid (assumed 4/4, first 60 bar-starts — prefer these for cut boundaries): {downbeats_preview}
- Energy curve: {analysis.get('energy_curve', [])}
- Onset/activity curve: {analysis.get('onset_curve', [])}
- Brightness curve: {analysis.get('brightness_curve', [])}
- Chord-change curve: {analysis.get('chord_change_curve', [])}
- Vocal-presence curve: {analysis.get('vocal_presence_curve', [])}
- Low-risk local cut candidates (already snapped to section + beat + vocal-safe): {candidate_preview}

CUT ALIGNMENT RULES (hard):
A. Every cut boundary MUST land on a downbeat (preferred) or at minimum on a beat. If you cannot justify the exact position in terms of the beat/downbeat grid, pick a nearby grid point instead.
B. NO cut boundary may fall inside a listed vocal interval. A boundary must be in an instrumental moment — either before the vocal phrase starts or after it ends. If that forces the plan to remove less than the target, take it: a short plan that never cuts through a lyric beats a tight plan that does.
C. Match context across the join: the bar-count, metric feel, and harmonic context (the chord-change curve) should be similar immediately before the cut-out point and immediately after the cut-in point. A verse cannot be stitched to a bridge unless the chord/tempo context matches at the seam.
D. Where possible, cut "between rhymes": out at the end of one lyric/phrase, back in at the start of the equivalent position of the next phrase — NOT one bar earlier, not one bar later. This is the single biggest reason edits sound invisible.
E. Crossfades of 150-400ms are available, but they do NOT fix a cut that lands mid-word or mid-beat — treat the crossfade as polish on an already-correct cut, not a fix.

OTHER REQUIREMENTS:
1. Listen to the actual audio. Detect repeated material (verse 1 ≈ verse 2 ⇒ safe to drop one), orchestral swells, drops, cadence points, and whether a seam will feel natural.
2. Preserve a coherent opening and the final payoff. Do not cut away the ending climax unless the audio clearly supports that.
3. Keep cuts non-overlapping and inside the track.
4. Favor plans that keep the emotional arc intact for figure skating performance.
5. Total removed time for each plan should land within ±3 seconds of the target, unless an audible cut is the only way — quality of the seam always beats hitting the second exactly.
6. In each cut's "reason" field, briefly cite the alignment: which downbeat it snaps to and which phrase you are joining to which. This forces your own checking.

Return JSON only, no markdown fences:
{{
  "rationale": "overall strategy across the ranked plans",
  "structure_notes": "what the song is doing musically",
  "warnings": "caveats or empty string",
  "options": [
    {{
      "title": "short descriptive name",
      "summary": "why this plan works musically",
      "confidence": 0.0,
      "transition_risk": "low|medium|high",
      "removed_sec": 0.0,
      "cuts": [
        {{"start": 0.0, "end": 0.0, "reason": "why this section should be removed"}}
      ]
    }}
  ],
  "best_option_index": 0
}}"""


def parse_json_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise RuntimeError(f"Model returned non-JSON: {text[:300]}") from e
        return json.loads(m.group(0))


def _align_boundary(
    t: float,
    boundaries: list[float],
    beats: list[float],
    vocals: list[dict],
    duration: float,
) -> float:
    """Run a cut boundary through the full alignment pipeline:
    section-snap -> beat-snap -> vocal-escape -> beat-resnap. This is used for
    both AI-proposed cuts and locally-generated candidate cuts so they converge
    on the same set of musically-safe positions."""
    aligned = snap_to_grid(t, boundaries, beats, duration)
    aligned = avoid_vocal(aligned, vocals, beats, boundaries, duration)
    return aligned


def _boundary_candidates(t: float, beats: list[float], window: float = 0.25, max_count: int = 3) -> list[float]:
    """Return up to `max_count` beat-grid candidates near `t` within ±window
    seconds, deduplicated and sorted by distance from `t`. Always includes `t`
    itself so the pair-optimizer can choose 'don't move'."""
    out = {round(float(t), 3): None}
    for b in beats or []:
        if abs(b - t) <= window:
            out[round(float(b), 3)] = None
    ranked = sorted(out.keys(), key=lambda x: abs(x - t))
    return ranked[:max_count + 1]  # +1 to include `t` alongside up to max_count beat anchors


def sanitize_cut_list(cuts: list, analysis: dict) -> list[dict]:
    duration = float(analysis["duration_sec"])
    boundaries = list(analysis.get("section_boundaries", []))
    beats = list(analysis.get("beat_times", []))
    vocals = list(analysis.get("likely_vocal_intervals", []))
    valid = []
    for c in cuts or []:
        try:
            s = max(0.0, float(c["start"]))
            e = min(duration, float(c["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        s = _align_boundary(s, boundaries, beats, vocals, duration)
        e = _align_boundary(e, boundaries, beats, vocals, duration)
        if e <= s + 0.1:
            continue
        valid.append({
            "start": round(s, 2),
            "end": round(e, 2),
            "reason": str(c.get("reason", ""))[:240],
            "transition_penalty": transition_penalty(analysis, s, e),
        })
    valid.sort(key=lambda c: (c["start"], c["end"]))
    merged = []
    for c in valid:
        if not merged or c["start"] >= merged[-1]["end"]:
            merged.append(c)

    # Second pass: jointly optimize each adjacent boundary pair that becomes a
    # splice in the output. For cuts[i] and cuts[i+1], the splice is between
    # cuts[i].end and cuts[i+1].start — pick the combination within the beat
    # tolerance that minimizes transition_penalty over the splice pair rather
    # than each side alone. Also scores the leading/trailing splices against
    # the start/end of the track so fades land on compatible phases.
    for i, cut in enumerate(merged):
        end_candidates = _boundary_candidates(cut["end"], beats)
        start_of_next = merged[i + 1]["start"] if i + 1 < len(merged) else None
        if start_of_next is not None:
            start_candidates = _boundary_candidates(start_of_next, beats)
            best = (cut["end"], start_of_next, transition_penalty(analysis, cut["end"], start_of_next))
            for e_cand in end_candidates:
                if e_cand <= cut["start"] + 0.1:
                    continue
                for s_cand in start_candidates:
                    if s_cand <= e_cand + 0.1:
                        continue
                    if i + 2 < len(merged) and s_cand >= merged[i + 2]["start"] - 0.1:
                        continue
                    pen = transition_penalty(analysis, e_cand, s_cand)
                    if pen < best[2]:
                        best = (e_cand, s_cand, pen)
            cut["end"] = round(best[0], 2)
            merged[i + 1]["start"] = round(best[1], 2)
            cut["splice_penalty"] = round(best[2], 3)
        else:
            # Trailing splice — evaluated against end-of-track.
            cut["splice_penalty"] = round(transition_penalty(analysis, cut["end"], duration), 3)

    return merged


def normalize_plan_response(parsed: dict, analysis: dict, model_name: str, provider: str) -> dict:
    options = []
    raw_options = parsed.get("options")
    if not raw_options and parsed.get("cuts"):
        raw_options = [{
            "title": "Primary plan",
            "summary": parsed.get("rationale", ""),
            "confidence": parsed.get("confidence", 0.6),
            "transition_risk": parsed.get("transition_risk", "medium"),
            "removed_sec": sum(max(0.0, float(c.get("end", 0)) - float(c.get("start", 0))) for c in parsed.get("cuts", [])),
            "cuts": parsed.get("cuts", []),
        }]
    for idx, opt in enumerate(raw_options or []):
        cuts = sanitize_cut_list(opt.get("cuts", []), analysis)
        removed = round(sum(c["end"] - c["start"] for c in cuts), 2)
        options.append({
            "title": str(opt.get("title") or f"Plan {idx + 1}")[:80],
            "summary": str(opt.get("summary") or "")[:300],
            "confidence": max(0.0, min(1.0, float(opt.get("confidence", 0.55) or 0.55))),
            "transition_risk": str(opt.get("transition_risk") or "medium")[:20],
            "removed_sec": removed,
            "cuts": cuts,
        })
    best_idx = int(parsed.get("best_option_index", 0) or 0) if options else 0
    best_idx = min(max(best_idx, 0), max(0, len(options) - 1))
    return {
        "rationale": str(parsed.get("rationale") or "")[:600],
        "structure_notes": str(parsed.get("structure_notes") or "")[:300],
        "warnings": str(parsed.get("warnings") or "")[:300],
        "options": options,
        "best_option_index": best_idx,
        "model": model_name,
        "provider": provider,
    }


def ai_suggest_cuts_with_gemini(src_path: Path, analysis: dict, target_sec: float, discipline: str = "", program_style: str = "balanced", aggressiveness: int = 50, num_options: int = 3) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai") from e

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY env var is not set. Add a Gemini API key to use full multimodal audio analysis.")

    client = genai.Client()
    prompt = build_music_edit_prompt(analysis, target_sec, discipline, program_style, aggressiveness, num_options)
    audio_part = types.Part.from_bytes(data=src_path.read_bytes(), mime_type=guess_audio_mime(src_path))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[audio_part, prompt],
    )
    parsed = parse_json_object(response.text or "")
    return normalize_plan_response(parsed, analysis, GEMINI_MODEL, "gemini")


def build_audition_optimize_prompt(
    analysis: dict,
    original_plan: dict,
    target_sec: float,
    discipline: str,
    program_style: str,
    aggressiveness: int,
) -> str:
    """Prompt that asks Gemini to LISTEN to a rendered audition preview, detect
    audible seams, and emit a refined cut list in the source timeline. The model
    does not receive the source audio again — the analysis summary captures its
    structure, and the audition is what needs critique."""
    duration = float(analysis.get("duration_sec") or 0.0)
    beat_period = round(60.0 / max(60.0, float(analysis.get("tempo_bpm") or 120.0)), 3)
    beats_preview = (analysis.get("beat_times") or [])[:40]
    downbeats_preview = (analysis.get("downbeat_times") or [])[:24]
    vocals = (analysis.get("likely_vocal_intervals") or [])[:20]
    original_cuts = [
        {
            "index": i + 1,
            "start": round(float(c.get("start", 0.0)), 2),
            "end": round(float(c.get("end", 0.0)), 2),
            "reason": str(c.get("reason") or "")[:200],
        }
        for i, c in enumerate(original_plan.get("cuts") or [])
    ]
    original_removed = round(
        sum(max(0.0, c["end"] - c["start"]) for c in original_cuts), 2,
    )
    target_remove_sec = round(max(0.0, duration - target_sec), 2)
    discipline_line = f"Skating discipline / context: {discipline}\n" if discipline else ""
    return f"""You are listening to a RENDERED PREVIEW of an edited figure-skating program. Your job is to identify AUDIBLE edit seams — places where the listener can tell a cut happened — and propose a refined cut list that eliminates them.

CUT SEMANTICS (CRITICAL — READ FIRST):
A "cut" is a REGION TO REMOVE from the source track. Each entry `{{"start": S, "end": E}}` means the audio between S and E is DELETED. The final program equals the source track with every cut region removed, joined with crossfades.
- Source track duration: {duration:.2f}s
- Target program duration: {target_sec:.1f}s
- Therefore total time to REMOVE (sum of all (end - start) across your cuts) must be ≈ {target_remove_sec:.1f}s.
- The ORIGINAL plan removes {original_removed:.1f}s across {len(original_cuts)} cut region(s).
- DO NOT return "keep" ranges. Returning cuts that sum to {target_sec:.1f}s (the target length) means you inverted the semantics and the output will be almost entirely deleted.

AUDIBLE-SEAM CHECKLIST (listen for these in the preview):
- Cuts that land off-beat (the grid skips or "trips").
- Cuts that sever a vocal phrase (word or syllable truncated).
- Key or mode jumps across the seam (chord A → unrelated chord B).
- Dynamics / reverb-tail mismatches (loud to quiet, long reverb to dry).
- Crossfade that sounds "smeared" because two different beats are overlapping.

WHAT YOU RECEIVE:
- The audition preview as the attached audio. This preview has already had the original cut regions removed, crossfaded, and faded. Any audible seam you hear in this preview is a join the original cuts created.
- The ORIGINAL cut plan (the list of regions that were removed) below. Cut indices (1-based) refer to these.
- Structural context from the SOURCE timeline: tempo, beat grid, section boundaries, vocal intervals, key estimate. All cut times in your response MUST be in the source timeline, NOT the audition's compressed timeline.

SOURCE TIMELINE CONTEXT:
- Duration: {duration:.2f}s
- Tempo BPM: {analysis.get('tempo_bpm')} (one beat ≈ {beat_period}s)
- Section boundaries: {analysis.get('section_boundaries', [])}
- Estimated key/mode: {analysis.get('estimated_key', {}).get('label')} (confidence {analysis.get('estimated_key', {}).get('confidence')})
- Likely vocal intervals (DO NOT place cut boundaries inside these): {vocals}
- First 40 beats: {beats_preview}
- First 24 downbeats (preferred cut landing points): {downbeats_preview}

ORIGINAL CUT PLAN (regions that were REMOVED; cut indices 1-based; times are source-timeline seconds):
{original_cuts}

TARGET:
- Target program duration: {target_sec:.1f}s
- Total to remove (budget): {target_remove_sec:.1f}s ± 3s
- Aggressiveness: {aggressiveness}/100
- Style: {program_style} — {style_description(program_style)}
{discipline_line}
RULES FOR THE REFINED PLAN:
1. Return a COMPLETE list of REMOVE regions in source-timeline seconds (not deltas). A refined plan replaces the original plan wholesale. Each region's (end - start) is the duration that gets deleted.
2. The sum of all (end - start) values in your `cuts` array MUST be approximately {target_remove_sec:.1f}s (±3s). If your sum is closer to {target_sec:.1f}s, you inverted the semantics — re-read CUT SEMANTICS above.
3. Every cut boundary MUST land on a downbeat if possible, otherwise on a beat. NO boundary may fall inside a listed vocal interval.
4. Cut regions must not overlap, and must stay within [0, {duration:.2f}].
5. If the preview sounds CLEAN (no audible seams), return the ORIGINAL cuts unchanged with confidence ≥ 0.9 and an empty `changes` array. Do not invent problems. The original cuts removed {original_removed:.1f}s total.
6. If you move a cut, explain in `changes` which cut index you moved and why (e.g. "cut #2 was mid-vocal at 63.1s; moved start to downbeat 65.27s to clear the phrase").
7. Do NOT add new cut regions just to hit the budget exactly. Quality of seams beats matching duration exactly.

Return JSON only (no markdown fences):
{{
  "title": "short descriptive label — 'Refined: <what you changed>' or 'Unchanged: clean'",
  "summary": "one sentence on the audible verdict",
  "confidence": 0.0,
  "transition_risk": "low|medium|high",
  "changes": ["cut #2 start shifted +1.3s to downbeat 65.27s to clear vocal"],
  "cuts": [
    {{"start": 0.0, "end": 0.0, "reason": "what audible problem this removal avoids or resolves"}}
  ]
}}"""


def ai_optimize_audition_with_gemini(
    audition_path: Path,
    analysis: dict,
    original_plan: dict,
    target_sec: float,
    discipline: str = "",
    program_style: str = "balanced",
    aggressiveness: int = 50,
) -> dict:
    """Send the rendered audition preview + textual context to Gemini, return a
    refined plan dict. Caller is responsible for running the returned cut list
    through `sanitize_cut_list` before rendering."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("google-genai SDK not installed. Run: pip install google-genai") from e

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY env var is not set. Add a Gemini API key to use audition optimization.")

    client = genai.Client()
    prompt = build_audition_optimize_prompt(
        analysis=analysis,
        original_plan=original_plan,
        target_sec=target_sec,
        discipline=discipline,
        program_style=program_style,
        aggressiveness=aggressiveness,
    )
    audio_part = types.Part.from_bytes(
        data=audition_path.read_bytes(),
        mime_type=guess_audio_mime(audition_path),
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[audio_part, prompt],
    )
    return parse_json_object(response.text or "")


def ai_suggest_cuts_with_claude_fallback(analysis: dict, target_sec: float, discipline: str = "", program_style: str = "balanced", aggressiveness: int = 50, num_options: int = 3) -> dict:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic SDK not installed. Run: pip install anthropic") from e

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY env var is not set. Claude fallback is unavailable.")

    prompt = build_music_edit_prompt(analysis, target_sec, discipline, program_style, aggressiveness, num_options) + "\n\nIMPORTANT: You do not have the raw audio in this fallback mode. Be conservative when local analysis leaves doubt."
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    parsed = parse_json_object(text)
    return normalize_plan_response(parsed, analysis, CLAUDE_MODEL, "claude")


def ai_suggest_cuts(src_path: Path, analysis: dict, target_sec: float, discipline: str = "", program_style: str = "balanced", aggressiveness: int = 50, num_options: int = 3, provider: str = AI_PROVIDER) -> dict:
    provider = (provider or AI_PROVIDER).strip().lower()
    if provider == "gemini":
        return ai_suggest_cuts_with_gemini(src_path, analysis, target_sec, discipline, program_style, aggressiveness, num_options)
    if provider == "claude":
        return ai_suggest_cuts_with_claude_fallback(analysis, target_sec, discipline, program_style, aggressiveness, num_options)
    raise RuntimeError(f"Unsupported ai_provider '{provider}'. Use 'gemini' or 'claude'.")


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Skating Music Editor")
    print(f"  Workspace: {WORK_DIR}")
    print(f"  AI provider: {AI_PROVIDER}")
    print(f"  Gemini model: {GEMINI_MODEL}")
    print(f"  Claude model: {CLAUDE_MODEL}")
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    print(f"  GEMINI_API_KEY: {'set' if has_gemini else 'NOT SET'}")
    print(f"  ANTHROPIC_API_KEY: {'set' if has_claude else 'NOT SET'}")
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
