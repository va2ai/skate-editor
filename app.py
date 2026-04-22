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
import uuid
from pathlib import Path

from typing import Iterable

from flask import Flask, jsonify, render_template, request, send_file
from pydub import AudioSegment

BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "workspace"
WORK_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "aac", "flac", "ogg"}
MAX_UPLOAD_MB = 50

# Default multimodal provider. Gemini is the primary full-audio path.
AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


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


def render_audio_from_cuts(
    src: Path,
    cuts: list[dict],
    crossfade_ms: int = 250,
    fade_in_ms: int = 100,
    fade_out_ms: int = 2000,
    target_format: str = "mp3",
    bitrate: str = "192k",
    output_id: str | None = None,
) -> dict:
    audio = AudioSegment.from_file(src)
    total_ms = len(audio)
    norm_cuts = normalize_cuts(cuts, total_ms / 1000.0)

    keep_regions: list[tuple[int, int]] = []
    cursor = 0
    for c in norm_cuts:
        s = max(0, int(c["start"] * 1000))
        e = min(total_ms, int(c["end"] * 1000))
        if s > cursor:
            keep_regions.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < total_ms:
        keep_regions.append((cursor, total_ms))

    if not keep_regions:
        raise ValueError("Nothing left to keep — your cuts cover the entire song")

    out: AudioSegment | None = None
    for (s, e) in keep_regions:
        clip = audio[s:e]
        if out is None:
            out = clip
        else:
            cf = min(crossfade_ms, len(out), len(clip))
            try:
                out = out.append(clip, crossfade=cf)
            except Exception:
                out = out + clip

    if out is None:
        raise ValueError("No audio produced")

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
    return render_template("index.html")


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

    Body: {
        "file_id": "...",
        "ext": "mp3",
        "target_sec": 150,
        "discipline": "Junior FS",
        "program_style": "balanced",
        "aggressiveness": 50,
        "num_options": 3,
        "use_ai": true
    }
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

    try:
        analysis = analyze_audio(src)
    except ImportError as e:
        return jsonify({"error": f"librosa not installed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"audio analysis failed: {e}"}), 500

    suggestion = None
    ai_error = None
    if target_sec:
        analysis["local_candidates"] = build_cut_candidates(analysis, float(target_sec), aggressiveness)
        analysis["recommended_opening_guard_sec"] = min(20.0, max(10.0, analysis["duration_sec"] * 0.08))
        analysis["recommended_ending_guard_sec"] = min(25.0, max(12.0, analysis["duration_sec"] * 0.1))
    try:
        waveform_path = generate_waveform_image(src, file_id=file_id, analysis=analysis)
        analysis["waveform_image_url"] = f"/waveform_image/{file_id}.{ext}?v={int(waveform_path.stat().st_mtime)}"
    except Exception as e:
        analysis["waveform_image_error"] = str(e)
    if use_ai and target_sec:
        try:
            suggestion = ai_suggest_cuts(
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
            ai_error = str(e)

    return jsonify({
        "analysis": analysis,
        "suggestion": suggestion,
        "ai_error": ai_error,
    })


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
    crossfade_ms = int(data.get("crossfade_ms", 250))
    fade_out_ms = int(data.get("fade_out_ms", 2000))
    fade_in_ms = int(data.get("fade_in_ms", 100))
    target_format = data.get("target_format", "mp3")
    bitrate = data.get("bitrate", "192k")

    src = WORK_DIR / f"{file_id}.{ext}"
    if not src.exists():
        return jsonify({"error": "source file not found"}), 404

    try:
        result = render_audio_from_cuts(
            src=src,
            cuts=cuts,
            crossfade_ms=crossfade_ms,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            target_format=target_format,
            bitrate=bitrate,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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

    return {
        'duration_sec': duration,
        'tempo_bpm': round(tempo_val, 1),
        'num_beats': len(beat_times),
        'beat_times': [round(t, 3) for t in beat_times],
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


def transition_penalty(analysis: dict, start: float, end: float) -> float:
    energy_curve = analysis.get("energy_curve", [])
    onset_curve = analysis.get("onset_curve", [])
    brightness_curve = analysis.get("brightness_curve", [])
    left_energy = average_curve(energy_curve, max(0.0, start - 4.0), start)
    right_energy = average_curve(energy_curve, end, min(analysis["duration_sec"], end + 4.0))
    left_onset = average_curve(onset_curve, max(0.0, start - 4.0), start)
    right_onset = average_curve(onset_curve, end, min(analysis["duration_sec"], end + 4.0))
    left_bright = average_curve(brightness_curve, max(0.0, start - 4.0), start)
    right_bright = average_curve(brightness_curve, end, min(analysis["duration_sec"], end + 4.0))
    return round(abs(left_energy - right_energy) * 0.55 + abs(left_onset - right_onset) * 0.25 + abs(left_bright - right_bright) * 0.2, 3)


def build_cut_candidates(analysis: dict, target_sec: float, aggressiveness: int = 50) -> list[dict]:
    duration = float(analysis["duration_sec"])
    needed_cut = max(0.0, duration - target_sec)
    if needed_cut <= 1.5:
        return []

    boundaries = list(analysis.get("section_boundaries", []))
    anchors = [0.0, *boundaries, duration]
    opening_guard = min(20.0, max(10.0, duration * 0.08))
    ending_guard = min(25.0, max(12.0, duration * 0.10))
    climax_time = float(analysis.get("estimated_climax_time", duration))
    phrase_grid = float(analysis.get("phrase_grid_sec", 8.0))
    candidates: list[dict] = []

    for i in range(len(anchors) - 1):
        start = float(anchors[i])
        end = float(anchors[i + 1])
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
        start = snap_to_boundaries(b0 + phrase_grid, boundaries, duration, tolerance=0.75)
        end = snap_to_boundaries(min(b1 - phrase_grid, start + window), boundaries, duration, tolerance=0.75)
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
    return f"""You are an elite figure-skating music editor. You are receiving the actual audio file, plus local structure analysis generated from signal processing. Build strong cut plans that preserve musical phrasing, momentum, climax, and skateability.

TASK:
- Current duration: {duration:.1f} seconds
- Target duration: {target_sec:.1f} seconds
- Need to remove about: {needed_cut:.1f} seconds
- Requested style: {program_style}
- Aggressiveness: {aggressiveness}/100
- Return exactly {num_options} ranked edit plans
{discipline_line}
LOCAL ANALYSIS SUMMARY:
- Tempo BPM: {analysis['tempo_bpm']}
- Phrase grid seconds: {analysis.get('phrase_grid_sec')}
- Section boundaries: {analysis.get('section_boundaries', [])}
- Estimated climax time: {analysis.get('estimated_climax_time')}
- Estimated key/mode: {analysis.get('estimated_key', {}).get('label')} (confidence {analysis.get('estimated_key', {}).get('confidence')})
- Chord-change density per minute: {analysis.get('chord_change_density_per_min')}
- Likely vocal intervals: {analysis.get('likely_vocal_intervals', [])[:12]}
- Energy curve: {analysis.get('energy_curve', [])}
- Onset/activity curve: {analysis.get('onset_curve', [])}
- Brightness curve: {analysis.get('brightness_curve', [])}
- Chord-change curve: {analysis.get('chord_change_curve', [])}
- Vocal-presence curve: {analysis.get('vocal_presence_curve', [])}
- Low-risk local cut candidates: {candidate_preview}

REQUIREMENTS:
1. Listen to the actual audio, not just the local analysis. Use what you hear to detect repeated material, lyrical transitions, orchestral swells, drops, cadence points, and whether joins will feel natural.
2. Preserve a coherent opening and the final payoff. Do not cut away the ending climax unless the audio clearly supports that.
3. Prefer cut boundaries near section or phrase changes unless there is a compelling musical reason not to.
4. Keep cuts non-overlapping and inside the track.
5. Each plan must be realistically editable with standard crossfades.
6. Favor plans that keep the emotional arc intact for figure skating performance.
7. If vocals or narrative cues make a candidate cut awkward, say so and avoid it.
8. Total removed time for each plan should land within ±3 seconds of the target unless the track makes that impossible.

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


def sanitize_cut_list(cuts: list, analysis: dict) -> list[dict]:
    duration = float(analysis["duration_sec"])
    boundaries = list(analysis.get("section_boundaries", []))
    valid = []
    for c in cuts or []:
        try:
            s = max(0.0, float(c["start"]))
            e = min(duration, float(c["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        s = snap_to_boundaries(s, boundaries, duration, tolerance=0.75)
        e = snap_to_boundaries(e, boundaries, duration, tolerance=0.75)
        if e <= s:
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
