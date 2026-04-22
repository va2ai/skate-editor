# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-page Flask app for editing figure-skating program music. The user supplies a track (file upload OR YouTube URL), the app runs librosa-based structural analysis, and Gemini listens to the audio to propose ranked cut plans. Each plan can be rendered to a preview "audition" MP3; the user can ask Gemini to listen to a rendered audition and propose a refined cut list; the final program is exported with crossfades via pydub/ffmpeg.

Backend: `app.py` (~1600 lines). Frontend: `templates/index.html` (~1270 lines, vanilla JS + wavesurfer.js v7 via CDN). No build step, no framework, no test suite.

## Running locally

```bash
pip install -r requirements.txt
# ffmpeg must be on PATH — pydub shells out to it; yt-dlp also uses it to extract audio.
# node must be on PATH — yt-dlp needs a JS runtime to solve YouTube's signature cipher;
# without it, downloads 403. (Configured via "js_runtimes": {"node": {}}.)
python app.py                   # dev: http://127.0.0.1:5000
```

Production / container:

```bash
docker build -t skate-editor .
docker run -p 5000:5000 -e GEMINI_API_KEY=... skate-editor
```

The Dockerfile installs `ffmpeg` + `nodejs` (yt-dlp JS runtime), binds gunicorn to `0.0.0.0:${PORT:-5000}`, and runs `--workers 1 --threads 4 --timeout 600`. The `--threads 4` is load-bearing: the NDJSON streaming endpoints spawn a heartbeat loop on the request thread and run the 60–120s Gemini call on a worker thread — a single-threaded worker deadlocks both.

## Deploying to Fly.io

`fly.toml` is in the repo. Scale-to-zero enabled (~5s cold start), persistent volume `skate_workspace` mounted at `/app/workspace` so uploads/renders/sidecars survive redeploys.

```bash
fly deploy                                              # redeploy code
fly logs                                                # tail
fly ssh console                                         # shell in
fly secrets set GEMINI_API_KEY=<key>                    # update key
fly volumes list                                        # see disk
```

App name and primary region are in `fly.toml:app` and `fly.toml:primary_region` — if you fork, change `app` to something globally unique and create a volume in the matching region with `fly volumes create skate_workspace --size 3 --region <region>`.

Environment variables (all optional):

- `AI_PROVIDER` — `gemini` (default) or `claude`
- `GEMINI_API_KEY` — required for the primary path (full-audio multimodal)
- `ANTHROPIC_API_KEY` — required only for the Claude text-only fallback
- `GEMINI_MODEL` (default `gemini-3-flash-preview`), `CLAUDE_MODEL` (default `claude-sonnet-4-6`)

## Architecture

### Request pipeline

```
POST /upload             → save to workspace/, decode with pydub, return file_id + duration
POST /download_url       → yt-dlp URL → extract to mp3 → same response shape as /upload
POST /analyze            → NDJSON stream: librosa analysis → local candidates → optional LLM plans
POST /render_auditions   → render N preview MP3s, one per AI plan, return preview/download URLs
POST /optimize_audition  → NDJSON stream: Gemini listens to a rendered preview, refines, re-renders
POST /process            → render the user's final cut list with crossfades + fades
GET  /audio/<id>.<ext>                    → stream raw/rendered audio
GET  /waveform_image/<id>.<ext>           → matplotlib PNG (cached, mtime-checked)
GET  /download/<id>.<fmt>                 → rendered output as attachment
GET  /download_source/<id>.<ext>?name=…   → source audio as attachment (name sanitized server-side)
GET  /build_id                            → JSON build id; used by client-side stale-cache detector
```

All server-side artifacts (uploads, rendered outputs, waveform PNGs, analysis sidecars) live in `workspace/` keyed by `uuid.hex` `file_id` or `output_id`. Nothing is cleaned up automatically — `.dockerignore` excludes the directory so it doesn't ship in images.

### Streaming endpoints (NDJSON)

`/analyze` and `/optimize_audition` return `application/x-ndjson` — one JSON object per line, with event types `status`, `heartbeat`, `result`, and `error`. This matters because:

- The slow work (Gemini multimodal inference, 60–120s per call) runs in a `threading.Thread`; the request thread yields a heartbeat line every 4s to keep the HTTP connection warm against browser/proxy idle timeouts.
- Clients MUST read the body as a stream (`r.body.getReader()` + line-buffer) and consume only the `type:"result"` line as the final payload. `res.json()` will throw — the body is multi-line, not a single JSON object.
- When adding or modifying these endpoints, preserve the `daemon=True` worker + `done.wait(timeout=4.0)` heartbeat loop. Wrap each additional slow step (AI call, re-render) in its own heartbeat window.

### Analysis pipeline (`analyze_audio`)

librosa extracts: tempo + beat grid, HPSS, chroma-cqt → agglomerative section boundaries, RMS energy, onset strength, spectral centroid (brightness), Krumhansl-Schmuckler key/mode, chroma-delta chord-change density, pYIN-based vocal detection. Curves are down-sampled (default 40 points) before being returned — the frontend and the LLM see the compact summary, not the raw frames. `downbeat_times` is a cheap heuristic (every 4th beat, assumed 4/4) — wrong for waltzes / 6/8 / heavy syncopation, but useful guidance; the AI still listens to the audio for ground truth.

After `/analyze` completes, the full analysis dict is written to `workspace/<file_id>.analysis.json` (sidecar). `/optimize_audition` reads from this sidecar instead of re-running librosa (which costs 20–90s). `load_analysis_sidecar(file_id)` returns `None` if absent → the endpoint returns 400 "run Analyze first".

### Cut alignment pipeline

Every cut boundary (AI-proposed and locally-generated candidate) runs through `_align_boundary(t, boundaries, beats, vocals, duration)`:

1. **`snap_to_grid`** — coarse snap to section boundary (±0.75s), then fine snap to nearest beat (±0.25s). This is the single biggest lever for inaudible cuts: landing between beats is the loudest cue that an edit happened.
2. **`avoid_vocal`** — if the snapped time falls inside a `likely_vocal_interval`, escape to the nearest non-vocal moment within ±1.5s and re-snap to the beat grid.

`transition_penalty(analysis, start, end)` combines envelope discontinuity (energy/onset/brightness across ±4s), **beat-phase** (0 on-beat, 1 exactly between beats), and **vocal intrusion** (1 if the boundary lands inside a sung phrase). `sanitize_cut_list` runs every AI-proposed cut through `_align_boundary` + `transition_penalty` before returning — never trust raw LLM coordinates.

### AI cut suggestions

Two provider paths, both routed through `ai_suggest_cuts`:

- **Gemini (primary)** — sends the full audio bytes + analysis summary + prompt via `types.Part.from_bytes`. This is the path that actually "listens" to the track.
- **Claude (fallback)** — text-only; gets the analysis summary with an explicit note that raw audio is unavailable.

Prompts are built by `build_music_edit_prompt`. Style keywords (`balanced`, `dramatic`, `lyrical`, `technical`, `aggressive`) are expanded to one-sentence descriptions from `STYLE_DESCRIPTIONS` before being injected into the prompt — **keep that dict in sync with `AI_STYLE_HINTS` in the template** (same keys, same wording). The user-facing hint under the style dropdown uses the same strings.

Responses go through `parse_json_object` (strips ```json fences, falls back to regex-extracting the outermost `{...}`), then `normalize_plan_response` → `sanitize_cut_list`.

### Audition optimization (`/optimize_audition`)

Each rendered audition card has a "✨ Optimize with AI" button. The server sends the rendered preview MP3 (not the source) to Gemini with `build_audition_optimize_prompt`, which frames the task as "listen to this preview, find audible seams, return a refined REMOVE list in source-timeline seconds". Non-obvious prompt requirements:

- **Cut semantics must be explicit**: `{start,end}` = REMOVE region, NOT keep region. The sum of `(end - start)` across cuts should equal `duration_sec - target_sec`, not `target_sec`. The first version of this prompt was ambiguous and Gemini returned keep-ranges; the current prompt has an explicit "CUT SEMANTICS" section up top to prevent that regression.
- Times must be in the source timeline, not the audition's compressed timeline.

The refined cut list runs through `sanitize_cut_list` → `render_audio_from_cuts`, and the new audition is returned in the same shape as `/render_auditions` with added `origin.optimized_from_output_id` / `origin.optimized_from_title` and a `changes` array. The frontend appends it as a new card next to the original (A/B comparison preserved); the new card is itself optimizable for recursive refinement.

### Rendering (`render_audio_from_cuts`)

Cuts are **regions to remove**, not regions to keep. The function inverts them into `keep_regions`, concatenates with `AudioSegment.append(..., crossfade=...)`, applies optional `fade_in`/`fade_out`, and exports via pydub. Crossfade is clamped to `min(crossfade_ms, len(out), len(clip))` to avoid pydub raising on short clips; on any append failure it falls back to a hard concat.

### Frontend (`templates/index.html`)

Single file, no bundler. wavesurfer.js v7 with `regions` + `timeline` plugins loaded from unpkg. State lives in plain JS variables; cuts are wavesurfer Region objects and are the source of truth — `getCuts()` reads them back at render time. Auditions are kept in a module-level `auditionsById` Map keyed by `output_id` so the optimize handler can reconstruct a plan's cut list for the request body.

Cache handling:
- Build id (`BUILD_ID` computed from app.py mtime + start time) is rendered into the page header and also exposed at `GET /build_id`. A small client-side self-check on load compares the two; if they mismatch, the pill turns red and a toast tells the user to hard-reload. This prevents silent "phantom" failures from cached JS after backend changes.
- Flask is configured with `TEMPLATES_AUTO_RELOAD = True`, so template edits take effect on the next request. **Python changes still need a server restart.**
- `GET /` sets `Cache-Control: no-store`.

### yt-dlp URL ingestion (`/download_url`)

Probe-then-download: `YoutubeDL.extract_info(url, download=False)` first so the 20-minute duration cap fails fast before any audio is fetched. Download-phase options include `"js_runtimes": {"node": {}}` (YouTube signature cipher); output is forced to MP3 via `FFmpegExtractAudio` so the rest of the pipeline doesn't have to branch on container format. Partial-file cleanup runs in the `except DownloadError` block (glob `{file_id}.*`).

## Conventions worth knowing

- **Time units**: API boundaries use seconds (float, 2 decimals). Internally, pydub works in ms — always multiply/divide by 1000 at the boundary. librosa works in frames; use `librosa.frames_to_time` at the boundary.
- **Cut semantics**: `cuts[i] = {start, end}` means "remove this range". The sum of durations across cuts equals what's removed, not what's kept. Any prompt that surfaces cuts to an LLM must restate this to avoid the keep/remove inversion.
- **Upload limit**: `MAX_UPLOAD_MB = 50`, enforced via `app.config["MAX_CONTENT_LENGTH"]` for POST bodies and as a post-download size check for yt-dlp output.
- **No auth, no rate limiting, no workspace cleanup** — this is designed for local/single-user use. Be careful about exposing it publicly.
- **matplotlib must use the `Agg` backend** (set inside `generate_waveform_image`) — any other backend will fail under gunicorn.
- **Flask threading**: the dev server must run with `threaded=True`; in production, gunicorn's `--threads 4` (set in the Dockerfile) serves the same role. The NDJSON streaming endpoints spawn a worker thread for the AI call; on a single-threaded runtime the heartbeat loop and the worker share a thread and the endpoint appears to hang.
