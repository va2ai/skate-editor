FROM python:3.11-slim

# ffmpeg for pydub + yt-dlp audio extraction; nodejs for yt-dlp's YouTube
# signature solver (without a JS runtime, YouTube downloads 403).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

# Fly injects $PORT; fall back to 5000 for `docker run` locally.
# --timeout 600 covers Gemini multimodal calls (60–120s each).
EXPOSE 8080
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --timeout 600 --workers 1 --threads 4 app:app"]
