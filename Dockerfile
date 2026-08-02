FROM python:3.12-slim

WORKDIR /app

# aiortc (via pipecat's webrtc extra) needs these at runtime for audio/video codecs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    libvpx-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Railway/Fly all inject PORT at runtime; server.py already reads it.
CMD ["python", "server.py"]

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
