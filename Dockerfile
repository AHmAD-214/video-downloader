FROM python:3.12-slim

# Install FFmpeg and yt-dlp
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    pip install --no-cache-dir yt-dlp && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html .

ENV PORT=10000

EXPOSE 10000

CMD ["python", "app.py"]
