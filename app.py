from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
import yt_dlp
import os
import tempfile
import asyncio
import time
from pathlib import Path

app = FastAPI(title="Video Downloader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(tempfile.gettempdir()) / "video_downloads"
TEMP_DIR.mkdir(exist_ok=True)


class DownloadRequest(BaseModel):
    url: str
    format: Literal["audio", "video", "merged"] = "merged"
    quality: Optional[str] = None


def get_ydl_opts(format_type: str, quality: Optional[str]) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 500 * 1024 * 1024,
        "extractor_retries": 3,
        "socket_timeout": 30,
        "http_chunk_size": 10485760,
    }

    if format_type == "audio":
        opts["format"] = "bestaudio/ba"
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]
    elif format_type == "video":
        if quality == "360":
            opts["format"] = "bestvideo[height<=360]+bestaudio/best[height<=360]"
        elif quality == "480":
            opts["format"] = "bestvideo[height<=480]+bestaudio/best[height<=480]"
        elif quality == "720":
            opts["format"] = "bestvideo[height<=720]+bestaudio/best[height<=720]"
        else:
            opts["format"] = "bestvideo+bestaudio/best"
        opts["merge_output_format"] = "mp4"
    elif format_type == "merged":
        opts["format"] = "bestvideo+bestaudio/best"
        opts["merge_output_format"] = "mp4"

    opts["outtmpl"] = str(TEMP_DIR / "%(title).80s.%(ext)s")
    return opts


async def download_file(url: str, format_type: str, quality: Optional[str], max_retries: int = 3) -> str:
    last_error = None
    for attempt in range(max_retries):
        opts = get_ydl_opts(format_type, quality)
        try:
            info = await asyncio.to_thread(
                yt_dlp.YoutubeDL(opts).extract_info, url, download=True
            )
            if info:
                filename = yt_dlp.YoutubeDL({"outtmpl": opts["outtmpl"]}).prepare_filename(info)
                if not os.path.exists(filename):
                    base = os.path.splitext(filename)[0]
                    for ext in [".mp3", ".mp4", ".webm", ".m4a", ".mkv"]:
                        if os.path.exists(base + ext):
                            filename = base + ext
                            break
                if os.path.exists(filename):
                    return filename
                raise FileNotFoundError("Downloaded file not found")
        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt+1}] Failed: {str(e)[:100]}")
            for f in TEMP_DIR.iterdir():
                if f.is_file() and f.stat().st_mtime > time.time() - 120:
                    try:
                        f.unlink()
                    except:
                        pass

    raise HTTPException(status_code=502, detail=f"Download failed after {max_retries} attempts: {str(last_error)[:200]}")


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return {"service": "Video Downloader", "status": "running"}


@app.post("/api/download")
async def download_video(req: DownloadRequest):
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        filepath = await download_file(req.url, req.format, req.quality)
        filename = os.path.basename(filepath)
        ext = filepath.lower().split(".")[-1]
        media_types = {
            "mp3": "audio/mpeg", "mp4": "video/mp4",
            "webm": "video/webm", "m4a": "audio/mp4", "mkv": "video/x-matroska",
        }
        media_type = media_types.get(ext, "application/octet-stream")

        def cleanup():
            try: os.remove(filepath)
            except: pass

        return FileResponse(path=filepath, filename=filename, media_type=media_type, background=cleanup)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)[:200]}")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
