from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Literal
import yt_dlp
import os
import tempfile
import asyncio
import random
import httpx
import time
from pathlib import Path

app = FastAPI(title="Video Downloader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Proxy Pool (Hydra) ---
PROXY_SOURCES = [
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/ErcinDedeworken/ProxyScraper/main/proxies.txt",
    "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text",
]

proxy_list: list[str] = []
proxy_last_fetched: float = 0
PROXY_REFRESH_INTERVAL = 600  # 10 minutes


async def fetch_proxies():
    """Fetch fresh proxies from multiple Hydra sources."""
    global proxy_list, proxy_last_fetched
    now = time.time()
    if now - proxy_last_fetched < PROXY_REFRESH_INTERVAL and proxy_list:
        return

    fresh_proxies: list[str] = []
    async with httpx.AsyncClient(timeout=15) as client:
        tasks = [client.get(url) for url in PROXY_SOURCES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            for line in result.text.strip().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    if not line.startswith("http"):
                        line = f"http://{line}"
                    fresh_proxies.append(line)

    random.shuffle(fresh_proxies)
    proxy_list = fresh_proxies
    proxy_last_fetched = now
    print(f"[Proxy] Fetched {len(proxy_list)} proxies")


def get_proxy() -> Optional[str]:
    """Get a random proxy from the pool."""
    if not proxy_list:
        return None
    return random.choice(proxy_list)


# --- Models ---
class DownloadRequest(BaseModel):
    url: str
    format: Literal["audio", "video", "merged"] = "merged"
    quality: Optional[str] = None


class DownloadStatus(BaseModel):
    status: str
    message: str
    filename: Optional[str] = None


# --- yt-dlp Config ---
TEMP_DIR = Path(tempfile.gettempdir()) / "video_downloads"
TEMP_DIR.mkdir(exist_ok=True)


def get_ydl_opts(format_type: str, quality: Optional[str], proxy: Optional[str]) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "max_filesize": 500 * 1024 * 1024,  # 500MB limit
        "extractor_retries": 3,
        "socket_timeout": 30,
        "http_chunk_size": 10485760,
    }

    if proxy:
        opts["proxy"] = proxy

    if format_type == "audio":
        opts["format"] = "bestaudio/ba"
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]
        opts["outtmpl"] = str(TEMP_DIR / "%(title).80s.%(ext)s")

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
        opts["outtmpl"] = str(TEMP_DIR / "%(title).80s.%(ext)s")

    elif format_type == "merged":
        opts["format"] = "bestvideo+bestaudio/best"
        opts["merge_output_format"] = "mp4"
        opts["outtmpl"] = str(TEMP_DIR / "%(title).80s.%(ext)s")

    return opts


async def download_with_retry(url: str, format_type: str, quality: Optional[str], max_retries: int = 3) -> str:
    """Download with proxy rotation on failure."""
    await fetch_proxies()

    last_error = None
    for attempt in range(max_retries):
        proxy = get_proxy() if proxy_list else None
        if proxy:
            print(f"[Attempt {attempt+1}] Using proxy: {proxy[:30]}...")
        else:
            print(f"[Attempt {attempt+1}] No proxy, direct connection")

        opts = get_ydl_opts(format_type, quality, proxy)

        try:
            info = await asyncio.to_thread(
                yt_dlp.YoutubeDL(opts).extract_info, url, download=True
            )
            if info:
                filename = yt_dlp.YoutubeDL({"outtmpl": opts["outtmpl"]}).prepare_filename(info)
                # Check if file exists (yt-dlp may change extension after post-processing)
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
            # Clean up partial files
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
    """Download video/audio from any supported platform."""
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        filepath = await download_with_retry(req.url, req.format, req.quality)
        filename = os.path.basename(filepath)

        # Determine content type
        ext = filepath.lower().split(".")[-1]
        content_types = {
            "mp3": "audio/mpeg",
            "mp4": "video/mp4",
            "webm": "video/webm",
            "m4a": "audio/mp4",
            "mkv": "video/x-matroska",
        }
        media_type = content_types.get(ext, "application/octet-stream")

        def cleanup():
            try:
                os.remove(filepath)
            except:
                pass

        return FileResponse(
            path=filepath,
            filename=filename,
            media_type=media_type,
            background=cleanup,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)[:200]}")


@app.get("/api/proxies")
async def proxy_status():
    """Check proxy pool status."""
    await fetch_proxies()
    return {
        "total_proxies": len(proxy_list),
        "last_refresh": proxy_last_fetched,
        "sources": len(PROXY_SOURCES),
    }


@app.get("/api/health")
async def health():
    """Health check endpoint (used by Worker keepalive)."""
    return {"status": "ok", "proxies": len(proxy_list)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
