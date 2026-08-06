from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import asyncio

app = FastAPI(title="VidLingo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_TOKEN = os.getenv("HF_TOKEN", "")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/process")
async def process_audio(file: UploadFile = File(...), source_lang: str = Form("en"), target_lang: str = Form("ar")):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"transcription": "", "translation": "", "error": "no_audio"}
    transcription = await transcribe(audio_bytes, source_lang)
    if not transcription:
        return {"transcription": "", "translation": "", "error": "transcription_failed"}
    translation = await translate(transcription, target_lang)
    return {"transcription": transcription, "translation": translation, "error": None}

async def transcribe(audio_bytes, language):
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    # Try up to 2 times (HF often returns 503 when model is loading)
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=25) as c:
                r = await c.post(
                    "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
                    headers=headers,
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"language": language},
                )
                if r.status_code == 503:
                    # Model is loading, wait and retry
                    if attempt == 0:
                        await asyncio.sleep(3)
                        continue
                    return ""
                if r.status_code == 422:
                    # Bad request - audio too short or invalid
                    return ""
                r.raise_for_status()
                text = r.json().get("text", "").strip()
                return text
        except httpx.TimeoutException:
            print(f"Transcribe timeout (attempt {attempt+1})")
            if attempt == 1:
                return ""
        except Exception as e:
            print(f"Transcribe error (attempt {attempt+1}): {e}")
            return ""
    return ""

async def translate(text, target_lang):
    # Try Google Translate first
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text},
            )
            r.raise_for_status()
            d = r.json()
            if d and d[0]:
                return "".join(i[0] for i in d[0])
    except Exception as e:
        print(f"Google translate error: {e}")
    # Fallback to MyMemory
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": f"auto|{target_lang}"},
            )
            r.raise_for_status()
            return r.json()["responseData"]["translatedText"]
    except Exception as e:
        print(f"MyMemory error: {e}")
        return text
