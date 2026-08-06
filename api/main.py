from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os

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
        return {"transcription": "", "translation": ""}
    transcription = await transcribe(audio_bytes, source_lang)
    if not transcription:
        return {"transcription": "", "translation": ""}
    translation = await translate(transcription, target_lang)
    return {"transcription": transcription, "translation": translation}

async def transcribe(audio_bytes, language):
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo",
                headers=headers,
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language": language},
            )
            if r.status_code == 503:
                return ""
            r.raise_for_status()
            return r.json().get("text", "").strip()
    except Exception as e:
        print(f"Transcribe error: {e}")
        return ""

async def translate(text, target_lang):
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://translate.googleapis.com/translate_a/single", params={"client":"gtx","sl":"auto","tl":target_lang,"dt":"t","q":text})
            r.raise_for_status()
            d = r.json()
            if d and d[0]: return "".join(i[0] for i in d[0])
    except: pass
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://api.mymemory.translated.net/get", params={"q": text, "langpair": f"auto|{target_lang}"})
            return r.json()["responseData"]["translatedText"]
    except: return text