from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
import numpy as np
from faster_whisper import WhisperModel
import warnings, io, wave
warnings.filterwarnings("ignore")

app = FastAPI(title="VidLingo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None

@app.on_event("startup")
async def load_model():
    global model
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/process")
async def process_audio(file: UploadFile = File(...), source_lang: str = Form("en"), target_lang: str = Form("ar")):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"transcription": "", "translation": ""}
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
        audio_np = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_np, language=source_lang, beam_size=3)
    transcription = " ".join(s.text.strip() for s in segments if s.text.strip())
    if not transcription:
        return {"transcription": "", "translation": ""}
    translation = await translate(transcription, target_lang)
    return {"transcription": transcription, "translation": translation}

async def translate(text, target_lang):
    deepl_key = os.getenv("DEEPL_API_KEY", "")
    if deepl_key:
        r = await translate_deepl(text, target_lang, deepl_key)
        if r: return r
    return await translate_google(text, target_lang)

async def translate_deepl(text, target_lang, api_key):
    lang_map = {"ar":"AR","en":"EN","fr":"FR","es":"ES","de":"DE","tr":"TR","ja":"JA","ko":"KO","zh":"ZH","hi":"HI","pt":"PT","ru":"RU","it":"IT","ur":"UR"}
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post("https://api-free.deepl.com/v2/translate", data={"auth_key": api_key, "text": text, "target_lang": lang_map.get(target_lang, target_lang.upper())})
            r.raise_for_status()
            return r.json()["translations"][0]["text"]
    except: return ""

async def translate_google(text, target_lang):
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
