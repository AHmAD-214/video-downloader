from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import io
import speech_recognition as sr

app = FastAPI(title="VidLingo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/process")
async def process_audio(file: UploadFile = File(...), source_lang: str = Form("en"), target_lang: str = Form("ar")):
    audio_bytes = await file.read()
    if not audio_bytes:
        return {"transcription": "", "translation": ""}
    transcription = await asyncio.to_thread(_transcribe, audio_bytes, source_lang)
    if not transcription:
        return {"transcription": "", "translation": ""}
    translation = await _translate(transcription, target_lang)
    return {"transcription": transcription, "translation": translation}

def _transcribe(audio_bytes, language):
    recognizer = sr.Recognizer()
    try:
        with io.BytesIO(audio_bytes) as f:
            with sr.AudioFile(f) as source:
                audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data, language=language).strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Google Speech error: {e}")
        return ""
    except Exception as e:
        print(f"Transcribe error: {e}")
        return ""

async def _translate(text, target_lang):
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
    except:
        pass
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text, "langpair": f"auto|{target_lang}"},
            )
            return r.json()["responseData"]["translatedText"]
    except:
        return text
