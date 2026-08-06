import os
import base64
import tempfile
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="VidLingo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found!")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")


class TranscribeRequest(BaseModel):
    audio_base64: str
    source_lang: str = "auto"
    target_lang: str = "ar"


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "ar"


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "groq_ready": groq_client is not None}


@app.post("/transcribe")
async def transcribe_and_translate(request: TranscribeRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing")
    try:
        audio_data = base64.b64decode(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio")

    if len(audio_data) < 1000:
        return {"text": "", "translation": "", "error": "Audio too short"}

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        with open(temp_path, "rb") as audio_file:
            lang_param = None if request.source_lang == "auto" else request.source_lang
            transcription = groq_client.audio.transcriptions.create(
                model=MODEL,
                file=audio_file,
                language=lang_param,
                response_format="text",
            )

        if not transcription or not transcription.strip():
            return {"text": "", "translation": "", "error": "No speech detected"}

        clean_text = transcription.strip()
        translation = await translate_text(clean_text, request.target_lang)
        return {"text": clean_text, "translation": translation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass


@app.post("/translate")
async def translate_only(request: TranslateRequest):
    if not request.text or not request.text.strip():
        return {"translation": ""}
    translation = await translate_text(request.text, request.target_lang)
    return {"translation": translation}


async def translate_text(text: str, target_lang: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            params = {"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text}
            resp = await client.get("https://translate.googleapis.com/translate_a/single", params=params, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                translated = "".join([item[0] for item in data[0] if isinstance(item, list) and item[0]])
                if translated and translated.strip():
                    return translated.strip()
    except Exception:
        pass

    try:
        async with httpx.AsyncClient() as client:
            params = {"q": text, "langpair": f"en|{target_lang}"}
            resp = await client.get("https://api.mymemory.translated.net/get", params=params, timeout=15.0)
            if resp.status_code == 200:
                data = resp.json()
                translated = data.get("responseData", {}).get("translatedText", "")
                if translated and "NO QUERY SPECIFIED" not in translated.upper():
                    return translated.strip()
    except Exception:
        pass

    return text


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
