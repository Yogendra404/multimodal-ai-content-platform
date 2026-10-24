import io, re, json, os, asyncio, tempfile
import edge_tts
from PIL import Image
from google.genai import types
from .llm import generate


# ---------- IMAGE (multi-modal input) ----------
def prepare_image(raw: bytes) -> bytes:
    """Normalise any upload to a max-1024px JPEG (handles huge/odd files)."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def describe_image(raw: bytes) -> dict:
    jpg = prepare_image(raw)
    prompt = (
        "Analyse this image. Return JSON with keys: "
        "description (2-3 sentences), objects (list), text_in_image (string or ''), "
        "confidence ('high'|'medium'|'low'), uncertainties (list of things you "
        "cannot determine, e.g. blur, dark, cropped). Never guess details you cannot see."
    )
    out = generate([types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), prompt],
                   json_mode=True, temperature=0.2)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"description": out, "objects": [], "text_in_image": "",
                "confidence": "low", "uncertainties": ["unparseable model output"]}


# ---------- VOICE IN (speech-to-text) ----------
def transcribe(audio: bytes, mime="audio/wav") -> str:
    prompt = ("Transcribe this audio exactly. If it is silent, pure noise, or "
              "unintelligible, reply with exactly: UNCLEAR")
    text = generate([types.Part.from_bytes(data=audio, mime_type=mime), prompt],
                    temperature=0.0).strip()
    return "" if text.upper().startswith("UNCLEAR") or not text else text


# ---------- VOICE OUT (text-to-speech) ----------
def _clean_for_speech(t: str) -> str:
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[#*_`>~]", "", t)
    return re.sub(r"\s+", " ", t).strip()


async def _synth(text, voice, path):
    await edge_tts.Communicate(text, voice).save(path)


def speak(text: str, voice="en-US-AriaNeural") -> bytes:
    text = _clean_for_speech(text)[:1800]
    fd, path = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
    try:
        asyncio.run(_synth(text, voice, path))
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)