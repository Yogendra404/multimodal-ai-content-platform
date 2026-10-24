import os, time
from google import genai
from google.genai import types

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_client = None


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def generate(contents, system=None, tools=None, temperature=0.7,
             json_mode=False, retries=3):
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        tools=tools,
        temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
    )
    for i in range(retries):
        try:
            r = get_client().models.generate_content(
                model=MODEL, contents=contents, config=cfg)
            return r.text or ""
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)  # backoff for 429/503