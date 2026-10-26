import os, time, base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # supports text + images
_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def image_part(image_bytes, mime="image/jpeg"):
    """Build an image item you can put inside the `contents` list."""
    b64 = base64.b64encode(image_bytes).decode()
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _to_content(contents):
    """Accepts a string, or a list of strings / image_part() dicts."""
    if isinstance(contents, str):
        return contents
    parts = []
    for c in contents:
        if isinstance(c, str):
            parts.append({"type": "text", "text": c})
        else:
            parts.append(c)
    return parts


def chat(messages, tools=None, temperature=0.7, json_mode=False, retries=3):
    """Low-level call. Returns the raw message (use for tool calling)."""
    kwargs = dict(model=MODEL, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for i in range(retries):
        try:
            r = get_client().chat.completions.create(**kwargs)
            return r.choices[0].message
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i)  # backoff for 429/5xx


def generate(contents, system=None, tools=None, temperature=0.7,
             json_mode=False, retries=3):
    """Same signature as before; returns the reply text."""
    if json_mode:
        system = (system or "") + "\nRespond only with valid JSON."
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _to_content(contents)})
    msg = chat(messages, tools=tools, temperature=temperature,
               json_mode=json_mode, retries=retries)
    return msg.content or ""