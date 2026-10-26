import os, re, time, base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Text / agent tools / judge
TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b")
# Used automatically when a request contains an image
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b")
_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                         base_url="https://api.groq.com/openai/v1")
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


def _has_image(messages):
    for m in messages:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in c):
            return True
    return False


def _strip_think(text):
    """Remove <think>...</think> blocks some reasoning models add."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def chat(messages, tools=None, temperature=0.7, json_mode=False, retries=3):
    """Low-level call. Returns the raw message (use for tool calling)."""
    model = VISION_MODEL if _has_image(messages) else TEXT_MODEL
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for i in range(retries):
        try:
            r = get_client().chat.completions.create(**kwargs)
            msg = r.choices[0].message
            if msg.content:
                msg.content = _strip_think(msg.content)
            return msg
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