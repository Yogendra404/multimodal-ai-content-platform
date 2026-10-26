import json
from .llm import chat

TOOL_LOG: list[dict] = []   # the eval harness reads this


def _log(name, **kw):
    TOOL_LOG.append({"tool": name, **kw})


def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Return the most frequent meaningful keywords from the text."""
    stop = set("the a an and or of to in on for with is are was were be this that it as at by from".split())
    words = [w.strip(".,!?:;()\"'").lower() for w in text.split()]
    freq = {}
    for w in words:
        if len(w) > 3 and w not in stop:
            freq[w] = freq.get(w, 0) + 1
    res = sorted(freq, key=freq.get, reverse=True)[:top_n]
    _log("extract_keywords", result=res)
    return res


def reading_time(text: str) -> str:
    """Estimate reading time of the text in seconds."""
    secs = round(len(text.split()) / 200 * 60)
    _log("reading_time", result=secs)
    return f"{secs} seconds"


def suggest_hashtags(topic: str) -> list[str]:
    """Suggest 5 hashtags for a topic."""
    tags = ["#" + w.capitalize() for w in topic.split() if len(w) > 3][:5]
    tags += ["#AI", "#Innovation"]
    _log("suggest_hashtags", result=tags[:5])
    return tags[:5]


TOOLS = [extract_keywords, reading_time, suggest_hashtags]
REGISTRY = {f.__name__: f for f in TOOLS}

# OpenAI needs a JSON description of each tool
TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "extract_keywords",
        "description": "Return the most frequent meaningful keywords from the text.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Text to analyse"},
            "top_n": {"type": "integer", "description": "How many keywords (default 5)"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "reading_time",
        "description": "Estimate the reading time of a text, in seconds.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "Text to measure"}},
            "required": ["text"]}}},
    {"type": "function", "function": {
        "name": "suggest_hashtags",
        "description": "Suggest 5 hashtags for a topic.",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "The topic"}},
            "required": ["topic"]}}},
]

SYSTEM = ("You are a content agent. Use your tools when they help: "
          "extract_keywords to ground the piece, reading_time to check length, "
          "suggest_hashtags for social posts. Be factual; if image details are "
          "flagged uncertain, do not state them as fact.")


def run_agent(request: str, context: str = "", max_steps: int = 5) -> dict:
    TOOL_LOG.clear()
    prompt = f"{context}\n\nTASK: {request}" if context else request
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt}]

    for _ in range(max_steps):
        msg = chat(messages, tools=TOOLS_SPEC)
        if not msg.tool_calls:
            return {"answer": msg.content or "", "tool_calls": list(TOOL_LOG)}

        # keep the assistant's tool request in the conversation
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        # run each requested tool and send the result back
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = REGISTRY[tc.function.name](**args)
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result)})

    # step limit reached: ask for the final text without tools
    msg = chat(messages)
    return {"answer": msg.content or "", "tool_calls": list(TOOL_LOG)}