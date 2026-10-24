from .llm import generate

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

SYSTEM = ("You are a content agent. Use your tools when they help: "
          "extract_keywords to ground the piece, reading_time to check length, "
          "suggest_hashtags for social posts. Be factual; if image details are "
          "flagged uncertain, do not state them as fact.")


def run_agent(request: str, context: str = "") -> dict:
    TOOL_LOG.clear()
    prompt = f"{context}\n\nTASK: {request}" if context else request
    answer = generate(prompt, system=SYSTEM, tools=TOOLS)
    return {"answer": answer, "tool_calls": list(TOOL_LOG)}